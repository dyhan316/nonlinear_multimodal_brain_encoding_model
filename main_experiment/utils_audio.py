from tqdm import tqdm
import torch
import torchaudio
from transformers import AutoModel, AutoModelForPreTraining, PreTrainedModel,\
                         AutoFeatureExtractor, WhisperModel
from typing import Dict, Iterable, List, Optional
from pathlib import Path
import collections
import itertools
import numpy as np

TARGET_SAMPLE_RATE = 16000

def extract_speech_features(model: PreTrainedModel, model_config: dict, wav: torch.Tensor,
                            chunksz_sec: float, contextsz_sec: float,
                            num_sel_frames = 1, frame_skip = 5, sel_layers: Optional[List[int]]=None,
                            batchsz: int = 1,
                            return_numpy: bool = True, move_to_cpu: bool = True,
                            disable_tqdm: bool = False, feature_extractor=None,
                            sampling_rate: int = TARGET_SAMPLE_RATE, require_full_context: bool = False,
                            stereo: bool = False):
    
    assert (num_sel_frames == 1), f"'num_sel_frames` must be 1 to ensure causal feature extraction, but got {num_sel_frames}. "\
        "This option will be deprecated in the future."
    if stereo:
        raise NotImplementedError("stereo not implemented")
    else:
        assert wav.ndim == 1, f"input `wav` must be 1-D but got {wav.ndim}"
    if return_numpy: assert move_to_cpu, "'move_to_cpu' must be true if returning numpy arrays"

    # Whisper needs special handling
    is_whisper_model = isinstance(model, WhisperModel)

    # Compute chunks & context sizes in terms of samples & context
    chunksz_samples = int(chunksz_sec * sampling_rate)
    contextsz_samples = int(contextsz_sec * sampling_rate)

    # `snippet_ends` has the last (exclusive) sample for each snippet
    snippet_ends = []
    if not require_full_context:
        # Add all snippets that are _less_ than the total input size
        # (context+chunk)
        snippet_ends.append(torch.arange(chunksz_samples, contextsz_samples+chunksz_samples, chunksz_samples))

    # Add all snippets that are exactly the length of the requested input
    # (`Tensor.unfold` is basically a sliding window).
    if wav.shape[0] >= chunksz_samples+contextsz_samples:
        # `unfold` fails if `wav.shape[0]` is less than the window size.
        snippet_ends.append(
            torch.arange(wav.shape[0]).unfold(0, chunksz_samples+contextsz_samples, chunksz_samples)[:,-1]+1
        )

    snippet_ends = torch.cat(snippet_ends, dim=0) # shape: (num_snippets,)

    if snippet_ends.shape[0] == 0:
        raise ValueError(f"No snippets possible! Stimulus is probably too short ({wav.shape[0]} samples). Consider reducing context size or setting `require_full_context=True`")

    # 2-D array where `[i,0]` and `[i,1]` are the start and end, respectively,
    # of snippet `i` in samples. Shape: (num_snippets, 2)
    snippet_times = torch.stack([torch.maximum(torch.zeros_like(snippet_ends),
                                               snippet_ends-(contextsz_samples+chunksz_samples)),
                                 snippet_ends], dim=1)

    # Remove snippets that are not long enough. (Seems easier to filter
    # after generating the snippet bounds than handling it above in each case)
    # TODO: is there any way to programatically check this in HuggingFace?
    # doesn't seem so (unlike s3prl).
    if 'min_input_length' in model_config:
        # this is stored originally in **samples**!!!
        min_length_samples = model_config['min_input_length']
    elif 'win_ms' in model.config:
        min_length_samples = model.config['win_ms'] / 1000. * TARGET_SAMPLE_RATE

    snippet_times = snippet_times[(snippet_times[:,1] - snippet_times[:,0]) >= min_length_samples]
    snippet_times_sec = snippet_times / sampling_rate # snippet_times, but in sec.

    module_features = collections.defaultdict(list) 
    out_features = [] # the final output of the model
    times = [] # times are shared across all layers

    #assert (frames_per_chunk % frame_skip) == 0, "These must be divisible"
    frame_len_sec = model_config['stride'] / TARGET_SAMPLE_RATE # length of an output frame (sec.)

    snippet_length_samples = snippet_times[:,1] - snippet_times[:,0] # shape: (num_snippets,)
    if require_full_context:
        assert all(snippet_length_samples == snippet_length_samples[0]), "uneven snippet lengths!"
        snippet_length_samples = snippet_length_samples[0]
        assert snippet_length_samples.ndim == 0

    # Set up the iterator over batches of snippets
    if require_full_context:
        # This case is simpler, so handle it explicitly
        snippet_batches = snippet_times.T.split(batchsz, dim=1)
    else:
        # First, batch the snippets that are of different lengths.
        snippet_batches = snippet_times.tensor_split(torch.where(snippet_length_samples.diff() != 0)[0]+1, dim=0)
        # Then, split any batches that are too big to fit into the given
        # batch size.
        snippet_iter = []
        for batch in snippet_batches:
            # split, *then* transpose
            if batch.shape[0] > batchsz:
                snippet_iter += batch.T.split(batchsz,dim=1)
            else:
                snippet_iter += [batch.T]
        snippet_batches = snippet_iter

    snippet_iter = snippet_batches
    if not disable_tqdm:
        snippet_iter = tqdm(snippet_iter, desc='snippet batches', leave=False)
    snippet_iter = enumerate(snippet_iter)


    # Iterate with a sliding window. stride = chunk_sz
    for batch_idx, (snippet_starts, snippet_ends) in snippet_iter:
        if ((snippet_ends - snippet_starts) < (contextsz_samples + chunksz_samples)).any() and require_full_context:
            raise ValueError("This shouldn't happen with require_full_context")

        # If we don't have enough samples, skip this chunk.
        if (snippet_ends - snippet_starts < min_length_samples).any():
            print('If this is true for any, then you might be losing more snippets than just the offending (too short) snippet')
            assert False

        # Construct the input waveforms for the batch
        batched_wav_in_list = []
        for batch_snippet_idx, (snippet_start, snippet_end) in enumerate(zip(snippet_starts, snippet_ends)):
            # Stacking might be inefficient, so populate a pre-allocated array.
            #batched_wav_in[batch_snippet_idx, :] = wav[snippet_start:snippet_end]
            # But stacking makes variable batch size easier!
            batched_wav_in_list.append(wav[snippet_start:snippet_end])
        batched_wav_in = torch.stack(batched_wav_in_list, dim=0)

        # The final batch may be incomplete if batchsz doesn't evenly divide
        # the number of snippets.
        if (snippet_starts.shape[0] != batched_wav_in.shape[0]) and (snippet_starts.shape[0] != batchsz):
            batched_wav_in = batched_wav_in[:snippet_starts.shape[0]]

        # Take the last 1 or 2 activations, and time-wise put it at the
        # end of chunk.
        output_inds = np.array([-1 - frame_skip*i for i in reversed(range(num_sel_frames))])

        # Use a pre-processor if given (e.g. to normalize the waveform), and
        # then feed into the model.
        if feature_extractor is not None:
            # This step seems to be NOT differentiable, since the feature
            # extractor first converts the Tensor to a numpy array, then back
            # into a Tensor.
            # If you want to backprop through the stimulus, you might have to
            # re-implement the feature extraction in PyTorch (in particular, the
            # normalization)

            if stereo: raise NotImplementedError("Support handling multi-channel audio with feature extractor")
            # It looks like most feature extractors (e.g.
            # Wav2Vec2FeatureExtractor) accept mono audio (i.e. 1-dimensional),
            # but it's unclear if they support stereo as well.

            feature_extractor_kwargs = {}
            if is_whisper_model:
                # Because Whisper auto-pads all inputs to 30 sec., we'll use
                # the attention mask to figure out when the "last" relevant
                # input was.
                features_key = 'input_features'
                feature_extractor_kwargs['return_attention_mask'] = True
            else:
                features_key = 'input_values'

            preprocessed_snippets = feature_extractor(list(batched_wav_in.cpu().numpy()),
                                                      return_tensors='pt',
                                                      sampling_rate=sampling_rate,
                                                      **feature_extractor_kwargs)
            if is_whisper_model:
                chunk_features = model.encoder(preprocessed_snippets[features_key].to(model.device))

                # Now we need to figure out which output index to use, since 2
                # conv layers downsample the inputs before passing them into
                # the encoder's Transformer layers. We can redo the encoder's
                # 1-D conv's on the attention mask to find the final output that
                # was influenced by the snippet.
                contributing_outs = preprocessed_snippets.attention_mask # 1 if part of waveform, 0 otherwise. shape: (batchsz, 3000)
                # Taking [0] works because all snippets have the same length.
                # Add the dimension back for `conv1d` to work
                # TODO: assert that all clips are the same length?
                contributing_outs = contributing_outs[0].unsqueeze(0)

                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+model.encoder.conv1.kernel_size).to(contributing_outs),
                                                               stride=model.encoder.conv1.stride,
                                                               padding=model.encoder.conv1.padding,
                                                               dilation=model.encoder.conv1.dilation,
                                                               groups=model.encoder.conv1.groups)
                # shape: (batchsz, 1500)
                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+model.encoder.conv2.kernel_size).to(contributing_outs),
                                                               stride=model.encoder.conv2.stride,
                                                               padding=model.encoder.conv2.padding,
                                                               dilation=model.encoder.conv2.dilation,
                                                               groups=model.encoder.conv1.groups)

                final_output = contributing_outs[0].nonzero().squeeze(-1).max()
            else:
                # sampling rates must match if not using a pre-processor
                assert sampling_rate == TARGET_SAMPLE_RATE, f"sampling rate mismatch! {sampling_rate} != {TARGET_SAMPLE_RATE}"

                chunk_features = model(preprocessed_snippets[features_key].to(model.device))
        else:
            chunk_features = model(batched_wav_in)

        # Make sure we have enough outputs
        if(chunk_features['last_hidden_state'].shape[1] < (num_sel_frames-1) * frame_skip - 1):
            print("Skipping:", snippet_idx, "only had", chunk_features['last_hidden_state'].shape[1],
                    "outputs, whereas", (num_sel_frames-1) * frame_skip - 1, "were needed.")
            continue

        assert len(output_inds) == 1, "Only one output per evaluation is "\
            "supported for Hugging Face (because they don't provide the downsampling rate)"

        if is_whisper_model:
            output_inds = [final_output]

        for out_idx, output_offset in enumerate(output_inds):
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1))

            output_representation = chunk_features['last_hidden_state'][:, output_offset, :] # shape: (batchsz, hidden_size)
            if move_to_cpu: output_representation = output_representation.cpu()
            if return_numpy: output_representation = output_representation.numpy()
            out_features.append(output_representation)

            # Collect features from individual layers
            # NOTE: outs['hidden_states'] might have an extra element at
            # the beginning for the feature extractor.
            # e.g. 25 "layers" --> CNN output + 24 transformer layers' output
            #! below : changed to see if it works #! extra layer is added at the BEGINNING!!
            #! TODO : sel_layers받아서 실제로 받아오도록 한후에 해보기! 일단은 audio config를 고치러 감!
            for layer_idx, layer_activations in enumerate(chunk_features['hidden_states']):
                # Only save layers that the user wants (if specified)
                if sel_layers: #! none, so skipped!
                    if layer_idx not in sel_layers: continue

                layer_representation = layer_activations[:, output_offset, :] # shape: (batchsz, hidden_size)
                if move_to_cpu: layer_representation = layer_representation.cpu()
                if return_numpy: layer_representation = layer_representation.numpy() # TODO: convert to numpy at the end

                if is_whisper_model:
                    # Leave the option open for using decoder layers in the
                    # future
                    module_name = f"encoder.{layer_idx}"
                else:
                    module_name = f"layer.{layer_idx}"

                module_features[module_name].append(layer_representation)

    out_features = np.concatenate(out_features, axis=0) if return_numpy else torch.cat(out_features, dim=0) # shape: (timesteps, features)
    
    module_features = {name: (np.concatenate(features, axis=0) if return_numpy else torch.cat(features, dim=0))\
                       for name, features in module_features.items()}

    assert all(features.shape[0] == out_features.shape[0] for features in module_features.values()),\
        "Missing timesteps in the module activations!! (possible PyTorch bug)"
    #===
    times = torch.cat(times, dim=0) / TARGET_SAMPLE_RATE # convert samples --> seconds. shape: (timesteps,)
    if return_numpy: times = times.numpy()

    del chunk_features # possible memory leak. remove if unneeded
    
    return {'final_outputs': out_features, 'times': times, 'module_features': module_features}


def forward_speech_model(model: PreTrainedModel, model_config: dict, wav: torch.Tensor,
                            chunksz_sec: float, contextsz_sec: float,
                            num_sel_frames = 1, frame_skip = 5, sel_layers: Optional[List[int]]=None,
                            batchsz: int = 1,
                            return_numpy: bool = True, move_to_cpu: bool = False,
                            disable_tqdm: bool = False, feature_extractor=None,
                            sampling_rate: int = TARGET_SAMPLE_RATE, require_full_context: bool = False,
                            stereo: bool = False):
    
    assert (num_sel_frames == 1), f"'num_sel_frames` must be 1 to ensure causal feature extraction, but got {num_sel_frames}. "\
        "This option will be deprecated in the future."
    if stereo:
        raise NotImplementedError("stereo not implemented")
    else:
        assert wav.ndim == 1, f"input `wav` must be 1-D but got {wav.ndim}"
    if return_numpy: assert move_to_cpu, "'move_to_cpu' must be true if returning numpy arrays"

    # Whisper needs special handling
    is_whisper_model = isinstance(model, WhisperModel)

    # Compute chunks & context sizes in terms of samples & context
    chunksz_samples = int(chunksz_sec * sampling_rate)
    contextsz_samples = int(contextsz_sec * sampling_rate)

    # `snippet_ends` has the last (exclusive) sample for each snippet
    snippet_ends = []
    if not require_full_context:
        # Add all snippets that are _less_ than the total input size
        # (context+chunk)
        snippet_ends.append(torch.arange(chunksz_samples, contextsz_samples+chunksz_samples, chunksz_samples))

    # Add all snippets that are exactly the length of the requested input
    # (`Tensor.unfold` is basically a sliding window).
    if wav.shape[0] >= chunksz_samples+contextsz_samples:
        # `unfold` fails if `wav.shape[0]` is less than the window size.
        snippet_ends.append(
            torch.arange(wav.shape[0]).unfold(0, chunksz_samples+contextsz_samples, chunksz_samples)[:,-1]+1
        )

    snippet_ends = torch.cat(snippet_ends, dim=0) # shape: (num_snippets,)

    if snippet_ends.shape[0] == 0:
        raise ValueError(f"No snippets possible! Stimulus is probably too short ({wav.shape[0]} samples). Consider reducing context size or setting `require_full_context=True`")

    # 2-D array where `[i,0]` and `[i,1]` are the start and end, respectively,
    # of snippet `i` in samples. Shape: (num_snippets, 2)
    snippet_times = torch.stack([torch.maximum(torch.zeros_like(snippet_ends),
                                               snippet_ends-(contextsz_samples+chunksz_samples)),
                                 snippet_ends], dim=1)

    # Remove snippets that are not long enough. (Seems easier to filter
    # after generating the snippet bounds than handling it above in each case)
    # TODO: is there any way to programatically check this in HuggingFace?
    # doesn't seem so (unlike s3prl).
    if 'min_input_length' in model_config:
        # this is stored originally in **samples**!!!
        min_length_samples = model_config['min_input_length']
    elif 'win_ms' in model.config:
        min_length_samples = model.config['win_ms'] / 1000. * TARGET_SAMPLE_RATE

    snippet_times = snippet_times[(snippet_times[:,1] - snippet_times[:,0]) >= min_length_samples]
    snippet_times_sec = snippet_times / sampling_rate # snippet_times, but in sec.

    module_features = collections.defaultdict(list) 
    out_features = [] # the final output of the model
    times = [] # times are shared across all layers

    #assert (frames_per_chunk % frame_skip) == 0, "These must be divisible"
    frame_len_sec = model_config['stride'] / TARGET_SAMPLE_RATE # length of an output frame (sec.)

    snippet_length_samples = snippet_times[:,1] - snippet_times[:,0] # shape: (num_snippets,)
    if require_full_context:
        assert all(snippet_length_samples == snippet_length_samples[0]), "uneven snippet lengths!"
        snippet_length_samples = snippet_length_samples[0]
        assert snippet_length_samples.ndim == 0

    # Set up the iterator over batches of snippets
    if require_full_context:
        # This case is simpler, so handle it explicitly
        snippet_batches = snippet_times.T.split(batchsz, dim=1)
    else:
        # First, batch the snippets that are of different lengths.
        snippet_batches = snippet_times.tensor_split(torch.where(snippet_length_samples.diff() != 0)[0]+1, dim=0)
        # Then, split any batches that are too big to fit into the given
        # batch size.
        snippet_iter = []
        for batch in snippet_batches:
            # split, *then* transpose
            if batch.shape[0] > batchsz:
                snippet_iter += batch.T.split(batchsz,dim=1)
            else:
                snippet_iter += [batch.T]
        snippet_batches = snippet_iter

    snippet_iter = snippet_batches
    if not disable_tqdm:
        snippet_iter = tqdm(snippet_iter, desc='snippet batches', leave=False)
    snippet_iter = enumerate(snippet_iter)


    # Iterate with a sliding window. stride = chunk_sz
    for batch_idx, (snippet_starts, snippet_ends) in snippet_iter:
        if ((snippet_ends - snippet_starts) < (contextsz_samples + chunksz_samples)).any() and require_full_context:
            raise ValueError("This shouldn't happen with require_full_context")

        # If we don't have enough samples, skip this chunk.
        if (snippet_ends - snippet_starts < min_length_samples).any():
            print('If this is true for any, then you might be losing more snippets than just the offending (too short) snippet')
            assert False

        # Construct the input waveforms for the batch
        batched_wav_in_list = []
        for batch_snippet_idx, (snippet_start, snippet_end) in enumerate(zip(snippet_starts, snippet_ends)):
            # Stacking might be inefficient, so populate a pre-allocated array.
            #batched_wav_in[batch_snippet_idx, :] = wav[snippet_start:snippet_end]
            # But stacking makes variable batch size easier!
            batched_wav_in_list.append(wav[snippet_start:snippet_end])
        batched_wav_in = torch.stack(batched_wav_in_list, dim=0)

        # The final batch may be incomplete if batchsz doesn't evenly divide
        # the number of snippets.
        if (snippet_starts.shape[0] != batched_wav_in.shape[0]) and (snippet_starts.shape[0] != batchsz):
            batched_wav_in = batched_wav_in[:snippet_starts.shape[0]]

        # Take the last 1 or 2 activations, and time-wise put it at the
        # end of chunk.
        output_inds = np.array([-1 - frame_skip*i for i in reversed(range(num_sel_frames))])

        # Use a pre-processor if given (e.g. to normalize the waveform), and
        # then feed into the model.
        if feature_extractor is not None:
            # This step seems to be NOT differentiable, since the feature
            # extractor first converts the Tensor to a numpy array, then back
            # into a Tensor.
            # If you want to backprop through the stimulus, you might have to
            # re-implement the feature extraction in PyTorch (in particular, the
            # normalization)

            if stereo: raise NotImplementedError("Support handling multi-channel audio with feature extractor")
            # It looks like most feature extractors (e.g.
            # Wav2Vec2FeatureExtractor) accept mono audio (i.e. 1-dimensional),
            # but it's unclear if they support stereo as well.

            feature_extractor_kwargs = {}
            if is_whisper_model:
                # Because Whisper auto-pads all inputs to 30 sec., we'll use
                # the attention mask to figure out when the "last" relevant
                # input was.
                features_key = 'input_features'
                feature_extractor_kwargs['return_attention_mask'] = True
            else:
                features_key = 'input_values'

            preprocessed_snippets = feature_extractor(list(batched_wav_in.cpu().numpy()),
                                                      return_tensors='pt',
                                                      sampling_rate=sampling_rate,
                                                      **feature_extractor_kwargs)
            if is_whisper_model:
                chunk_features = model.encoder(preprocessed_snippets[features_key].to(model.device))
                import pdb ; pdb.set_trace()
                # Now we need to figure out which output index to use, since 2
                # conv layers downsample the inputs before passing them into
                # the encoder's Transformer layers. We can redo the encoder's
                # 1-D conv's on the attention mask to find the final output that
                # was influenced by the snippet.
                contributing_outs = preprocessed_snippets.attention_mask # 1 if part of waveform, 0 otherwise. shape: (batchsz, 3000)
                # Taking [0] works because all snippets have the same length.
                # Add the dimension back for `conv1d` to work
                # TODO: assert that all clips are the same length?
                contributing_outs = contributing_outs[0].unsqueeze(0)

                # import pdb; pdb.set_trace()
                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+model.encoder.conv1.kernel_size).to(contributing_outs),
                                                               stride=model.encoder.conv1.stride,
                                                               padding=model.encoder.conv1.padding,
                                                               dilation=model.encoder.conv1.dilation,
                                                               groups=model.encoder.conv1.groups)
                # shape: (batchsz, 1500)
                contributing_outs = torch.nn.functional.conv1d(contributing_outs,
                                                               torch.ones((1,1)+model.encoder.conv2.kernel_size).to(contributing_outs),
                                                               stride=model.encoder.conv2.stride,
                                                               padding=model.encoder.conv2.padding,
                                                               dilation=model.encoder.conv2.dilation,
                                                               groups=model.encoder.conv1.groups)

                final_output = contributing_outs[0].nonzero().squeeze(-1).max()
            else:
                # sampling rates must match if not using a pre-processor
                assert sampling_rate == TARGET_SAMPLE_RATE, f"sampling rate mismatch! {sampling_rate} != {TARGET_SAMPLE_RATE}"

                chunk_features = model(preprocessed_snippets[features_key].to(model.device))
        else:
            chunk_features = model(batched_wav_in)

        # Make sure we have enough outputs
        if(chunk_features['last_hidden_state'].shape[1] < (num_sel_frames-1) * frame_skip - 1):
            print("Skipping:", snippet_idx, "only had", chunk_features['last_hidden_state'].shape[1],
                    "outputs, whereas", (num_sel_frames-1) * frame_skip - 1, "were needed.")
            continue

        assert len(output_inds) == 1, "Only one output per evaluation is "\
            "supported for Hugging Face (because they don't provide the downsampling rate)"

        if is_whisper_model:
            output_inds = [final_output]

        for out_idx, output_offset in enumerate(output_inds):
            times.append(torch.stack([snippet_starts, snippet_ends], dim=1))

            output_representation = chunk_features['last_hidden_state'][:, output_offset, :] # shape: (batchsz, hidden_size)
            if move_to_cpu: output_representation = output_representation.cpu()
            if return_numpy: output_representation = output_representation.numpy()
            out_features.append(output_representation)

            # Collect features from individual layers
            # NOTE: outs['hidden_states'] might have an extra element at
            # the beginning for the feature extractor.
            # e.g. 25 "layers" --> CNN output + 24 transformer layers' output
            #! below : changed to see if it works #! extra layer is added at the BEGINNING!!
            #! TODO : sel_layers받아서 실제로 받아오도록 한후에 해보기! 일단은 audio config를 고치러 감!
            # for layer_idx, layer_activations in enumerate(chunk_features['hidden_states']):
            #     # Only save layers that the user wants (if specified)
            #     if sel_layers: #! none, so skipped!
            #         if layer_idx not in sel_layers: continue

            #     layer_representation = layer_activations[:, output_offset, :] # shape: (batchsz, hidden_size)
            #     if move_to_cpu: layer_representation = layer_representation.cpu()
            #     if return_numpy: layer_representation = layer_representation.numpy() # TODO: convert to numpy at the end

            #     if is_whisper_model:
            #         # Leave the option open for using decoder layers in the
            #         # future
            #         module_name = f"encoder.{layer_idx}"
            #     else:
            #         module_name = f"layer.{layer_idx}"

            #     module_features[module_name].append(layer_representation)

    out_features = np.concatenate(out_features, axis=0) if return_numpy else torch.cat(out_features, dim=0) # shape: (timesteps, features)
    
    # module_features = {name: (np.concatenate(features, axis=0) if return_numpy else torch.cat(features, dim=0))\
    #                    for name, features in module_features.items()}

    # assert all(features.shape[0] == out_features.shape[0] for features in module_features.values()),\
    #     "Missing timesteps in the module activations!! (possible PyTorch bug)"
    #===
    times = torch.cat(times, dim=0) / TARGET_SAMPLE_RATE # convert samples --> seconds. shape: (timesteps,)
    if return_numpy: times = times.numpy()

    del chunk_features # possible memory leak. remove if unneeded
    return {'final_outputs': out_features, 'times': times}      
    # return {'final_outputs': out_features, 'times': times, 'module_features': module_features}
