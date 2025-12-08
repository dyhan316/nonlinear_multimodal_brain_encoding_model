# Generate MODEL_LAYER_LIST from models_list.txt:
MODEL_LAYER_LIST=()
if [ ! -f scripts/models_list.txt ]; then
    echo "models_list.txt not found, skipping MODEL_LAYER_LIST generation" >&2
else
    for MODEL in "${MODEL_LIST[@]}"; do
        # find the first line that starts with the model name
        match=$(grep -m1 -E "^${MODEL}" scripts/models_list.txt || true)
        if [ -z "$match" ]; then
            echo "Warning: no entry for $MODEL in models_list.txt" >&2
            continue
        fi

        # preferred: extract number after "num_layer=" (allow spaces)
        num=$(echo "$match" | sed -nE 's/.*num_layer[[:space:]]*=[[:space:]]*([0-9]+).*/\1/p')

        # fallback: extract trailing "_<num>" if present
        if [ -z "$num" ]; then
            num=$(echo "$match" | sed -nE 's/.*_([0-9]+).*/\1/p')
        fi

        if [ -z "$num" ]; then
            echo "Warning: could not parse num_layer for $MODEL (line: $match)" >&2
            continue
        fi

        MODEL_LAYER_LIST+=( "${MODEL}_$num" )
    done

    echo "Generated MODEL_LAYER_LIST: ${MODEL_LAYER_LIST[*]}"
fi