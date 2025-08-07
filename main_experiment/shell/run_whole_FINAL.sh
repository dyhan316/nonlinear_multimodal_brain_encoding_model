#!/bin/sh

conda activate "your_environment_name"

cd "your_project_path"

####set settings####
#fixed settings 
name=default_name #default_name

#for loop settings
subs=("S1") #("S3" "S2" "S1" ) 
stim_setting=("default" )
enc_setting=("MLP_128" "MLP_128_128" "MLP_128_128_128_128")   #("MLP_128" "MLP_128_128" "MLP_128_128_128_128") ("MLP_512" "MLP_512_512" "MLP_512_512_512_512") #  #("default" ) ("MLP_2048" "MLP_2048_2048" "MLP_2048_2048_2048_2048") 
resp_setting=("PCA_256" "PCA_1024" "PCA_4096") #("default" ) #("PCAonIMP_32" "PCAonIMP_128" "PCAonIMP_1024" "PCAonIMP_4096" ) #("default_32" "default_128" "default_1024" "default_4096" ) #("PCA_32" "PCA_128" "PCA_1024" "PCA_4096" ) # PCAonIMP_1024, default_1024
WRresps=("default" ) # Add more WRresps here


##Run each step sequentially 
for sub in "${subs[@]}"; do
    for stim_setting in "${stim_setting[@]}"; do
        for enc_setting in "${enc_setting[@]}"; do
            for resp_setting in "${resp_setting[@]}"; do
                for WRresp in "${WRresps[@]}"; do
                    echo $sub $stim_setting $enc_setting $resp_setting $WRresp "started"
                    python train_EM_for_main.py --subject $sub --gpt $gpt --stim_setting $stim_setting --enc_setting $enc_setting --resp_setting $resp_setting --name $name  ; echo "train_EM.py finished"
                    python evaluate_encoder.py --subject $sub --gpt $gpt --stim_setting $stim_setting --enc_setting $enc_setting --resp_setting $resp_setting --name $name ; echo "evaluate_encoder done"

                    echo $sub $stim_setting $enc_setting $resp_setting $WRresp "done"
                done
            done
        done
    done
done
