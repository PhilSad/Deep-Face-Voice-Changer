# WIP

# Process data
['infer/modules/train/preprocess.py', '/workspace/train_folder', '40000', '32', '/workspace/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test', 'False', '3.0']

python infer/modules/train/preprocess.py /workspace/train_folder \
        40000 # target sample rate [32000, 400000, 48000] \
        32  # Number of CPU processes used for pitch extraction and data processing \
        /workspace/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test # Experiment Dir \
        False # no parralle \
        3.0 # ? \

# extract features
python infer/modules/train/extract/extract_f0_print.py \
        /workspace/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test # Experiment Dir \
        32 # Number of CPU processes used for pitch extraction and data processing \
        harvest # Pitch extraction method \

python infer/modules/train/extract_feature_print.py \
        cuda:0 # GPU ID \
        1 # Number of GPUs \
        0 # GPU offset \
        0 # Number of CPU processes used for feature extraction \
        /workspace/Retrieval-based-Voice-Conversion-WebUI/logs/mi-test # Experiment Dir \
        v2 # Feature extraction method \

# train 3a
"/usr/bin/python" infer/modules/train/train.py -e "macron" -sr 40k -f0 1 -bs 8 -g 0 -te 20 -se 5 -pg assets/pretrained_v2/f0G40k.pth -pd assets/pretrained_v2/f0D40k.pth -l 0 -c 0 -sw 0 -v v2
python infer/modules/train/train.py \
        -e "macron" # experiment dir \
        -sr 40k # sample rate \
        -f0 1 # use f0 as one of the inputs of the model \
        -bs 8 # batch size \
        -g 0 # GPU ID \
        -te 20 # total_epoch \
        -se 5 # checkpoint save frequency (epoch) \
        -pg assets/pretrained_v2/f0G40k.pth # Pretrained Discriminator path \
        -pd assets/pretrained_v2/f0D40k.pth # Pretrained Generator path \
        -l 0 # if only save the latest G/D pth file \
        -c 0 # if caching the dataset in GPU memory \
        -sw 0 # save the extracted model in weights directory when saving checkpoints \
        -v v2 # model version \