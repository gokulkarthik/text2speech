CUDA_VISIBLE_DEVICES=6 python3 main.py --dataset_name ai4b-tts \
    --language hi \
    --model vits \
    --batch_size 12 \
    --batch_size_eval 12 \
    --epochs 1000 \
    --phoneme_language hi \
    --use_phonemes f