python3 main.py --dataset_name ai4b-ta \
    --dataset_path '../../data/tts/ai4b_preprocessed' \
    --language ta \
    --model vits \
    --batch_size 32 \
    --batch_size_eval 32 \
    --epochs 1000 \
    --phoneme_language ta \
    --use_phonemes f