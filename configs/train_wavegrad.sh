CUDA_VISIBLE_DEVICES='0' python3 vocoder_wavegrad.py --dataset_name indictts \
    --language hi \
    --speaker all \
    --model wavegrad \
    --batch_size 96 \
    --batch_size_eval 96 \
    --epochs 10000 \
    --eval_split_size 50 \
    --port 10004 \
    --mixed_precision t