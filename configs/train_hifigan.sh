CUDA_VISIBLE_DEVICES='2' python3 vocoder.py --dataset_name googletts \
    --language ta \
    --speaker all \
    --batch_size 16 \
    --batch_size_eval 16 \
    --epochs 5000 \
    --port 10004 \
    --mixed_precision t \
    --run_description "hifigan_ta_multi"