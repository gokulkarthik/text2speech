CUDA_VISIBLE_DEVICES='0' python3 vocoder.py --dataset_name indictts \
    --language ta \
    --speaker male \
    --batch_size 16 \
    --batch_size_eval 16 \
    --epochs 10000 \ 
    --port 10004 \
    --run_description "hifigan_ta_male"