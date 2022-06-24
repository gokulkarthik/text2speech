CUDA_VISIBLE_DEVICES='0' python3 -m trainer.distribute --script vocoder.py --dataset_name indictts \
    --language ta \
    --speaker female \
    --batch_size 16 \
    --batch_size_eval 16 \
    --epochs 1000 \ 
    --port 10004 \
    --run_description "hifigan_ta_female"