CUDA_VISIBLE_DEVICES='4,5,6,7' python3 -m trainer.distribute --script vocoder.py \
    --speaker tamilfemale \
    --batch_size 16 \
    --batch_size_eval 16 \
    --port 10004 \
    --run_description "hifigan_ta_male"