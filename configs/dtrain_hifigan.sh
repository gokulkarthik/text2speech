CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -m trainer.distribute --script vocoder.py --dataset_name indictts \
    --language ta \
    --speaker male \
    --batch_size 8 \
    --batch_size_eval 8 \
    --epochs 10000 \
    --port 10004 \
    --run_description "hifigan_ta_male"