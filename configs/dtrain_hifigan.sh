CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -m trainer.distribute --script vocoder.py --dataset_name indictts \
    --language ta \
    --speaker male \
    --batch_size 16 \
    --batch_size_eval 16 \
    --batch_group_size 5 \
    --epochs 5000 \
    --port 10004 \
    --mixed_precision t \
    --run_description "hifigan_ta_male"