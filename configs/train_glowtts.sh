CUDA_VISIBLE_DEVICES='0' python3 main.py --dataset_name indictts \
    --language ta \
    --speaker female \
    --use_speaker_embedding f \
    --model glowtts \
    --batch_size 8 \
    --batch_size_eval 8 \
    --epochs 1000 \
    --run_description "glowtts_ta_female"