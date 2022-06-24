CUDA_VISIBLE_DEVICES='0,1,2,3' python3 -m traineer.dristribute --script main.py --dataset_name indictts \
    --language ta \
    --speaker female \
    --use_speaker_embedding f \
    --model vits \
    --batch_size 8 \
    --batch_size_eval 8 \
    --epochs 1000 \ 
    --run_description "vits_ta_female"