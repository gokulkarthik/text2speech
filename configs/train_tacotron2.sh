python3 main.py --dataset_name indictts \
    --language ta \
    --speaker male \
    --max_audio_len 441000 \
    --max_text_len 400 \
    --audio_config with_norm \
    --model tacotron2 \
    --use_speaker_embedding t \
    --batch_size 32 \
    --batch_size_eval 32 \
    --batch_group_size 0 \
    --epochs 2500 \
    --lr 0.0001 \
    --lr_scheduler NoamLR \
    --lr_scheduler_warmup_steps 4000 \
    --lr_scheduler_step_size 500 \
    --lr_scheduler_gamma 0.1 \
    --lr_scheduler_threshold_step 500 \
    --num_workers 0 \
    --num_workers_eval 0 \
    --output_path output_indic-tts-acoustic/ta-male \
    --mixed_precision t \
    --run_description ""

    # --output_path output_indic-tss-acoustic/ta \
    # --mixed_precision t \
    # --run_description ""