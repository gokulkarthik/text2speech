
# tef13xfpcont
python3 train_align_off.py --dataset_name indictts \
    --language tef13x \
    --speaker all \
    --max_audio_len 441000 \
    --max_text_len 400 \
    --model fastpitch \
    --hidden_channels 512 \
    --use_speaker_embedding t \
    --use_d_vector_file f \
    --use_speaker_encoder_as_loss f \
    --use_ssim_loss f \
    --use_aligner t \
    --use_pre_computed_alignments f \
    --batch_size 32 \
    --batch_size_eval 32 \
    --batch_group_size 0 \
    --epochs 1000 \
    --aligner_epochs 0 \
    --lr 0.0001 \
    --lr_scheduler NoamLR \
    --num_workers 0 \
    --num_workers_eval 0 \
    --mixed_precision t \
    --output_path output_indic_fastpitch/tef13x \
    --pretrained_checkpoint_path output/tef13x_fastpitch_indictts_all-September-30-2022_08+53PM-a2d54d6/best_model.pth \
    --run_description align_off 