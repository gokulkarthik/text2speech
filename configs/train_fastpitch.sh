# hifpmale
python3 main.py --dataset_name indictts \
    --language ta \
    --speaker female \
    --max_audio_len 441000 \
    --max_text_len 400 \
    --model fastpitch \
    --hidden_channels 512 \
    --use_speaker_embedding f \
    --use_d_vector_file f \
    --use_speaker_encoder_as_loss f \
    --use_ssim_loss f \
    --use_aligner t \
    --use_pre_computed_alignments f \
    --batch_size 32 \
    --batch_size_eval 32 \
    --batch_group_size 0 \
    --epochs 2500 \
    --aligner_epochs 2500 \
    --lr 0.0001 \
    --lr_scheduler NoamLR \
    --lr_scheduler_warmup_steps 4000 \
    --num_workers 0 \
    --num_workers_eval 0 \
    --mixed_precision t 

# # hifpfemale
# python3 main.py --dataset_name indictts \
#     --language hi \
#     --speaker female \
#     --max_audio_len 441000 \
#     --max_text_len 400 \
#     --model fastpitch \
#     --hidden_channels 512 \
#     --use_speaker_embedding f \
#     --use_d_vector_file f \
#     --use_speaker_encoder_as_loss f \
#     --use_ssim_loss f \
#     --use_aligner t \
#     --use_pre_computed_alignments f \
#     --batch_size 32 \
#     --batch_size_eval 32 \
#     --batch_group_size 0 \
#     --epochs 2500 \
#     --aligner_epochs 2500 \
#     --lr 0.0001 \
#     --lr_scheduler NoamLR \
#     --lr_scheduler_warmup_steps 4000 \
#     --num_workers 0 \
#     --num_workers_eval 0 \
#     --mixed_precision t 

# # tef13x
# python3 main.py --dataset_name indictts \
#     --language tef13x \
#     --speaker all \
#     --max_audio_len 441000 \
#     --max_text_len 400 \
#     --model fastpitch \
#     --hidden_channels 512 \
#     --use_speaker_embedding t \
#     --use_d_vector_file f \
#     --use_speaker_encoder_as_loss f \
#     --use_ssim_loss f \
#     --use_aligner t \
#     --use_pre_computed_alignments f \
#     --batch_size 8 \
#     --batch_size_eval 8 \
#     --batch_group_size 0 \
#     --epochs 2500 \
#     --aligner_epochs 2500 \
#     --lr 0.0001 \
#     --lr_scheduler NoamLR \
#     --lr_scheduler_warmup_steps 4000 \
#     --num_workers 0 \
#     --num_workers_eval 0 \
#     --mixed_precision t 

# python3 main.py --dataset_name indictts \
#     --language te \
#     --speaker all \
#     --max_audio_len 882000 \
#     --max_text_len 800 \
#     --model fastpitch \
#     --hidden_channels 512 \
#     --use_speaker_embedding t \
#     --use_d_vector_file f \
#     --use_speaker_encoder_as_loss f \
#     --use_ssim_loss f \
#     --use_aligner t \
#     --use_pre_computed_alignments f \
#     --batch_size 8 \
#     --batch_size_eval 8 \
#     --batch_group_size 0 \
#     --epochs 2500 \
#     --aligner_epochs 2500 \
#     --lr 0.0001 \
#     --lr_scheduler NoamLR \
#     --lr_scheduler_warmup_steps 4000 \
#     --num_workers 0 \
#     --num_workers_eval 0 \
#     --mixed_precision t \
#     --pretrained_checkpoint_path output/te_fastpitch_indictts_all-September-28-2022_05+02PM-a2d54d6/checkpoint_90000.pth
