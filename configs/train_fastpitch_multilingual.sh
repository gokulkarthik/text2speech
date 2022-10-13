# # dravidian
# python3 main.py --dataset_name indictts_multilingual_aksharamukha \
#     --language dravidian \
#     --speaker all \
#     --max_audio_len 441000 \
#     --max_text_len 400 \
#     --model fastpitch \
#     --hidden_channels 512 \
#     --use_speaker_embedding t \
#     --use_language_embedding t \
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

# indoaryan
python3 main.py --dataset_name indictts_multilingual_aksharamukha \
    --language indoaryan \
    --speaker all \
    --max_audio_len 441000 \
    --max_text_len 400 \
    --model fastpitch \
    --hidden_channels 512 \
    --use_speaker_embedding t \
    --use_language_embedding t \
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