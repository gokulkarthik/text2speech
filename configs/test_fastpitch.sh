#samples_te_fastpitch_indictts_all_f1__3x
mkdir output_wavs
mkdir output_wavs/samples_te_fastpitch_indictts_all_f1__3x

python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts/te/metadata_test.csv" \
    --model_path output/tef13x_fastpitch_indictts_all-September-30-2022_08+53PM-a2d54d6/best_model.pth \
    --config_path output/tef13x_fastpitch_indictts_all-September-30-2022_08+53PM-a2d54d6/config.json \
    --vocoder_path indic_vocoders/te_hifigan_all-September-20-2022_10+44AM-a2d54d6/best_model.pth \
    --vocoder_config_path indic_vocoders/te_hifigan_all-September-20-2022_10+44AM-a2d54d6/config.json \
    --out_path output_wavs/samples_te_fastpitch_indictts_all_f1__3x \
    --use_cuda t \

python3 scripts/evaluate_mcd.py \
    output_wavs/samples_te_fastpitch_indictts_all_f1__3x \
    /home/praveen/ttsteam/datasets/indictts/te/wavs-22k

python3 scripts/evaluate_f0.py \
    output_wavs/samples_te_fastpitch_indictts_all_f1__3x \
    /home/praveen/ttsteam/datasets/indictts/te/wavs-22k


# #samples_te_fastpitch_indictts_all_2xlengththreshold
# mkdir output_wavs
# mkdir output_wavs/samples_te_fastpitch_indictts_all_2xlengththreshold

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts/te/metadata_test.csv" \
#     --model_path output/te_fastpitch_indictts_all-September-29-2022_06+04AM-a2d54d6/best_model.pth \
#     --config_path output/te_fastpitch_indictts_all-September-29-2022_06+04AM-a2d54d6/config.json \
#     --vocoder_path indic_vocoders/te_hifigan_all-September-20-2022_10+44AM-a2d54d6/best_model.pth \
#     --vocoder_config_path indic_vocoders/te_hifigan_all-September-20-2022_10+44AM-a2d54d6/config.json \
#     --out_path output_wavs/samples_te_fastpitch_indictts_all_2xlengththreshold \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_wavs/samples_te_fastpitch_indictts_all_2xlengththreshold \
#     /home/praveen/ttsteam/datasets/indictts/te/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_wavs/samples_te_fastpitch_indictts_all_2xlengththreshold \
#     /home/praveen/ttsteam/datasets/indictts/te/wavs-22k


#-------------------------------------------------------------------------------
# EVALUATE FASTPITCH

# cp -r output/hi_fastpitch_indictts_male_6_ae:0_ema:False_pretrained_asrloss0.2-September-08-2022_11+37PM-7863a47/* output/store/hi
# cp -r output/hi_fastpitch_indictts_male_32_ae:1000_ema:True_description-September-08-2022_11+38PM-7863a47/* output/store/hi
# cp -r output/hi_fastpitch_indictts_male_6_ae\:0_ema\:False_pretrained_continue-September-12-2022_04+41PM-7863a47/* output/store/hi
# cp -r output/te_fastpitch_indictts_all-September-15-2022_12+56PM-a2d54d6/* output/store/te
# cp -r output/hi_fastpitch_indictts_all-September-15-2022_11+39AM-a2d54d6/* output/store/hi
# cp -r output/mr_fastpitch_indictts_all-September-15-2022_11+41AM-a2d54d6/* output/store/mr

 
# mkdir output_wavs
# mkdir output_wavs/demo_mr_fastpitch_indictts_all

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts/mr/demo.csv" \
#     --model_path output/store/mr/best_model.pth \
#     --config_path output/store/mr/config.json \
#     --vocoder_path saved_models/hifigan/mr/male_female/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/mr/male_female/config.json \
#     --out_path output_wavs/demo_mr_fastpitch_indictts_all \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_wavs/demo_hi_fastpitch_indictts_all \
#     /home/praveen/ttsteam/datasets/indictts/mr/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_wavs/demo_hi_fastpitch_indictts_all \
#     /home/praveen/ttsteam/datasets/indictts/mr/wavs-22k

#-------------------------------------------------------------------------------

# cp output/ta_fastpitch_googletts_all-July-21-2022_12+01PM-11635b8/best_model_423130.pth output/store/ta/fastpitch_multi/
# cp output/ta_fastpitch_googletts_all-July-21-2022_12+01PM-11635b8/config.json output/store/ta/fastpitch_multi/
# cp output/ta_fastpitch_googletts_all-July-21-2022_12+01PM-11635b8/speakers.pth output/store/ta/fastpitch_multi/
# cp output_vocoder/ta_hifigan_all-July-27-2022_06+13AM-d52256a/best_model_94336.pth output_vocoder/store/ta/hifigan_multi/
# cp output_vocoder/ta_hifigan_all-July-27-2022_06+13AM-d52256a/checkpoint_400000.pth output_vocoder/store/ta/hifigan_multi/
# cp output_vocoder/ta_hifigan_all-July-27-2022_06+13AM-d52256a/config.json output_vocoder/store/ta/hifigan_multi/

# python3 -m TTS.bin.synthesize --text "../../datasets/indictts/ta/samples.csv" \
#     --model_path output/store/ta/fastpitch/best_model.pth \
#     --config_path output/store/ta/fastpitch/config.json \
#     --vocoder_path output_vocoder/store/ta/hifigan/checkpoint_1060000.pth \
#     --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
#     --out_path output_wavs/samples_indictts_ta_male_fastpitch

# python3 scripts/evaluate_mcd.py \
#     output_wavs/ta_male_fastpitch_hifi/ \
#     data_dir/indictts/ta/wavs-20k-test-male

# python3 scripts/evaluate_f0.py \
#     output_wavs/ta_male_fastpitch_hifi/ \
#     /data_dir/indictts/ta/wavs-20k-test-male