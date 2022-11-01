# hifphg
mkdir output_asr
mkdir output_asr/hifphg
python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/metadata_test_30.csv" \
    --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-asr/output/hi_fastpitch_indictts_all_8_ae:0_ema:False_pretrained_asrloss0.5-October-19-2022_11+47PM-08fc70d/best_model.pth \
    --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-asr/output/hi_fastpitch_indictts_all_8_ae:0_ema:False_pretrained_asrloss0.5-October-19-2022_11+47PM-08fc70d/config_v2.json \
    --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/hi/best_model.pth \
    --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/hi/config.json \
    --out_path output_asr/hifphg \
    --use_cuda t

python3 scripts/evaluate_mcd.py \
    output_asr/hifphg \
    /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

python3 scripts/evaluate_f0.py \
    output_asr/hifphg \
    /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# # tafphg
# mkdir output_asr
# mkdir output_asr/tafphg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-asr/output_ta/output_ta/ta_fastpitch_indictts_all_8_ae:0_ema:False_pretrained_asrloss0.5-October-19-2022_11+51PM-08fc70d/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-asr/output_ta/output_ta/ta_fastpitch_indictts_all_8_ae:0_ema:False_pretrained_asrloss0.5-October-19-2022_11+51PM-08fc70d/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/ta/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/ta/config.json \
#     --out_path output_asr/tafphg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_asr/tafphg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_asr/tafphg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/