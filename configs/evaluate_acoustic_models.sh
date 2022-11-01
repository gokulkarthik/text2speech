# Vocoders
# saved_models/hifigan/v1/hi/best_model.pth
# saved_models/hifigan/v1/ta/best_model.pth
# saved_models/multiband_melgan/hi/best_model.pth
# saved_models/multiband_melgan/ta/best_model.pth
# saved_models/wavegrad/hi/best_model.pth
# # saved_models/wavegrad/ta/best_model.pth

# /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic_fastpitch/hi/hi_fastpitch_indictts_all_align_off-October-08-2022_02+41AM-5098e7d/


# cp ../../text2speech-exp/recipes/indictts/multiband_melgan/run-September-12-2022_06+15PM-7863a47/config.json multiband_melgan/hi/config.json
cp ../../text2speech-exp/saved_models_from_e2e/models/multiband_melgan/ta/male_female/* multiband_melgan/ta/

# # hifastpitch
# mkdir output_indic_fastpitch/evaluation_samples/hi
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/hi/metadata_test.csv" \
#     --model_path output_indic_fastpitch/hi/hi_fastpitch_indictts_all-September-28-2022_01+30PM-5098e7d/best_model.pth \
#     --config_path output_indic_fastpitch/hi/hi_fastpitch_indictts_all-September-28-2022_01+30PM-5098e7d/config.json \
#     --vocoder_path saved_models/hifigan/v1/hi/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/hi/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/hi \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/hi \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/hi \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/hi/wavs-22k/