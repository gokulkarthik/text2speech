### MODELS TO EVALUATE
# hifphg
# hifpmbmg
# hifpwg

# higthg
# higtmbmg
# higtwg

# hitaco2hg
# hitaco2mbmg
# hitaco2wg

# hivits

# tafphg
# tafpmbmg
# tafpwg

# tagthg
# tagtmbmg
# tagtwg

# tataco2hg
# tataco2mbmg
# tataco2wg

# tavits

### SCRIPTS 

# # hifphg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/hifphg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/hi/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/hi/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/hi/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/hi/config.json \
#     --out_path output_mxn_eval/hifphg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/hifphg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/hifphg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/


# # hifpmbmg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/hifpmbmg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/hi/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/hi/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/multiband_melgan/hi/male_female/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/multiband_melgan/hi/male_female/config.json \
#     --out_path output_mxn_eval/hifpmbmg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/hifpmbmg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/hifpmbmg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/


# # hifpwg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/hifpwg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/hi/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/hi/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/wavegrad/hi/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/wavegrad/hi/config.json \
#     --out_path output_mxn_eval/hifpwg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/hifpwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/hifpwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval_postprocessed_v1/hifpwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval_postprocessed_v1/hifpwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# # higthg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/higthg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/hi/hi_glowtts_indictts_all_-October-13-2022_08+23PM-5098e7d/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/hi/hi_glowtts_indictts_all_-October-13-2022_08+23PM-5098e7d/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/hi/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/hi/config.json \
#     --out_path output_mxn_eval/higthg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/higthg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/higthg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/


# # higtmbmg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/higtmbmg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/hi/hi_glowtts_indictts_all_-October-13-2022_08+23PM-5098e7d/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/hi/hi_glowtts_indictts_all_-October-13-2022_08+23PM-5098e7d/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/multiband_melgan/hi/male_female/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/multiband_melgan/hi/male_female/config.json \
#     --out_path output_mxn_eval/higtmbmg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/higtmbmg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/higtmbmg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# # higtwg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/higtwg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/hi/hi_glowtts_indictts_all_-October-13-2022_08+23PM-5098e7d/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/hi/hi_glowtts_indictts_all_-October-13-2022_08+23PM-5098e7d/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/wavegrad/hi/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/wavegrad/hi/config.json \
#     --out_path output_mxn_eval/higtwg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/higtwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/higtwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

python3 scripts/evaluate_mcd.py \
    output_mxn_eval_postprocessed_v1/higtwg \
    /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

python3 scripts/evaluate_f0.py \
    output_mxn_eval_postprocessed_v1/higtwg \
    /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/


# # # hitaco2hg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/hitaco2hg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/metadata_test_30_male.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tss-acoustic/hi/hi_tacotron2_indictts_all_-October-15-2022_07+06PM-5098e7d/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tss-acoustic/hi/hi_tacotron2_indictts_all_-October-15-2022_07+06PM-5098e7d/config_v3.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/output_tacotron2_vocoders/hi_hifigan_all_with_norm-October-21-2022_11+21AM-5098e7d/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/output_tacotron2_vocoders/hi_hifigan_all_with_norm-October-21-2022_11+21AM-5098e7d/config.json \
#     --out_path output_mxn_eval/hitaco2hg \
#     --use_cuda t

# /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tss-acoustic/hi/hi_tacotron2_indictts_all_-October-15-2022_07+06PM-5098e7d
# /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/output_tacotron2_vocoders/hi_hifigan_all_with_norm-October-21-2022_11+21AM-5098e7d/best_model.pth
# /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/output_tacotron2_vocoders/hi_hifigan_all_with_norm-October-21-2022_11+21AM-5098e7d/config.json
# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/hitaco2hg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/hitaco2hg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/


# # hivits
# mkdir output_mxn_eval
# mkdir output_mxn_eval/hivits
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/hi/hi_vits_indictts_all_-October-13-2022_08+22PM-5098e7d/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/hi/hi_vits_indictts_all_-October-13-2022_08+22PM-5098e7d/config.json \
#     --out_path output_mxn_eval/hivits \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/hivits \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/hivits \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

# # tafphg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/tafphg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/ta/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/ta/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/ta/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/ta/config.json \
#     --out_path output_mxn_eval/tafphg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/tafphg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/tafphg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# # tafpmbmg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/tafpmbmg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/ta/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/ta/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/multiband_melgan/ta/male_female/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/multiband_melgan/ta/male_female/config.json \
#     --out_path output_mxn_eval/tafpmbmg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/tafpmbmg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/tafpmbmg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/


# # tafpwg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/tafpwg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/ta/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/ta/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/saved_models/wavegrad/ta/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/saved_models/wavegrad/ta/config.json \
#     --out_path output_mxn_eval/tafpwg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/tafpwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/tafpwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval_postprocessed_v1/tafpwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval_postprocessed_v1/tafpwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/


# # tagthg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/tagthg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/ta/ta_glowtts_indictts_all_-October-13-2022_08+24PM-5098e7d/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/ta/ta_glowtts_indictts_all_-October-13-2022_08+24PM-5098e7d/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/ta/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/hifigan/v1/ta/config.json \
#     --out_path output_mxn_eval/tagthg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/tagthg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/tagthg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/


# # tagtmbmg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/tagtmbmg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/ta/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/fastpitch/v1/ta/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/multiband_melgan/ta/male_female/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam/repos/text2speech-ta/saved_models/multiband_melgan/ta/male_female/config.json \
#     --out_path output_mxn_eval/tagtmbmg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/tagtmbmg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/tagtmbmg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# tagtwg
# mkdir output_mxn_eval
# mkdir output_mxn_eval/tagtwg
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/ta/ta_glowtts_indictts_all_-October-13-2022_08+24PM-5098e7d/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/ta/ta_glowtts_indictts_all_-October-13-2022_08+24PM-5098e7d/config.json \
#     --vocoder_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/saved_models/wavegrad/ta/best_model.pth \
#     --vocoder_config_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/saved_models/wavegrad/ta/config.json \
#     --out_path output_mxn_eval/tagtwg \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/tagtwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/tagtwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval_postprocessed_v1/tagtwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval_postprocessed_v1/tagtwg \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/


# # tavits
# mkdir output_mxn_eval
# mkdir output_mxn_eval/tavits
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/metadata_test_30.csv" \
#     --model_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/ta/ta_vits_indictts_all_-October-13-2022_08+23PM-5098e7d/best_model.pth \
#     --config_path /nlsasfs/home/ai4bharat/praveens/ttsteam_manidl/repos/text2speech-gokul/output_indic-tts-acoustic/ta/ta_vits_indictts_all_-October-13-2022_08+23PM-5098e7d/config.json \
#     --out_path output_mxn_eval/tavits \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_mxn_eval/tavits \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_mxn_eval/tavits \
#     /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/ta/wavs-22k/
