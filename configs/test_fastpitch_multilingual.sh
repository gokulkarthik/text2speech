# # mlfpdravidian  kn
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpdravidian_indictts_kn

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/dravidian/metadata_test.csv" \
#     --model_path output/dravidian_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_04+55PM-3c8e9b3/best_model.pth \
#     --config_path output/dravidian_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_04+55PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/kn/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/kn/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpdravidian_indictts_kn \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpdravidian_indictts_kn \
#     /home/praveen/ttsteam/datasets/indictts/kn/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpdravidian_indictts_kn \
#     /home/praveen/ttsteam/datasets/indictts/kn/wavs-22k

# # mlfpdravidian ml
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpdravidian_indictts_ml

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/dravidian/metadata_test.csv" \
#     --model_path output/dravidian_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_04+55PM-3c8e9b3/best_model.pth \
#     --config_path output/dravidian_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_04+55PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/ml/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/ml/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpdravidian_indictts_ml \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpdravidian_indictts_ml \
#     /home/praveen/ttsteam/datasets/indictts/ml/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpdravidian_indictts_ml \
#     /home/praveen/ttsteam/datasets/indictts/ml/wavs-22k


# # mlfpdravidian ta
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpdravidian_indictts_ta

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/dravidian/metadata_test.csv" \
#     --model_path output/dravidian_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_04+55PM-3c8e9b3/best_model.pth \
#     --config_path output/dravidian_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_04+55PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/ta/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/ta/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpdravidian_indictts_ta \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpdravidian_indictts_ta \
#     /home/praveen/ttsteam/datasets/indictts/ta/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpdravidian_indictts_ta \
#     /home/praveen/ttsteam/datasets/indictts/ta/wavs-22k


# # mlfpdravidian te
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpdravidian_indictts_te

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/dravidian/metadata_test.csv" \
#     --model_path output/dravidian_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_04+55PM-3c8e9b3/best_model.pth \
#     --config_path output/dravidian_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_04+55PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/te/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/te/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpdravidian_indictts_te \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpdravidian_indictts_te \
#     /home/praveen/ttsteam/datasets/indictts/tef13x/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpdravidian_indictts_te \
#     /home/praveen/ttsteam/datasets/indictts/tef13x/wavs-22k


# # mlfpindoaryan as
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpindoaryan_indictts_as

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/indoaryan/metadata_test.csv" \
#     --model_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/best_model.pth \
#     --config_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/as/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/as/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpindoaryan_indictts_as \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_as \
#     /home/praveen/ttsteam/datasets/indictts/as/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_as \
#     /home/praveen/ttsteam/datasets/indictts/as/wavs-22k


# # mlfpindoaryan bn
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpindoaryan_indictts_bn

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/indoaryan/metadata_test.csv" \
#     --model_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/best_model.pth \
#     --config_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/bn/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/bn/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpindoaryan_indictts_bn \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_bn \
#     /home/praveen/ttsteam/datasets/indictts/bn/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_bn \
#     /home/praveen/ttsteam/datasets/indictts/bn/wavs-22k


# # mlfpindoaryan brx (Not Supported)
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpindoaryan_indictts_brx

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/indoaryan/metadata_test.csv" \
#     --model_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/best_model.pth \
#     --config_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/brx/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/brx/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpindoaryan_indictts_brx \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_brx \
#     /home/praveen/ttsteam/datasets/indictts/brx/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_brx \
#     /home/praveen/ttsteam/datasets/indictts/brx/wavs-22k


# # mlfpindoaryan gu
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpindoaryan_indictts_gu

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/indoaryan/metadata_test.csv" \
#     --model_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/best_model.pth \
#     --config_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/gu/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/gu/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpindoaryan_indictts_gu \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_gu \
#     /home/praveen/ttsteam/datasets/indictts/gu/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_gu \
#     /home/praveen/ttsteam/datasets/indictts/gu/wavs-22k


# # mlfpindoaryan hi
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpindoaryan_indictts_hi

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/indoaryan/metadata_test.csv" \
#     --model_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/best_model.pth \
#     --config_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/hi/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/hi/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpindoaryan_indictts_hi \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_hi \
#     /home/praveen/ttsteam/datasets/indictts/hi/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_hi \
#     /home/praveen/ttsteam/datasets/indictts/hi/wavs-22k


# # mlfpindoaryan mni (Not Supported)
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpindoaryan_indictts_mni

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/indoaryan/metadata_test.csv" \
#     --model_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/best_model.pth \
#     --config_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/mni/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/mni/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpindoaryan_indictts_mni \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_mni \
#     /home/praveen/ttsteam/datasets/indictts/mni/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_mni \
#     /home/praveen/ttsteam/datasets/indictts/mni/wavs-22k


# # mlfpindoaryan mr
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpindoaryan_indictts_mr

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/indoaryan/metadata_test.csv" \
#     --model_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/best_model.pth \
#     --config_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/mr/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/mr/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpindoaryan_indictts_mr \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_mr \
#     /home/praveen/ttsteam/datasets/indictts/mr/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_mr \
#     /home/praveen/ttsteam/datasets/indictts/mr/wavs-22k


# # mlfpindoaryan or
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpindoaryan_indictts_or

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/indoaryan/metadata_test.csv" \
#     --model_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/best_model.pth \
#     --config_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/or/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/or/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpindoaryan_indictts_or \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_or \
#     /home/praveen/ttsteam/datasets/indictts/or/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_or \
#     /home/praveen/ttsteam/datasets/indictts/or/wavs-22k


# # mlfpindoaryan raj
# mkdir output_mlfp_wavs
# mkdir output_mlfp_wavs/samples_mlfpindoaryan_indictts_raj

# python3 -m TTS.bin.synthesize --text "~/ttsteam/datasets/indictts_multilingual_aksharamukha/indoaryan/metadata_test.csv" \
#     --model_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/best_model.pth \
#     --config_path output/indoaryan_fastpitch_indictts_multilingual_aksharamukha_all-October-10-2022_05+21PM-3c8e9b3/config_v2.json \
#     --vocoder_path ../text2speech-ta/saved_models/hifigan/v1/raj/best_model.pth \
#     --vocoder_config_path ../text2speech-ta/saved_models/hifigan/v1/raj/config.json \
#     --out_path output_mlfp_wavs/samples_mlfpindoaryan_indictts_raj \
#     --use_cuda t \

# python3 scripts/evaluate_mcd.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_raj \
#     /home/praveen/ttsteam/datasets/indictts/raj/wavs-22k

# python3 scripts/evaluate_f0.py \
#     output_mlfp_wavs/samples_mlfpindoaryan_indictts_raj \
#     /home/praveen/ttsteam/datasets/indictts/raj/wavs-22k