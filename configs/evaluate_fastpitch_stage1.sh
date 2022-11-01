# # asfastpitch
# mkdir output_indic_fastpitch/evaluation_samples/as
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/as/metadata_test.csv" \
#     --model_path output_indic_fastpitch/as/as_fastpitch_indictts_all-September-28-2022_12+57PM-5098e7d/best_model.pth \
#     --config_path output_indic_fastpitch/as/as_fastpitch_indictts_all-September-28-2022_12+57PM-5098e7d/config.json \
#     --vocoder_path saved_models/hifigan/v1/as/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/as/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/as \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/as \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/as/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/as \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/as/wavs-22k/


# # bnfastpitch
# mkdir output_indic_fastpitch/evaluation_samples/bn
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/bn/metadata_test.csv" \
#     --model_path output_indic_fastpitch/bn/bn_fastpitch_indictts_all-September-28-2022_01+02PM-5098e7d/best_model.pth \
#     --config_path output_indic_fastpitch/bn/bn_fastpitch_indictts_all-September-28-2022_01+02PM-5098e7d/config.json \
#     --vocoder_path saved_models/hifigan/v1/bn/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/bn/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/bn \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/bn \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/bn/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/bn \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/bn/wavs-22k/


# # brxfastpitch
# mkdir output_indic_fastpitch/evaluation_samples/brx
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/brx/metadata_test.csv" \
#     --model_path output_indic_fastpitch/brx/brx_fastpitch_indictts_all-September-28-2022_01+28PM-5098e7d/best_model.pth \
#     --config_path output_indic_fastpitch/brx/brx_fastpitch_indictts_all-September-28-2022_01+28PM-5098e7d/config.json \
#     --vocoder_path saved_models/hifigan/v1/brx/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/brx/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/brx \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/brx \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/brx/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/brx \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/brx/wavs-22k/


# # gufastpitch
# mkdir output_indic_fastpitch/evaluation_samples/gu
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/gu/metadata_test.csv" \
#     --model_path output_indic_fastpitch/gu/gu_fastpitch_indictts_all-September-28-2022_01+28PM-5098e7d/best_model.pth \
#     --config_path output_indic_fastpitch/gu/gu_fastpitch_indictts_all-September-28-2022_01+28PM-5098e7d/config.json \
#     --vocoder_path saved_models/hifigan/v1/gu/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/gu/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/gu \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/gu \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/gu/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/gu \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/gu/wavs-22k/


# hifastpitch
mkdir output_indic_fastpitch/evaluation_samples/hi_stage1
python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/metadata_test.csv" \
    --model_path output_indic_fastpitch/hi/hi_fastpitch_indictts_all-September-28-2022_01+30PM-5098e7d/best_model.pth \
    --config_path output_indic_fastpitch/hi/hi_fastpitch_indictts_all-September-28-2022_01+30PM-5098e7d/config.json \
    --vocoder_path saved_models/hifigan/v1/hi/best_model.pth \
    --vocoder_config_path saved_models/hifigan/v1/hi/config.json \
    --out_path output_indic_fastpitch/evaluation_samples/hi_stage1 \
    # --use_cuda t

python3 scripts/evaluate_mcd.py \
    output_indic_fastpitch/evaluation_samples/hi_stage1 \
    /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/

python3 scripts/evaluate_f0.py \
    output_indic_fastpitch/evaluation_samples/hi_stage1 \
    /nlsasfs/home/ai4bharat/praveens/ttsteam/datasets/indictts/hi/wavs-22k/


# # mnifastpitch
# mkdir output_indic_fastpitch/evaluation_samples/mni
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/mni/metadata_test.csv" \
#     --model_path output_indic_fastpitch/mni/mni_fastpitch_indictts_all-September-28-2022_01+29PM-5098e7d/best_model.pth \
#     --config_path output_indic_fastpitch/mni/mni_fastpitch_indictts_all-September-28-2022_01+29PM-5098e7d/config.json \
#     --vocoder_path saved_models/hifigan/v1/mni/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/mni/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/mni \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/mni \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/mni/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/mni \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/mni/wavs-22k/


# # mrfastpitch
# mkdir output_indic_fastpitch/evaluation_samples/mr
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/mr/metadata_test.csv" \
#     --model_path output_indic_fastpitch/mr/mr_fastpitch_indictts_all-September-28-2022_01+42PM-5098e7d/best_model.pth \
#     --config_path output_indic_fastpitch/mr/mr_fastpitch_indictts_all-September-28-2022_01+42PM-5098e7d/config.json \
#     --vocoder_path saved_models/hifigan/v1/mr/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/mr/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/mr \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/mr \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/mr/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/mr \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/mr/wavs-22k/


# # orfastpitch
# mkdir output_indic_fastpitch/evaluation_samples/or
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/or/metadata_test.csv" \
#     --model_path output_indic_fastpitch/or/or_fastpitch_indictts_all-September-28-2022_07+52PM-5098e7d/best_model.pth \
#     --config_path output_indic_fastpitch/or/or_fastpitch_indictts_all-September-28-2022_07+52PM-5098e7d/config.json \
#     --vocoder_path saved_models/hifigan/v1/or/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/or/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/or \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/or \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/or/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/or \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/or/wavs-22k/


# # rajfastpitch
# mkdir output_indic_fastpitch/evaluation_samples/raj
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/raj/metadata_test.csv" \
#     --model_path output_indic_fastpitch/raj/raj_fastpitch_indictts_all-September-28-2022_01+32PM-5098e7d/best_model.pth \
#     --config_path output_indic_fastpitch/raj/raj_fastpitch_indictts_all-September-28-2022_01+32PM-5098e7d/config.json \
#     --vocoder_path saved_models/hifigan/v1/raj/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/raj/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/raj \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/raj \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/raj/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/raj \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/raj/wavs-22k/


# # knfastpitch
# mkdir output_indic_fastpitch/evaluation_samples/kn
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/kn/metadata_test.csv" \
#     --model_path output_indic_fastpitch/kn/kn_fastpitch_indictts_all-September-28-2022_03+36PM-a2d54d6/best_model.pth \
#     --config_path output_indic_fastpitch/kn/kn_fastpitch_indictts_all-September-28-2022_03+36PM-a2d54d6/config.json \
#     --vocoder_path saved_models/hifigan/v1/kn/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/kn/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/kn \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/kn \
#     /nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/kn/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/kn \
#     /nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/kn/wavs-22k/


# # tafastpitch
# mkdir output_indic_fastpitch/evaluation_samples/ta
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/ta/metadata_test.csv" \
#     --model_path output_indic_fastpitch/ta/ta_fastpitch_indictts_all-September-28-2022_03+36PM-a2d54d6/best_model.pth \
#     --config_path output_indic_fastpitch/ta/ta_fastpitch_indictts_all-September-28-2022_03+36PM-a2d54d6/config.json \
#     --vocoder_path saved_models/hifigan/v1/ta/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/ta/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/ta \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/ta \
#     /nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/ta/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/ta \
#     /nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/ta/wavs-22k/


# # mlfastpitch
# mkdir output_indic_fastpitch/evaluation_samples/ml
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/ml/metadata_test.csv" \
#     --model_path output_indic_fastpitch/ml/ml_fastpitch_indictts_all-September-28-2022_03+58PM-a2d54d6/best_model.pth \
#     --config_path output_indic_fastpitch/ml/ml_fastpitch_indictts_all-September-28-2022_03+58PM-a2d54d6/config.json \
#     --vocoder_path saved_models/hifigan/v1/ml/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/ml/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/ml \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/ml \
#     /nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/ml/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/ml \
#     /nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/ml/wavs-22k/


# # tefastpitch
# mkdir output_indic_fastpitch/evaluation_samples/te
# python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/te/metadata_test.csv" \
#     --model_path output_indic_fastpitch/te/te_fastpitch_indictts_all-September-29-2022_05+02PM-a2d54d6/best_model.pth \
#     --config_path output_indic_fastpitch/te/te_fastpitch_indictts_all-September-29-2022_05+02PM-a2d54d6/config.json \
#     --vocoder_path saved_models/hifigan/v1/te/best_model.pth \
#     --vocoder_config_path saved_models/hifigan/v1/te/config.json \
#     --out_path output_indic_fastpitch/evaluation_samples/te \
#     --use_cuda t

# python3 scripts/evaluate_mcd.py \
#     output_indic_fastpitch/evaluation_samples/te \
#     /nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/te/wavs-22k/

# python3 scripts/evaluate_f0.py \
#     output_indic_fastpitch/evaluation_samples/te \
#     /nlsasfs/home/ai4bharat/vinodg/ttsteam/datasets/indictts/te/wavs-22k/
