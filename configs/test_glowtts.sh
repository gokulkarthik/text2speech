cp exp_dir/best_model.pth output/store/ta/glowtts/
cp exp_dir/config.json output/store/ta/glowtts/
cp voc_exp_dir/best_model.pth output_vocoder/store/ta/hifigan/
cp voc_exp_dir/config.json output_vocoder/store/ta/hifigan/

python3 -m TTS.bin.synthesize --text "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/ta/metadata_test.csv" \
    --model_path output/store/ta/glowtts/best_model.pth \
    --config_path output/store/ta/glowtts/config.json \
    --vocoder_path output_vocoder/store/ta/hifigan/best_model.pth \
    --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
    --out_path output_wavs/ta_glowtts_female/

# python3 scripts/evaluate_mcd.py \
#     output_wavs/ta_glowtts_female/ \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/ta/wavs-20k-test-female

# python3 scripts/evaluate_f0.py \
#     output_wavs/ta_glowtts_female/ \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/ta/wavs-20k-test-female