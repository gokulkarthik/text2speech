cp exp_dir/best_model.pth output/store/ta/glowtts/
cp exp_dir/config.json output/store/ta/glowtts/
# cp exp_dir/best_model.pth output_vocoder/store/ta/hifigan/
# cp exp_dir/config.json output_vocoder/store/ta/hifigan/

python3 -m TTS.bin.synthesize --text "../../datasets/indictts/ta/metadata_test_male.csv" \
    --model_path output/store/ta/glowtts/best_model.pth \
    --config_path output/store/ta/glowtts/config.json \
    --vocoder_path output_vocoder/store/ta/hifigan/checkpoint_633000.pth \
    --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
    --out_path output_wavs/ta_male_glowtts_hifi
    
# python3 scripts/evaluate_mcd.py \
#     output_wavs/ta_male_glowtts_hifi/ \
#     data_dir/indictts/ta/wavs-20k-test-male

# python3 scripts/evaluate_f0.py \
#     output_wavs/ta_male_glowtts_hifi/ \
#     /data_dir/indictts/ta/wavs-20k-test-male