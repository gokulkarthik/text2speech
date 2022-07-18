cp exp_dir/best_model.pth output/store/ta/vits/
cp exp_dir/config.json output/store/ta/vits/

python3 -m TTS.bin.synthesize --text "../../datasets/indictts/ta/metadata_test_male.csv" \
    --model_path output/store/ta/vits/best_model.pth \
    --config_path output/store/ta/vits/config.json \
    --out_path output_wavs/ta_male_vits_none

# python3 scripts/evaluate_mcd.py \
#     output_wavs/ta_male_vits_none/ \
#     data_dir/indictts/ta/wavs-20k-test-male

# python3 scripts/evaluate_f0.py \
#     output_wavs/ta_male_vits_none/ \
#     /data_dir/indictts/ta/wavs-20k-test-male