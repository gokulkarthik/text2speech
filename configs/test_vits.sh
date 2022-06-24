cp exp_dir/best_model.pth output/store/ta/vits/
cp exp_dir/config.json output/store/ta/vits/

python3 -m TTS.bin.synthesize --text_file_path "/nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/ta/metadata_test.csv" \
    --model_path output/store/ta/vits/best_model.pth \
    --config_path output/store/ta/vits/config.json \
    --speaker_name female \
    --out_folder output_wavs/ta_vits_female/

python3 scripts/evaluate_mcd.py \
    output_wavs/ta_vits_female/ \
    /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/ta/wavs-20k-test-female

python3 scripts/evaluate_f0.py \
    output_wavs/ta_vits_female/ \
    /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/ta/wavs-20k-test-female