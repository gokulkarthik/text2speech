cp output/ta_vits_indictts_female-June-28-2022_06+34AM-4373f00/best_model.pth output/store/ta/vits/
cp output/ta_vits_indictts_female-June-28-2022_06+34AM-4373f00/config.json output/store/ta/vits/

python3 -m TTS.bin.synthesize --text "../../datasets/indictts/ta/metadata_test_female.csv" \
    --model_path output/store/ta/vits/best_model.pth \
    --config_path output/store/ta/vits/config.json \
    --out_path output_wavs/ta_vits_female

# python3 scripts/evaluate_mcd.py \
#     output_wavs/ta_vits_female/ \
#     ../../datasets/indictts/ta/wavs-20k-test-female

# python3 scripts/evaluate_f0.py \
#     output_wavs/ta_vits_female/ \
#     ../..//datasets/indictts/ta/wavs-20k-test-female