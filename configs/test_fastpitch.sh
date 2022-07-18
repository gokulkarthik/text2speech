cp output/ta_fastpitch_indictts_male-July-08-2022_01+12PM-d4b542f/best_model.pth output/store/ta/fastpitch_a1/
cp output/ta_fastpitch_indictts_male-July-08-2022_01+12PM-d4b542f/config.json output/store/ta/fastpitch_a1/
# cp output_vocoder/ta_hifigan_male-July-04-2022_10+18AM-d4b542f/best_model.pth output_vocoder/store/ta/hifigan/
# cp output_vocoder/ta_hifigan_male-July-04-2022_10+18AM-d4b542f/config.json output_vocoder/store/ta/hifigan/

python3 -m TTS.bin.synthesize --text "../../datasets/indictts/ta/metadata_test_male.csv" \
    --model_path output/store/ta/fastpitch_a1/best_model.pth \
    --config_path output/store/ta/fastpitch_a1/config.json \
    --vocoder_path output_vocoder/store/ta/hifigan_fastpitch_a1/checkpoint_157000.pth \
    --vocoder_config_path output_vocoder/store/ta/hifigan_fastpitch_a1/config.json \
    --out_path output_wavs/ta_fastpitch_a1_male
    

# python3 scripts/evaluate_mcd.py \
#     output_wavs/ta_glowtts_female/ \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/ta/wavs-20k-test-female

# python3 scripts/evaluate_f0.py \
#     output_wavs/ta_glowtts_female/ \
#     /nlsasfs/home/ai4bharat/manidl/ttsteam/datasets/indictts/ta/wavs-20k-test-female