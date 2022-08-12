# cp output/ta_fastpitch_googletts_all-July-21-2022_12+01PM-11635b8/best_model_423130.pth output/store/ta/fastpitch_multi/
# cp output/ta_fastpitch_googletts_all-July-21-2022_12+01PM-11635b8/config.json output/store/ta/fastpitch_multi/
# cp output/ta_fastpitch_googletts_all-July-21-2022_12+01PM-11635b8/speakers.pth output/store/ta/fastpitch_multi/
# cp output_vocoder/ta_hifigan_all-July-27-2022_06+13AM-d52256a/best_model_94336.pth output_vocoder/store/ta/hifigan_multi/
# cp output_vocoder/ta_hifigan_all-July-27-2022_06+13AM-d52256a/checkpoint_400000.pth output_vocoder/store/ta/hifigan_multi/
# cp output_vocoder/ta_hifigan_all-July-27-2022_06+13AM-d52256a/config.json output_vocoder/store/ta/hifigan_multi/

python3 -m TTS.bin.synthesize --text "../../datasets/googletts/ta/samples.csv" \
    --model_path output/store/ta/fastpitch_multi/best_model.pth \
    --config_path output/store/ta/fastpitch_multi/config.json \
    --vocoder_path output_vocoder/store/ta/hifigan_multi/checkpoint_400000.pth \
    --vocoder_config_path output_vocoder/store/ta/hifigan_multi/config.json \
    --out_path output_wavs/samples_googletts_ta_multi_fastpitch

# python3 scripts/evaluate_mcd.py \
#     output_wavs/ta_male_fastpitch_hifi/ \
#     data_dir/indictts/ta/wavs-20k-test-male

# python3 scripts/evaluate_f0.py \
#     output_wavs/ta_male_fastpitch_hifi/ \
#     /data_dir/indictts/ta/wavs-20k-test-male