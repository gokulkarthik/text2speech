python3 -m TTS.bin.synthesize --text "தீயினாற் சுட்டபுண் உள்ளாறும் ஆறாதே." \
    --model_path output/store/ta/fastpitch/best_model.pth \
    --config_path output/store/ta/fastpitch/config.json \
    --vocoder_path output_vocoder/store/ta/hifigan/best_model.pth \
    --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
    --out_path output_wavs/temp_fp.wav