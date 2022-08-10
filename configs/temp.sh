# python3 -m TTS.bin.synthesize --text "நரேந்திர மோதி செஸ் ஒலிம்பியாட் நிகழ்வுக்கு வரும்போது சென்னையில் பலூன் பறக்கத் தடை" \
#     --model_path output/store/ta/fastpitch_multi/best_model_423130.pth \
#     --config_path output/store/ta/fastpitch_multi/config.json \
#     --speaker_idx taf_00008 \
#     --vocoder_path output_vocoder/store/ta/hifigan/checkpoint_633000.pth \
#     --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
#     --out_path output_wavs/temp_hifi.wav
    
# python3 -m TTS.bin.synthesize --text "நரேந்திர மோதி செஸ் ஒலிம்பியாட் நிகழ்வுக்கு வரும்போது சென்னையில் பலூன் பறக்கத் தடை" \
#     --model_path output/store/ta/fastpitch_multi/best_model_423130.pth \
#     --config_path output/store/ta/fastpitch_multi/config.json \
#     --speaker_idx tag_05935 \
#     --vocoder_path output_vocoder/store/ta/hifigan/checkpoint_633000.pth \
#     --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
#     --out_path output_wavs/temp2_hifi.wav

# python3 -m TTS.bin.synthesize --text "நரேந்திர மோதி செஸ் ஒலிம்பியாட் நிகழ்வுக்கு வரும்போது சென்னையில் பலூன் பறக்கத் தடை" \
#     --model_path output/store/ta/fastpitch_multi/best_model_423130.pth \
#     --config_path output/store/ta/fastpitch_multi/config.json \
#     --speaker_idx tag_09720 \
#     --vocoder_path output_vocoder/store/ta/hifigan/checkpoint_633000.pth \
#     --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
#     --out_path output_wavs/temp3_hifi.wav
    
python3 -m TTS.bin.synthesize --text "நரேந்திர மோதி செஸ் ஒலிம்பியாட் நிகழ்வுக்கு வரும்போது சென்னையில் பலூன் பறக்கத் தடை" \
    --model_path output/store/ta/fastpitch_multi/best_model_423130.pth \
    --config_path output/store/ta/fastpitch_multi/config.json \
    --speaker_idx taf_00008 \
    --out_path output_wavs/temp.wav
    
python3 -m TTS.bin.synthesize --text "நரேந்திர மோதி செஸ் ஒலிம்பியாட் நிகழ்வுக்கு வரும்போது சென்னையில் பலூன் பறக்கத் தடை" \
    --model_path output/store/ta/fastpitch_multi/best_model_423130.pth \
    --config_path output/store/ta/fastpitch_multi/config.json \
    --speaker_idx tag_05935 \
    --out_path output_wavs/temp2.wav

python3 -m TTS.bin.synthesize --text "நரேந்திர மோதி செஸ் ஒலிம்பியாட் நிகழ்வுக்கு வரும்போது சென்னையில் பலூன் பறக்கத் தடை" \
    --model_path output/store/ta/fastpitch_multi/best_model_423130.pth \
    --config_path output/store/ta/fastpitch_multi/config.json \
    --speaker_idx tag_09720 \
    --out_path output_wavs/temp3.wav