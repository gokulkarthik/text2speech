python3 -m TTS.bin.synthesize --text "நேஷனல் ஹெரால்ட் ஊழல் குற்றச்சாட்டு தொடர்பாக, காங்கிரஸ் நாடாளுமன்ற உறுப்பினர் ராகுல் காந்தியிடம், அமலாக்கத்துறை, திங்கள் கிழமையன்று பத்து மணி நேரத்திற்கும் மேலாக விசாரணை நடத்திய நிலையில், செவ்வாய்க்கிழமை மீண்டும் விசாரணைக்கு ஆஜராகிறார்." \
    --model_path output/store/ta/glowtts/best_model.pth \
    --config_path output/store/ta/glowtts/config.json \
    --vocoder_path output_vocoder/store/ta/hifigan/best_model.pth \
    --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
    --speakers_file_path output/store/ta/glowtts/speakers.json \
    --speaker_idx male \
    --out_path output/store/wavs/ta-glowtts-eg1-male.wav

python3 -m TTS.bin.synthesize --text "நேஷனல் ஹெரால்ட் ஊழல் குற்றச்சாட்டு தொடர்பாக, காங்கிரஸ் நாடாளுமன்ற உறுப்பினர் ராகுல் காந்தியிடம், அமலாக்கத்துறை, திங்கள் கிழமையன்று பத்து மணி நேரத்திற்கும் மேலாக விசாரணை நடத்திய நிலையில், செவ்வாய்க்கிழமை மீண்டும் விசாரணைக்கு ஆஜராகிறார்." \
    --model_path output/store/ta/glowtts/best_model.pth \
    --config_path output/store/ta/glowtts/config.json \
    --vocoder_path output_vocoder/store/ta/hifigan/best_model.pth \
    --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
    --speakers_file_path output/store/ta/glowtts/speakers.json \
    --speaker_idx female \
    --out_path output/store/wavs/ta-glowtts-eg1-female.wav

python3 -m TTS.bin.synthesize --text "ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின் பேரிலும் நுட்பமாகவும் நடத்துகிறார்." \
    --model_path output/store/ta/glowtts/best_model.pth \
    --config_path output/store/ta/glowtts/config.json \
    --vocoder_path output_vocoder/store/ta/hifigan/best_model.pth \
    --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
    --speakers_file_path output/store/ta/glowtts/speakers.json \
    --speaker_idx male \
    --out_path output/store/wavs/ta-glowtts-eg2-male.wav

python3 -m TTS.bin.synthesize --text "ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின் பேரிலும் நுட்பமாகவும் நடத்துகிறார்." \
    --model_path output/store/ta/glowtts/best_model.pth \
    --config_path output/store/ta/glowtts/config.json \
    --vocoder_path output_vocoder/store/ta/hifigan/best_model.pth \
    --vocoder_config_path output_vocoder/store/ta/hifigan/config.json \
    --speakers_file_path output/store/ta/glowtts/speakers.json \
    --speaker_idx female \
    --out_path output/store/wavs/ta-glowtts-eg2-female.wav


# python3 -m TTS.bin.synthesize --text "நேஷனல் ஹெரால்ட் ஊழல் குற்றச்சாட்டு தொடர்பாக, காங்கிரஸ் நாடாளுமன்ற உறுப்பினர் ராகுல் காந்தியிடம், அமலாக்கத்துறை, திங்கள் கிழமையன்று பத்து மணி நேரத்திற்கும் மேலாக விசாரணை நடத்திய நிலையில், செவ்வாய்க்கிழமை மீண்டும் விசாரணைக்கு ஆஜராகிறார்." \
#     --model_path output/store/ta/vits/best_model.pth \
#     --config_path output/store/ta/vits/config.json \
#     --out_path output/store/wavs/ta-vits-eg1.wav

# python3 -m TTS.bin.synthesize --text "ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின் பேரிலும் நுட்பமாகவும் நடத்துகிறார்." \
#     --model_path output/store/ta/vits/best_model.pth \
#     --config_path output/store/ta/vits/config.json \
#     --out_path output/store/wavs/ta-vits-eg2.wav

# python3 -m TTS.bin.synthesize --text "நேஷனல் ஹெரால்ட் ஊழல் குற்றச்சாட்டு தொடர்பாக, காங்கிரஸ் நாடாளுமன்ற உறுப்பினர் ராகுல் காந்தியிடம், அமலாக்கத்துறை, திங்கள் கிழமையன்று பத்து மணி நேரத்திற்கும் மேலாக விசாரணை நடத்திய நிலையில், செவ்வாய்க்கிழமை மீண்டும் விசாரணைக்கு ஆஜராகிறார்." \
#     --model_path output/store/ta/vits/last_model.pth \
#     --config_path output/store/ta/vits/config.json \
#     --out_path output/store/wavs/ta-vits_last-eg1.wav

# python3 -m TTS.bin.synthesize --text "ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின் பேரிலும் நுட்பமாகவும் நடத்துகிறார்." \
#     --model_path output/store/ta/vits/last_model.pth \
#     --config_path output/store/ta/vits/config.json \
#     --out_path output/store/wavs/ta-vits_last-eg2.wav


# python3 -m TTS.bin.synthesize --text "बिहार, राजस्थान और उत्तर प्रदेश से लेकर हरियाणा, मध्य प्रदेश एवं उत्तराखंड में सेना में भर्ती से जुड़ी 'अग्निपथ स्कीम' का विरोध जारी है." \
#     --model_path output/store/hi/glowtts/best_model.pth \
#     --config_path output/store/hi/glowtts/config.json \
#     --speakers_file_path output/store/hi/glowtts/speakers.json \
#     --speaker_idx male \
#     --out_path output/store/wavs/hi-glowtts-eg1-male.wav

# python3 -m TTS.bin.synthesize --text "बिहार, राजस्थान और उत्तर प्रदेश से लेकर हरियाणा, मध्य प्रदेश एवं उत्तराखंड में सेना में भर्ती से जुड़ी 'अग्निपथ स्कीम' का विरोध जारी है." \
#     --model_path output/store/hi/glowtts/best_model.pth \
#     --config_path output/store/hi/glowtts/config.json \
#     --speakers_file_path output/store/hi/glowtts/speakers.json \
#     --speaker_idx female \
#     --out_path output/store/wavs/hi-glowtts-eg1-female.wav

# python3 -m TTS.bin.synthesize --text "संयुक्त अरब अमीरात यानी यूएई ने बुधवार को एक फ़ैसला लिया कि अगले चार महीनों तक वो भारत से ख़रीदा हुआ गेहूँ को किसी और को नहीं बेचेगा." \
#     --model_path output/store/hi/glowtts/best_model.pth \
#     --config_path output/store/hi/glowtts/config.json \
#     --speakers_file_path output/store/hi/glowtts/speakers.json \
#     --speaker_idx male \
#     --out_path output/store/wavs/hi-glowtts-eg2-male.wav

# python3 -m TTS.bin.synthesize --text "संयुक्त अरब अमीरात यानी यूएई ने बुधवार को एक फ़ैसला लिया कि अगले चार महीनों तक वो भारत से ख़रीदा हुआ गेहूँ को किसी और को नहीं बेचेगा." \
#     --model_path output/store/hi/glowtts/best_model.pth \
#     --config_path output/store/hi/glowtts/config.json \
#     --speakers_file_path output/store/hi/glowtts/speakers.json \
#     --speaker_idx female \
#     --out_path output/store/wavs/hi-glowtts-eg2-female.wav


# python3 -m TTS.bin.synthesize --text "बिहार, राजस्थान और उत्तर प्रदेश से लेकर हरियाणा, मध्य प्रदेश एवं उत्तराखंड में सेना में भर्ती से जुड़ी 'अग्निपथ स्कीम' का विरोध जारी है." \
#     --model_path output/store/hi/vits/best_model.pth \
#     --config_path output/store/hi/vits/config.json \
#     --out_path output/store/wavs/hi-vits-eg1.wav

# python3 -m TTS.bin.synthesize --text "संयुक्त अरब अमीरात यानी यूएई ने बुधवार को एक फ़ैसला लिया कि अगले चार महीनों तक वो भारत से ख़रीदा हुआ गेहूँ को किसी और को नहीं बेचेगा." \
#     --model_path output/store/hi/vits/best_model.pth \
#     --config_path output/store/hi/vits/config.json \
#     --out_path output/store/wavs/hi-vits-eg2.wav

# python3 -m TTS.bin.synthesize --text "बिहार, राजस्थान और उत्तर प्रदेश से लेकर हरियाणा, मध्य प्रदेश एवं उत्तराखंड में सेना में भर्ती से जुड़ी 'अग्निपथ स्कीम' का विरोध जारी है." \
#     --model_path output/store/hi/vits/last_model.pth \
#     --config_path output/store/hi/vits/config.json \
#     --out_path output/store/wavs/hi-vits_last-eg1.wav

# python3 -m TTS.bin.synthesize --text "संयुक्त अरब अमीरात यानी यूएई ने बुधवार को एक फ़ैसला लिया कि अगले चार महीनों तक वो भारत से ख़रीदा हुआ गेहूँ को किसी और को नहीं बेचेगा." \
#     --model_path output/store/hi/vits/last_model.pth \
#     --config_path output/store/hi/vits/config.json \
#     --out_path output/store/wavs/hi-vit_lasts-eg2.wav