# text2speech-ta

Text-to-speech in Tamil using CoquiTTS.

    Acoustic Model: Glow TTS
    Vocoder: Griffin-Lim
    Dataset: AI4B

Reference: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)

1. Format AI4B dataset in LJSpeech format using [dataset_analysis/FormatDatasets.ipynb](./dataset_analysis/FormatDatasets.ipynb)
2. Analyze AI4B dataset to check TTS suitability using [dataset_analysis/AnalyzeDataset.ipynb](./dataset_analysis/AnalyzeDataset.ipynb)
3. Set the configuration with [run.sh](./run.sh) (or/and) [main.py](./main.py)
4. Train the TTS model by executing `sh train.sh`
5. Test the TTS model by executing `sh test.sh`
6. Check the output wav files at [output/github/](./output/github/)