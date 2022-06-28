# text2speech-indian

Text-to-speech in Indian languages using CoquiTTS. The models are trained using IndicTTS dataset.

### Supported Acoustics Models:
1. GlowTTS (Text2Mel)
2. VITS (Text2Speech)

### Supported Vocoders:
1. HiFiGAN (Mel2Speech)

### Supported Languages:
1. Tamil (ta)
2. Hindi (hi)

Reference: [https://github.com/coqui-ai/TTS](https://github.com/coqui-ai/TTS)

### Environment Setup:
```
> conda create -n tts-env
> conda activate tts-env

> pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
> pip3 install -r requirements.txt

> git clone https://github.com/gokulkarthik/Trainer # fixed wandb logger
> cp Trainer/trainer/logging/wandb_logger.py to the local Trainer installation

> git clone https://github.com/gokulkarthik/TTS # added multiple output support for TTS.bin.synthesis
> cp TTS/TTS/bin/synthesize.py to the local TTS installation
```

### Running Steps:
1. Format IndicTTS dataset in LJSpeech format using [dataset_analysis/FormatDatasets.ipynb](./dataset_analysis/FormatDatasets.ipynb)
2. Analyze IndicTTS dataset to check TTS suitability using [dataset_analysis/AnalyzeDataset.ipynb](./dataset_analysis/AnalyzeDataset.ipynb)
3. Set the configuration with [main.py](./main.py), [vocoder.py](./vocoder.py), [configs](./configs) and [run.sh](./run.sh).
4. Train and test by executing `sh run.sh`