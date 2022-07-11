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
# 1. Create environment
sudo apt-get install libsndfile1-dev
conda create -n tts-env
conda activate tts-env

# 2. Setup PyTorch
pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# 3. Setup Trainer
git clone https://github.com/gokulkarthik/Trainer 

cd Trainer
pip3 install -e .[all]
cd ..
[or]
cp Trainer/trainer/logging/wandb_logger.py to the local Trainer installation # fixed wandb logger
cp Trainer/trainer/trainer.py to the local Trainer installation # fixed model.module.test_log and added code to log epoch 
add `gpus = [str(gpu) for gpu in gpus]` in line 53 of trainer/distribute.py

# 4. Setup TTS
git clone https://github.com/gokulkarthik/TTS 

cd TTS
pip3 install -e .[all]
cd ..
[or]
cp TTS/TTS/bin/synthesize.py to the local TTS installation # added multiple output support for TTS.bin.synthesis

# 5. Install other requirements
> pip3 install -r requirements.txt
```


### Data Setup:
1. Format IndicTTS dataset in LJSpeech format using [dataset_analysis/FormatDatasets.ipynb](./dataset_analysis/FormatDatasets.ipynb)
2. Analyze IndicTTS dataset to check TTS suitability using [dataset_analysis/AnalyzeDataset.ipynb](./dataset_analysis/AnalyzeDataset.ipynb)

### Running Steps:
1. Set the configuration with [main.py](./main.py), [vocoder.py](./vocoder.py), [configs](./configs) and [run.sh](./run.sh).
2. Train and test by executing `sh run.sh`