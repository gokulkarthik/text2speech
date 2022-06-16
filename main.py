import argparse
import os

import pandas as pd

from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.encoder.utils.training import init_training
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def str2bool(v):
    """
    src: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse 
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Traning and evaluation script for hateful meme classification')

    # dataset parameters
    parser.add_argument('--dataset_name', default='ai4b-ta', choices=['ljspeech', 'ai4b-ta'])
    parser.add_argument('--dataset_path', default='../../data/tts/ai4b_preprocessed', type=str)
    parser.add_argument('--language', default='ta', choices=['en', 'ta'])

    # model parameters
    parser.add_argument('--model', default='vits', choices=['glowtts', 'vits'])
    parser.add_argument('--use_speaker_embedding', default=True, type=str2bool)

    # training parameters
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--batch_size_eval', default=16, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--num_workers_eval', default=16, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--use_phonemes', default=False, type=str2bool)
    parser.add_argument('--phoneme_language', default='ta', choices=['en-us', 'ta'])
    parser.add_argument('--print_step', default=25, type=int)
    parser.add_argument('--print_eval', default=False, type=str2bool)
    parser.add_argument('--mixed_precision', default=False, type=str2bool)
    parser.add_argument('--output_path', default='output', type=str)
    parser.add_argument('--save_step', default=1000, type=int)

    # distributed training parameters
    parser.add_argument('--continue_path', default="", type=str)
    parser.add_argument('--restore_path', default="", type=str)
    parser.add_argument('--group_id', default="", type=str)
    parser.add_argument('--use_ddp', default=False, type=bool)
    parser.add_argument('--rank', default=0, type=int)


    return parser

def formatter_ai4b(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs-20k", cols[0] + ".wav")
            text = cols[1].strip()
            speaker_name = cols[2].strip()
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name})
    return items

def main(args):
    tamil_chars_df = pd.read_csv('Characters-Tamil.csv')
    tamil_chars = sorted(list(set(list("".join(tamil_chars_df['Character'].values.tolist())))))
    print(tamil_chars, len(tamil_chars))
    print("".join(tamil_chars))
    tamil_chars_extra = ['ௗ', 'ஹ', 'ஜ', 'ஸ', 'ஷ']
    tamil_chars_extra = sorted(list(set(list("".join(tamil_chars_extra)))))
    print(tamil_chars_extra, len(tamil_chars_extra))
    print("".join(tamil_chars_extra))
    

    dataset_config = BaseDatasetConfig(
        name=args.dataset_name, meta_file_train="metadata.csv", path=args.dataset_path, language=args.language
    )

    if args.model == 'glowtts':
        from TTS.tts.configs.shared_configs import CharactersConfig

        config = GlowTTSConfig(
            batch_size=args.batch_size,
            eval_batch_size=args.batch_size_eval,
            num_loader_workers=args.num_workers,
            num_eval_loader_workers=args.num_workers_eval,
            run_name=f"glowtts_{args.dataset_name}",
            run_eval=True,
            test_delay_epochs=-1,
            epochs=args.epochs,
            text_cleaner="multilingual_cleaners",
            use_phonemes=args.use_phonemes,
            phoneme_language=args.phoneme_language,
            phoneme_cache_path=os.path.join(args.output_path, "phoneme_cache"),
            print_step=args.print_step,
            print_eval=args.print_eval,
            mixed_precision=args.mixed_precision,
            output_path=args.output_path,
            datasets=[dataset_config],
            save_step=args.save_step,
            characters=CharactersConfig(
                characters_class="TTS.tts.models.vits.VitsCharacters",
                pad="<PAD>",
                eos="<EOS>",
                bos="<BOS>",
                blank="<BLNK>",
                characters="!¡'(),-.:;¿?$%&‘’‚“`”„" + "".join(tamil_chars) + "".join(tamil_chars_extra),
                punctuations="!¡'(),-.:;¿? ",
                phonemes=None,
            ),
            test_sentences=[
                "நேஷனல் ஹெரால்ட் ஊழல் குற்றச்சாட்டு தொடர்பாக, காங்கிரஸ் நாடாளுமன்ற உறுப்பினர் ராகுல் காந்தியிடம், அமலாக்கத்துறை, திங்கள் கிழமையன்று பத்து மணி நேரத்திற்கும் மேலாக விசாரணை நடத்திய நிலையில், செவ்வாய்க்கிழமை மீண்டும் விசாரணைக்கு ஆஜராகிறார்.",
                "ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின் பேரிலும் நுட்பமாகவும் நடத்துகிறார்.",
            ],
            use_speaker_embedding=args.use_speaker_embedding,
            #dashboard_logger = 'wandb'
        )

    elif args.model == "vits":

        from TTS.tts.models.vits import CharactersConfig

        audio_config = BaseAudioConfig(
            win_length=1024,
            hop_length=256,
            num_mels=80,
            preemphasis=0.0,
            ref_level_db=20,
            log_func="np.log",
            do_trim_silence=True,
            trim_db=45.0,
            mel_fmin=0,
            mel_fmax=None,
            spec_gain=1.0,
            signal_norm=False,
            do_amp_to_db_linear=False,
        )

        # vitsArgs = VitsArgs(
        #     # use_language_embedding=True,
        #     # embedded_language_dim=4,
        #     use_speaker_embedding=True,
        #     use_sdp=False,
        # )

        config = VitsConfig(
            #model_args=vitsArgs,
            audio=audio_config,
            run_name=f"vits_{args.dataset_name}",
            use_speaker_embedding=args.use_speaker_embedding,
            batch_size=args.batch_size,
            eval_batch_size=args.batch_size_eval,
            batch_group_size=5,
            num_loader_workers=args.num_workers,
            num_eval_loader_workers=args.num_workers_eval,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=args.epochs,
            text_cleaner="multilingual_cleaners",
            use_phonemes=args.use_phonemes,
            phoneme_language=args.phoneme_language,
            phoneme_cache_path=os.path.join(args.output_path, "phoneme_cache"),
            compute_input_seq_cache=True,
            print_step=args.print_step,
            print_eval=args.print_eval,
            mixed_precision=args.mixed_precision,
            output_path=args.output_path,
            datasets=[dataset_config],
            characters=CharactersConfig(
                characters_class="TTS.tts.models.vits.VitsCharacters",
                pad="<PAD>",
                eos="<EOS>",
                bos="<BOS>",
                blank="<BLNK>",
                characters="!¡'(),-.:;¿?$%&‘’‚“`”„" + "".join(tamil_chars) + "".join(tamil_chars_extra),
                punctuations="!¡'(),-.:;¿? ",
                phonemes=None,
            ),
            test_sentences=[
                ["நேஷனல் ஹெரால்ட் ஊழல் குற்றச்சாட்டு தொடர்பாக, காங்கிரஸ் நாடாளுமன்ற உறுப்பினர் ராகுல் காந்தியிடம், அமலாக்கத்துறை, திங்கள் கிழமையன்று பத்து மணி நேரத்திற்கும் மேலாக விசாரணை நடத்திய நிலையில், செவ்வாய்க்கிழமை மீண்டும் விசாரணைக்கு ஆஜராகிறார்.", "female", None, "ta"],
                ["ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின் பேரிலும் நுட்பமாகவும் நடத்துகிறார்.", "male", None, "ta"],
            ],
            #dashboard_logger = 'wandb'
        )


    # load preprocessors
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # load data
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
        formatter=formatter_ai4b
    )

    speaker_manager = SpeakerManager()
    speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
   

    # load model
    if args.model == 'glowtts':
        model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)
        config.num_speakers = speaker_manager.num_speakers
    elif args.model == 'vits':
        model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)
        config.model_args.num_speakers = speaker_manager.num_speakers

    # set trainer
    # trainer_args, config, output_path, _, c_logger, wandb_logger = init_training(config)
    # trainer = Trainer(
    #     trainer_args, config, args.output_path, c_logger, wandb_logger, model=model, train_samples=train_samples, eval_samples=eval_samples
    # )
    trainer = Trainer(
        TrainerArgs(), config, args.output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # run training
    trainer.fit()


if __name__ == '__main__':
    parser = get_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)