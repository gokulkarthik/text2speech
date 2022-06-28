import argparse
import os
import string

import pandas as pd

from argparse import Namespace
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseAudioConfig, BaseDatasetConfig, BaseTTSConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.models.vits import Vits, VitsArgs
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from utils import str2bool


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Traning and evaluation script for acoustic / e2e TTS model ')

    # dataset parameters
    parser.add_argument('--dataset_name', default='indictts', choices=['ljspeech', 'indictts'])
    parser.add_argument('--dataset_path', default='../../datasets/indictts/{}', type=str)
    parser.add_argument('--language', default='ta', choices=['en', 'ta', 'hi'])
    parser.add_argument('--speaker', default='all') # eg. all, male, female, tamilmale, tamilfemale, hindimale, ...
    parser.add_argument('--use_phonemes', default=False, type=str2bool)
    parser.add_argument('--phoneme_language', default='en-us', choices=['en-us'])
    parser.add_argument('--add_blank', default=False, type=str2bool)
    parser.add_argument('--text_cleaner', default='multilingual_cleaners', choices=['multilingual_cleaners'])
    parser.add_argument('--eval_split_size', default=0.01)
    parser.add_argument('--min_audio_len', default=1)
    parser.add_argument('--max_audio_len', default=20*22050)
    parser.add_argument('--min_text_len', default=1)
    parser.add_argument('--max_text_len', default=400)

    # model parameters
    parser.add_argument('--model', default='vits', choices=['glowtts', 'vits'])
    parser.add_argument('--use_speaker_embedding', default=True, type=str2bool)

    # training parameters
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--batch_size_eval', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_workers_eval', default=8, type=int)
    parser.add_argument('--mixed_precision', default=False, type=str2bool)
    parser.add_argument('--compute_input_seq_cache', default=False, type=str2bool)

    # training - logging parameters 
    parser.add_argument('--run_description', default='None', type=str)
    parser.add_argument('--output_path', default='output', type=str)
    parser.add_argument('--test_delay_epochs', default=0, type=int)   
    parser.add_argument('--print_step', default=25, type=int)
    parser.add_argument('--plot_step', default=100, type=int)
    parser.add_argument('--save_step', default=1000, type=int)
    parser.add_argument('--save_n_checkpoints', default=5, type=int)
    parser.add_argument('--save_best_after', default=1000, type=int)
    parser.add_argument('--target_loss', default=None)
    parser.add_argument('--print_eval', default=False, type=str2bool)
    parser.add_argument('--run_eval', default=True, type=str2bool)
    
    # distributed training parameters
    parser.add_argument('--port', default=54321, type=int)
    parser.add_argument('--continue_path', default="", type=str)
    parser.add_argument('--restore_path', default="", type=str)
    parser.add_argument('--group_id', default="", type=str)
    parser.add_argument('--use_ddp', default=True, type=bool)
    parser.add_argument('--rank', default=0, type=int)
    #parser.add_argument('--gpus', default='0', type=str)

    # vits
    parser.add_argument('--use_sdp', default=True, type=str2bool)

    return parser


def formatter_indictts(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
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


def filter_speaker(samples, speaker):
    if speaker == 'all':
        return samples
    samples = [sample for sample in samples if sample['speaker_name']==speaker]
    return samples


def get_lang_chars(language):
    if language == 'ta':
        lang_chars_df = pd.read_csv('chars/Characters-Tamil.csv')
        lang_chars = sorted(list(set(list("".join(lang_chars_df['Character'].values.tolist())))))
        print(lang_chars, len(lang_chars))
        print("".join(lang_chars))
        lang_chars_extra = ['ௗ', 'ஹ', 'ஜ', 'ஸ', 'ஷ']
        lang_chars_extra = sorted(list(set(list("".join(lang_chars_extra)))))
        print(lang_chars_extra, len(lang_chars_extra))
        print("".join(lang_chars_extra))
        lang_chars = lang_chars + lang_chars_extra

    elif language == 'hi':
        lang_chars_df = pd.read_csv('chars/Characters-Hindi.csv')
        lang_chars = sorted(list(set(list("".join(lang_chars_df['Character'].values.tolist())))))
        print(lang_chars, len(lang_chars))
        print("".join(lang_chars))
        lang_chars_extra = []
        lang_chars_extra = sorted(list(set(list("".join(lang_chars_extra)))))
        print(lang_chars_extra, len(lang_chars_extra))
        print("".join(lang_chars_extra))
        lang_chars = lang_chars + lang_chars_extra

    elif language == 'en':
        lang_chars = string.ascii_lowercase

    return lang_chars


def get_test_sentences(language):
    if language == 'ta':
        test_sentences = [
                "நேஷனல் ஹெரால்ட் ஊழல் குற்றச்சாட்டு தொடர்பாக, காங்கிரஸ் நாடாளுமன்ற உறுப்பினர் ராகுல் காந்தியிடம், அமலாக்கத்துறை, திங்கள் கிழமையன்று பத்து மணி நேரத்திற்கும் மேலாக விசாரணை நடத்திய நிலையில், செவ்வாய்க்கிழமை மீண்டும் விசாரணைக்கு ஆஜராகிறார்.",
                "ஒரு விஞ்ஞானி தம் ஆராய்ச்சிகளை எவ்வளவோ கணக்காகவும் முன் யோசனையின் பேரிலும் நுட்பமாகவும் நடத்துகிறார்.",
            ]

    elif language == 'hi':
        test_sentences = [
                "बिहार, राजस्थान और उत्तर प्रदेश से लेकर हरियाणा, मध्य प्रदेश एवं उत्तराखंड में सेना में भर्ती से जुड़ी 'अग्निपथ स्कीम' का विरोध जारी है.",
                "संयुक्त अरब अमीरात यानी यूएई ने बुधवार को एक फ़ैसला लिया कि अगले चार महीनों तक वो भारत से ख़रीदा हुआ गेहूँ को किसी और को नहीं बेचेगा.",
            ]

    elif language == 'en':
        test_sentences = [
                "Brazilian police say a suspect has confessed to burying the bodies of missing British journalist Dom Phillips and indigenous expert Bruno Pereira.",
                "Protests have erupted in India over a new reform scheme to hire soldiers for a fixed term for the armed forces",
            ]

    return test_sentences
    

def main(args):

    # set dataset config
    dataset_config = BaseDatasetConfig(
        name=args.dataset_name, 
        meta_file_train=f"metadata_train_{args.speaker}.csv", 
        meta_file_val=f"metadata_test_{args.speaker}.csv", 
        path=args.dataset_path.format(args.language), 
        language=args.language
    )

    #lang_chars = get_lang_chars(args.language)
    samples, _ = load_tts_samples(
        dataset_config, 
        eval_split=False,
        formatter=formatter_indictts)
    samples = filter_speaker(samples, args.speaker)
    texts = "".join(item["text"] for item in samples)
    lang_chars = sorted(list(set(texts)))
    print(lang_chars, len(lang_chars))
    del samples

    # set audio config
    if args.model == 'glowtts':
        audio_config = BaseAudioConfig()
    elif args.model == 'vits':
        audio_config = BaseAudioConfig(
            log_func="np.log",
            spec_gain=1.0,
            signal_norm=False,
            do_amp_to_db_linear=False,
        )

    # set characters config
    if args.model == 'glowtts':
        from TTS.tts.configs.shared_configs import CharactersConfig
        characters_config = CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsCharacters",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
            #characters="!¡'(),-.:;¿?$%&‘’‚“`”„" + "".join(lang_chars),
            characters="".join(lang_chars),
            punctuations="!¡'(),-.:;¿? ",
            phonemes=None
        )
    elif args.model == 'vits':
        from TTS.tts.models.vits import CharactersConfig
        characters_config = CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsCharacters",
            pad="<PAD>",
            eos="<EOS>",
            bos="<BOS>",
            blank="<BLNK>",
            #characters="!¡'(),-.:;¿?$%&‘’‚“`”„" + "".join(lang_chars),
            characters="".join(lang_chars),
            punctuations="!¡'(),-.:;¿? ",
            phonemes=None
        )

    # set base tts config
    base_tts_config = Namespace(
        # input representation
        audio=audio_config,
        use_phonemes=args.use_phonemes,
        phoneme_language=args.phoneme_language,
        compute_input_seq_cache=args.compute_input_seq_cache,
        text_cleaner=args.text_cleaner,
        phoneme_cache_path=os.path.join(args.output_path, "phoneme_cache"),
        characters=characters_config,
        add_blank=args.add_blank,
        # dataset
        datasets=[dataset_config],
        min_audio_len=args.min_audio_len,
        max_audio_len=args.max_audio_len,
        min_text_len=args.min_text_len,
        max_text_len=args.max_text_len,
        # data loading
        num_loader_workers=args.num_workers,
        num_eval_loader_workers=args.num_workers_eval,
        # trainer - run
        output_path=args.output_path,
        project_name='acoustic_model',
        run_name=f'{args.language}_{args.model}_{args.dataset_name}_{args.speaker}',
        run_description=args.run_description,
        # trainer - loggging
        print_step=args.print_step,
        plot_step=args.plot_step,
        dashboard_logger='wandb',
        wandb_entity='gokulkarthik',
        # trainer - checkpointing
        save_step=args.save_step,
        save_n_checkpoints=args.save_n_checkpoints,
        save_best_after=args.save_best_after,
        # trainer - eval
        print_eval=args.print_eval,
        run_eval=args.run_eval,
        # trainer - test
        test_delay_epochs=args.test_delay_epochs,
        # trainer - distibuted training
        distributed_url=f'tcp://localhost:{args.port}',
        # trainer - training
        mixed_precision=args.mixed_precision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size_eval,
        # test
        #test_sentences_file=f'test_sentences/{args.language}.txt',
        test_sentences=get_test_sentences(args.language),
        eval_split_size=args.eval_split_size,
    )
    base_tts_config = vars(base_tts_config)

    # set model config 
    if args.model == 'glowtts':
        config = GlowTTSConfig(
            **base_tts_config,
            use_speaker_embedding=args.use_speaker_embedding,
        )
    elif args.model == "vits":
        vitsArgs = VitsArgs(
            use_speaker_embedding=args.use_speaker_embedding,
            use_sdp=args.use_sdp,
        )
        config = VitsConfig(
            **base_tts_config,
            model_args=vitsArgs,
            use_speaker_embedding=args.use_speaker_embedding,   
        )

    # set preprocessors
    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # load data
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        #eval_split_size=config.eval_split_size,
        formatter=formatter_indictts
    )
    train_samples = filter_speaker(train_samples, args.speaker)
    eval_samples = filter_speaker(eval_samples, args.speaker)
    print("Train Samples: ", len(train_samples))
    print("Eval Samples: ", len(eval_samples))
    
    # set speaker manager
    if args.speaker == 'all':
        speaker_manager = SpeakerManager()
        speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
    else:
        speaker_manager = None
    
   
    # load model
    if args.model == 'glowtts':
        model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)
        if args.speaker == 'all':
            config.num_speakers = speaker_manager.num_speakers
    elif args.model == 'vits':
        model = Vits(config, ap, tokenizer, speaker_manager=speaker_manager)
        if args.speaker == 'all':
            config.num_speakers = speaker_manager.num_speakers
            config.model_args.num_speakers = speaker_manager.num_speakers

    # set trainer
    trainer = Trainer(
        TrainerArgs(continue_path=args.continue_path, restore_path=args.restore_path, use_ddp=args.use_ddp, rank=args.rank, group_id=args.group_id), 
        config, 
        args.output_path, 
        model=model, 
        train_samples=train_samples, 
        eval_samples=eval_samples
    )

    # run training
    trainer.fit()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    parser = get_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
