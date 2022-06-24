import argparse
import os
from ossaudiodev import SNDCTL_SEQ_RESETSAMPLES

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

from utils import str2bool

def get_arg_parser():
    parser = argparse.ArgumentParser(description='Traning and evaluation script for vocoder model ')

    # dataset parameters
    parser.add_argument('--dataset_name', default='indictts', type=str)
    parser.add_argument('--dataset_path', default='../../datasets/indictts/{}/wavs-20k', type=str)
    parser.add_argument('--language', default='ta', choices=['en', 'ta', 'hi'])
    parser.add_argument('--speaker', default='all') # eg. all, female, male
    parser.add_argument('--eval_split_size', default=10, type=int)

    # model parameters
    parser.add_argument('--model', default='hifigan', choices=['hifigan'])
    parser.add_argument('--seq_len', default=8192, type=int)
    parser.add_argument('--pad_short', default=2000, type=int)
    parser.add_argument('--use_noise_augment', default=True, type=str2bool)

    # training parameters
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--batch_size_eval', default=8, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--num_workers_eval', default=8, type=int)
    parser.add_argument('--lr_gen', default=0.0001, type=float)
    parser.add_argument('--lr_disc', default=0.0001, type=float)
    parser.add_argument('--mixed_precision', default=False, type=str2bool)

    # training - logging parameters 
    parser.add_argument('--run_description', default='None', type=str)
    parser.add_argument('--output_path', default='output_vocoder', type=str)
    parser.add_argument('--test_delay_epochs', default=0, type=int)
    parser.add_argument('--print_step', default=25, type=int)
    parser.add_argument('--plot_step', default=25, type=int)
    parser.add_argument('--save_step', default=1000, type=int)
    parser.add_argument('--save_n_checkpoints', default=5, type=int)
    parser.add_argument('--save_best_after', default=1000, type=int)
    parser.add_argument('--target_loss', default='loss_1')
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

    return parser

def filter_speaker(samples, speaker, dataset_name='indictts', language='ta'):
    if speaker == 'all':
        return samples
    if dataset_name == 'indictts':
        if args.language in  ['ta', 'hi']:
            start_idx = 5
        samples = [sample for sample in samples if sample.rsplit('/', 1)[-1].split('_')[1][start_idx:]==speaker]
    return samples

def main(args):

    config = HifiganConfig(
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size_eval,
        num_loader_workers=args.num_workers,
        num_eval_loader_workers=args.num_workers_eval,
        run_eval=args.run_eval,
        test_delay_epochs=args.test_delay_epochs,
        save_step=args.save_step,
        save_best_after=args.save_best_after,
        save_n_checkpoints=args.save_n_checkpoints,
        target_loss=args.target_loss,
        epochs=args.epochs,
        seq_len=args.seq_len,
        pad_short=args.pad_short,
        use_noise_augment=args.use_noise_augment,
        eval_split_size=args.eval_split_size,
        print_step=args.print_step,
        plot_step=args.plot_step,
        print_eval=args.print_eval,
        mixed_precision=args.mixed_precision,
        lr_gen=args.lr_gen,
        lr_disc=args.lr_disc,
        data_path=args.dataset_path.format(args.language),
        #output_path=f'{args.output_path}/{args.language}_{args.model}',
        output_path=args.output_path,
        distributed_url=f'tcp://localhost:{args.port}',
        dashboard_logger='wandb',
        project_name='vocoder',
        run_name=f'{args.language}_{args.model}_{args.speaker}',
        run_description=args.run_description,
        wandb_entity='gokulkarthik'
    )

    ap = AudioProcessor(**config.audio.to_dict())

    eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)
    eval_samples = filter_speaker(eval_samples, args.speaker, dataset_name=args.dataset_name, language=args.language)
    train_samples = filter_speaker(train_samples, args.speaker, dataset_name=args.dataset_name, language=args.language)

    model = GAN(config, ap)

    trainer = Trainer(
        TrainerArgs(continue_path=args.contiue_path, restore_path=args.restore_path, use_ddp=args.use_ddp, rank=args.rank, group_id=args.group_id), 
        config, 
        config.output_path, 
        model=model, 
        train_samples=train_samples, 
        eval_samples=eval_samples
    )
    trainer.fit()


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

    parser = get_arg_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)