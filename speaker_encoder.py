import argparse
import json
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from TTS.config import load_config
from TTS.encoder.models.resnet import ResNetSpeakerEncoder
from TTS.tts.configs.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.datasets import TTSDataset, load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


def get_arg_parser():
    parser = argparse.ArgumentParser(description='Training and evaluation script for speaker enocder model ')

    # dataset parameters
    parser.add_argument('--dataset_name', default='googletts', choices=['googletts'])
    parser.add_argument('--language', default='ta', choices=['en', 'ta', 'hi'])
    parser.add_argument('--dataset_path', default='../../datasets/{}/{}', type=str)   

    # multi-speaker parameters
    parser.add_argument('--multispeaker_model_path', default='output/store/ta/fastpitch_multi/best_model.pth', type=str)
    parser.add_argument('--multispeaker_config_path', default='output/store/ta/fastpitch_multi/config.json', type=str)
    parser.add_argument('--multispeaker_id_path', default='output/store/ta/fastpitch_multi/speakers.json', type=str)

    # training parameters
    parser.add_argument('--gpus', default='0', help='GPU ids concatenated with space')
    parser.add_argument('--strategy', default=None)
    parser.add_argument('--limit_train_batches', default=1.0)
    parser.add_argument('--limit_val_batches', default=1.0)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--max_epochs', default=1000, type=int)
    parser.add_argument('--log_every_n_steps', type=int, default=50)
    parser.add_argument('--val_check_interval', default=1.0)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--loss_fn_name', default='cosine', choices=['l1', 'l2', 'cosine'])
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--gradient_clip_val', type=float, default=0.1)

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


class ZeroShotSpeakerEncoder(pl.LightningModule):

    def __init__(self, speaker_embedding_layer, speaker_encoder, args):
        super().__init__()
        self.speaker_embedding_layer = speaker_embedding_layer
        self.speaker_encoder = speaker_encoder
        self.loss_fn_name = args.loss_fn_name
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        if self.loss_fn_name == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.loss_fn_name == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.loss_fn_name == 'cosine':
            self.criterion = torch.nn.CosineEmbeddingLoss()

    def forward(self, batch):
        speaker_embeddings_pred = self.speaker_encoder(batch['mel'].transpose(1,2))
        return speaker_embeddings_pred

    def common_step(self, batch, batch_idx):
        speaker_embeddings_gt = self.speaker_embedding_layer(batch['speaker_ids'])
        speaker_embeddings_pred = self.speaker_encoder(batch['mel'].transpose(1,2))
        if self.loss_fn_name == 'cosine':
            loss = self.criterion(speaker_embeddings_gt, speaker_embeddings_pred, torch.ones_like(speaker_embeddings_gt[:,0]))
        else:
            loss = self.criterion(speaker_embeddings_gt, speaker_embeddings_pred)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log('val/loss', loss)
        return loss

    def configure_optimizers(self):
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]}
            ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

        return optimizer



def main(args):

    # load speaker embedding
    speaker_embedding_values = torch.load(args.multispeaker_model_path)['model']['emb_g.weight']
    num_speaker_features = speaker_embedding_values.shape[1]
    print("Speaker Embeddings Shape:", speaker_embedding_values.shape)
    speaker_embedding_layer = nn.Embedding(*speaker_embedding_values.shape)
    speaker_embedding_layer.weight = nn.Parameter(speaker_embedding_values)

    with open(args.multispeaker_id_path) as f:
        speaker_id_mapping = json.load(f)
    print("Speaker ID Map:", speaker_id_mapping)

    # setup ap and tokenizer
    tts_config = load_config(args.multispeaker_config_path)
    ap = AudioProcessor.init_from_config(tts_config)
    tokenizer, tts_config = TTSTokenizer.init_from_config(tts_config)

    # load data
    dataset_config = BaseDatasetConfig(
        name=args.dataset_name, 
        meta_file_train="metadata_train.csv", 
        meta_file_val="metadata_test.csv",
        path=args.dataset_path, 
        language=args.language
    )

    train_samples, eval_samples = load_tts_samples(
        dataset_config, 
        eval_split=True,
        formatter=formatter_indictts
    )
    print("Train Samples: ", len(train_samples))
    print("Eval Samples: ", len(eval_samples))
    print("Sample: ", train_samples[0])

    train_dataset = TTSDataset(
        outputs_per_step= 1,
        compute_linear_spec=False,
        ap=ap,
        tokenizer=tokenizer,
        samples=train_samples,
        speaker_id_mapping=speaker_id_mapping,
    )
    eval_dataset = TTSDataset(
        outputs_per_step= 1,
        compute_linear_spec=False,
        ap=ap,
        tokenizer=tokenizer,
        samples=eval_samples,
        speaker_id_mapping=speaker_id_mapping,
    )
    print("Dataset Sample: ", train_dataset[0])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=train_dataset.collate_fn,
        shuffle=False,
        drop_last=False
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=train_dataset.collate_fn,
        shuffle=False,
        drop_last=False
    )
    sample_batch = next(iter(train_loader))
    print("Dataloader Sample (keys): ", sample_batch.keys())
    print("Mel Shape: ", sample_batch['mel'].shape)
    
    # setup speaker encoder
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    speaker_encoder = ResNetSpeakerEncoder(
        input_dim=80,
        proj_dim=num_speaker_features,
        layers=[3, 4, 6, 3],
        num_filters=[32, 64, 128, 256],
        encoder_type="ASP",
        log_input=False,
        use_torch_spec=False,
        audio_config=BaseAudioConfig(
            trim_db=60.0,
            mel_fmin=0.0,
            mel_fmax=8000,
            log_func="np.log",
            spec_gain=1.0,
            signal_norm=False,
        ),
    )
    speaker_encoder = speaker_encoder.to(device)

    # test forward pass
    speaker_embeddings_gt = speaker_embedding_layer(sample_batch['speaker_ids'].to(device))
    print("GT Shape:", speaker_embeddings_gt.shape)
    speaker_embeddings_pred = speaker_encoder(sample_batch['mel'].transpose(1,2).to(device))
    print("Pred Shape:", speaker_embeddings_pred.shape)

    # setup model
    seed_everything(42, workers=True)
    model = ZeroShotSpeakerEncoder(speaker_embedding_layer=speaker_embedding_layer, speaker_encoder=speaker_encoder, args=args)

    # setup trainer
    wandb_logger = WandbLogger(project='speaker_encoder', config=args)
    checkpoint_callback = ModelCheckpoint(dirpath='output_speaker_encoder', filename=wandb_logger.experiment.name+'-{epoch:02d}',  monitor='val/loss', mode='min', verbose=True, save_weights_only=True, save_top_k=1, save_last=True)
    trainer = Trainer(gpus=args.gpus, max_epochs=args.max_epochs, gradient_clip_val=args.gradient_clip_val, 
        logger=wandb_logger, log_every_n_steps=args.log_every_n_steps, val_check_interval=args.val_check_interval,
        strategy=args.strategy, callbacks=[checkpoint_callback],
        limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches,
        deterministic=False)

    # train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

    parser = get_arg_parser()
    args = parser.parse_args()

    args.dataset_path = args.dataset_path.format(args.dataset_name, args.language)
    if args.dataset_name == 'googletts':
        args.dataset_path += '/processed'
    args.gpus = [int(id_) for id_ in args.gpus.split()]
    if args.strategy == 'ddp':
        args.strategy = DDPPlugin(find_unused_parameters=False)
    elif args.strategy == 'none':
        args.strategy = None


    main(args)