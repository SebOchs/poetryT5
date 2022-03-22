import sys
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from poetryT5.litByT5 import LitPoetryT5


def finetuning(batch_size=16, epochs=4, acc_grad=8, top_k=3, model_size='small'):
    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath='models/',
        monitor='distance',
        mode='min',
        filename='poetry_t5_{epoch}-{distance:.4f}',
        save_top_k=top_k
    )

    # Initialize model and trainer
    poetry_model = LitPoetryT5(batch_size, model_size)

    trainer = pl.Trainer(
        gpus=1,
        max_epochs=epochs,
        accumulate_grad_batches=acc_grad,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=100,
    )
    trainer.fit(poetry_model)
    print("Best model with batchsize {} and acc_grad {} is: ".format(batch_size, acc_grad) +
          checkpoint_callback.best_model_path)


def main():
    finetuning(batch_size=8, acc_grad=32, epochs=100, top_k=100, model_size='small')

if __name__ == '__main__':
    main()
