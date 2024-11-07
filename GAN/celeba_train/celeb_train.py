import pytorch_lightning as pl
import torch
from torchvision import transforms

from celeb_module import LightningCelebA
from celeb_data import CelebADataModule_Classifier

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

def main():
    torch.manual_seed(1)

    savedir = "/home/mila/m/mohsin.hasan/scratch/celeba_logs"

    # setup lightning module
    model = LightningCelebA(num_classes=3, lr=1e-3)

    data_module = CelebADataModule_Classifier(batch_size=256, num_workers=4)

    # setup trainer
    callbacks = [ModelCheckpoint(
        save_top_k=1, mode='max', monitor="valid_acc", 
        dirpath = savedir)]  # save top 1 model 
                                             
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks = callbacks,
        accelerator="auto",
        devices="auto",
        logger = WandbLogger(project="celeba", name="celeba_classifier", save_dir = savedir),
        log_every_n_steps=100,
        default_root_dir=savedir
    )

    trainer.fit(model = model, datamodule=data_module)
    

if __name__ == "__main__":
    main()