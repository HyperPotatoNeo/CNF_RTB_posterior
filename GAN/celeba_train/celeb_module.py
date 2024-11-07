import torch
from torchvision import datasets, transforms
from torchvision.models import resnet101

import pytorch_lightning as pl
import torchmetrics

# pre trained densenet model 
class CelebAClassifier(torch.nn.Module):
    
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = resnet101(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)
        
    def forward(self, x):
        # resize x to 224x224
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear')

        return self.resnet(x)
    

class LightningCelebA(pl.LightningModule):
    def __init__(self, num_classes=2, lr=1e-3):
        super().__init__()

        self.num_classes = num_classes
        self.learning_rate = lr
        self.model = CelebAClassifier(num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters(ignore=['mode'])

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        pred_y = torch.argmax(logits, dim=1)
        
        return loss, y, pred_y

    def training_step(self, batch, batch_idx):
        loss, true_y, pred_y = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        
        self.model.eval()
        with torch.no_grad():
            _, true_y, pred_y = self._shared_step(batch, batch_idx)
        self.train_acc.update(pred_y, true_y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        self.model.train()
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, true_y, pred_y = self._shared_step(batch, batch_idx)
        self.log("valid_loss", loss)
        self.valid_acc.update(pred_y, true_y)
        self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, true_y, pred_y = self._shared_step(batch, batch_idx)
        self.test_acc.update(pred_y, true_y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
