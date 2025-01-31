import pytorch_lightning as pl
import torch
from torchvision import transforms

from celeb_module import LightningCelebA
from celeb_data import CelebADataModule_Classifier

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #lmodule = LightningCelebA(num_classes=2, lr=1e-3)
    lmodule = LightningCelebA.load_from_checkpoint("~/scratch/celeba_logs/hair_classifier.ckpt")

    lmodule.to(device)
    lmodule.eval()

    # eval on test set
    data_module = CelebADataModule_Classifier(batch_size=256, num_workers=4)

    data_module.setup()
    train_loader, valid_loader, test_loader = data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()

    val_acc = 0
    for img, label in valid_loader:
        img = img.to(device)
        label = label.to(device)

        #print("img.shape: ", img.shape)
        #print("label.shape: ", label.shape)
        #print("label: ", label)
        with torch.no_grad():
            logits = lmodule(img)
        
        #print("logits.shape: ", logits.shape)

        # get predicted class
        pred = torch.argmax(logits, dim=1)
        
        # get accuracy
        acc = (pred == label).float().mean()
        val_acc += acc.item()
    
    
    val_acc /= len(valid_loader)

    print("Validation Accuracy: ", val_acc)

if __name__ == "__main__":
    main()
