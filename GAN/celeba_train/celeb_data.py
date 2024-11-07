from celeb_module import CelebAClassifier, LightningCelebA
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import pytorch_lightning as pl

DATA_ROOT = '/home/mila/m/mohsin.hasan/scratch/data/'
#DATA_ROOT = "/network/datasets/torchvision/"


#custom_transforms = transforms.Compose([
#    transforms.CenterCrop((160, 160)),
#    transforms.Resize([64, 64]),
#    transforms.ToTensor(),
#    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#])

custom_transforms = transforms.Compose(
            [
                transforms.Resize((64, 64)),  # Resize to 64x64
                transforms.ToTensor()  # Convert to tensor
            ]
)


def get_dataloaders_celeba(batch_size, num_workers=0,
                           train_transforms=custom_transforms,
                           test_transforms=custom_transforms,
                           download=True):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()
        
    def get_smile(attr):
        return attr[31]

    def get_hair_multi(attr):
        # return 0 for black (attr 9), 
        # 1 for blond (attr 10), 
        # 2 for brown (attr 12), 
        # 3 for gray (attr 18), 
        # 4 for other 
        hair_color = None

        if attr[8] == 1:
            hair_color = 0
        elif attr[9] == 1:
            hair_color = 1
        elif attr[11] == 1:
            hair_color = 2
        elif attr[17] == 1:
            hair_color = 3
        else:
            hair_color = 4
        return hair_color

    def get_hair(attr):
        return attr[9]
        """
        hair_color = None
        if attr[8] == 1:
            hair_color = 1            
        elif attr[9] == 1:
            hair_color = 2
        else:
            hair_color = 0
        return hair_color
        """

    train_dataset = datasets.CelebA(root=DATA_ROOT,
                                    split='train',
                                    transform=train_transforms,
                                    target_type='attr',
                                    target_transform=get_hair,
                                    download=download)

    valid_dataset = datasets.CelebA(root=DATA_ROOT,
                                    split='valid',
                                    target_type='attr',
                                    target_transform=get_hair,
                                    transform=test_transforms)

    test_dataset = datasets.CelebA(root=DATA_ROOT,
                                   split='test',
                                   target_type='attr',
                                   target_transform=get_hair,
                                   transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False)
    
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader


class CelebADataModule_Classifier(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = DATA_ROOT, #"/network/datasets/torchvision/",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        shuffle: bool = True,
        persistent_workers: bool = False,
        dummy: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers
        self.dummy = dummy

        # Transforms for train/validation with 64x64 resizing
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),  # Resize to 64x64
                transforms.ToTensor(),  # Convert to tensor
                #transforms.Normalize(
                #    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                #),
            ]
        )

    def setup(self, stage=None):
        if self.dummy:
            print("Setting up CelebA DataModule...")
            self.train_dataset = datasets.FakeData(1000, (3, 64, 64), 1000, transforms.ToTensor())
            self.val_dataset = datasets.FakeData(50, (3, 64, 64), 1000, transforms.ToTensor())
            return
        print("Setting up CelebA DataModule...")
        
        def get_smile(attr):
            return attr[31]

        def get_hair_multi(attr):
            # return 0 for black (attr 9), 
            # 1 for blond (attr 10), 
            # 2 for brown (attr 12), 
            # 3 for gray (attr 18), 
            # 4 for other 
            hair_color = None

            if attr[8] == 1:
                hair_color = 0
            elif attr[9] == 1:
                hair_color = 1
            elif attr[11] == 1:
                hair_color = 2
            elif attr[17] == 1:
                hair_color = 3
            else:
                hair_color = 4
            return hair_color
        
        def get_hair(attr):
            return attr[9]
            """
            hair_color = None
            if attr[8] == 1:
                hair_color = 1            
            elif attr[9] == 1:
                hair_color = 2
            else:
                hair_color = 0
            return hair_color
            """
        self.train_dataset = datasets.CelebA(root=DATA_ROOT,
                                    split='train',
                                    target_type='attr',
                                    target_transform=get_hair,
                                    transform = self.transform,
                                    download=False)

        self.val_dataset = datasets.CelebA(root=DATA_ROOT,
                                    split='valid',
                                    target_type='attr',
                                    target_transform=get_hair,
                                    transform = self.transform,
                                    download=False)

        self.test_dataset = datasets.CelebA(root=DATA_ROOT,
                                   split='test',
                                   target_type='attr',
                                   target_transform=get_hair, 
                                   transform = self.transform,
                                   download=False)
        
        print("Setup complete.")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            #num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            #num_workers=self.num_workers,
            pin_memory=True,
        )

if __name__ == '__main__':

    # get dataset and check 
    train_loader, valid_loader, test_loader = get_dataloaders_celeba(batch_size=256,
                                                                    train_transforms=custom_transforms,
                                                                    test_transforms=custom_transforms, 
                                                                    download=True,
                                                                    num_workers=4)
    
    import matplotlib.pyplot as plt
    import numpy as np
    import torchvision

    for images, labels in train_loader:
        break

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images, and Label")
    # add caption of label
    plt.imshow(np.transpose(torchvision.utils.make_grid(
        images[:64], 
        padding=2,
        normalize=True),
        (1, 2, 0)))
    plt.show()
    plt.savefig('celeba_train.png')


    # make a subplot of 10 images with labels
    fig, ax = plt.subplots(2, 5, figsize=(20, 10))
    for i in range(10):
        ax[i//5, i%5].imshow(np.transpose(images[i], (1, 2, 0)))
        ax[i//5, i%5].set_title(f"Label: {labels[i]}")
        ax[i//5, i%5].axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('celeba_train_2.png')

    print("labels: ", labels)

    # get num of 1 labels in train, valid, test
    train_smile = 0
    valid_smile = 0
    test_smile = 0

    num_train = 0
    num_valid = 0
    num_test = 0

    
    for images, labels in train_loader:
        train_smile += labels.sum().item()
        num_train += len(labels)

    print(f"Train smile ratio: {train_smile/num_train}, num_smile {train_smile}/num_total {num_train}")
    

    for images, labels in valid_loader:
        valid_smile += labels.sum().item()
        num_valid += len(labels)

    print(f"Valid smile ratio: {valid_smile/num_valid}, num_smile {valid_smile}/num_total {num_valid}")
    
    for images, labels in test_loader:
        test_smile += labels.sum().item()
        num_test += len(labels)

    # label = 1 means smiling, 0 means not smiling

   
    
    print(f"Test smile ratio: {test_smile/num_test}, num_smile {test_smile}/num_total {num_test}")

    