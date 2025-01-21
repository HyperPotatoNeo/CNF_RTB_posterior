import types
import torch
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np
import ImageReward as RM 
from cifar10_models.vgg import vgg13_bn
from sngan_cifar10.sngan_cifar10 import Discriminator, SNGANConfig

#from aesthetic_reward.mlp_model import MLP
import clip 

def score_differentiable(self, prompt, img):
    # text encode
    text_input = self.blip.tokenizer(prompt, padding='max_length', truncation=True, max_length=35, return_tensors="pt").to(self.device)
        
    image = F.interpolate(
        img, size=224, mode='bilinear', align_corners=False)
    means = [0.48145466, 0.4578275, 0.40821073]
    stds  = [0.26862954, 0.26130258, 0.27577711]

    mean_t = torch.tensor(means).view(1, -1, 1, 1).cuda()  # shape: (1, 3, 1, 1)
    std_t  = torch.tensor(stds).view(1, -1, 1, 1).cuda()   # shape: (1, 3, 1, 1)

    # Normalize: (x - mean) / std
    image = (image - mean_t) / std_t
    image_embeds = self.blip.visual_encoder(image)
    
    # text encode cross attention with image
    image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(self.device)
    text_output = self.blip.text_encoder(text_input.input_ids,
                                            attention_mask = text_input.attention_mask,
                                            encoder_hidden_states = image_embeds,
                                            encoder_attention_mask = image_atts,
                                            return_dict = True,
                                        )
    
    txt_features = text_output.last_hidden_state[:,0,:].float() # (feature_dim)
    rewards = self.mlp(txt_features)
    rewards = (rewards - self.mean) / self.std
    
    return rewards.squeeze()#.detach().cpu().numpy().item()

class ImageRewardPrompt():
    def __init__(self, device, prompt, differentiable=False):
        self.device = device
        self.prompt = prompt 
        self.reward_model = RM.load("ImageReward-v1.0").to("cuda")
        self.differentiable = differentiable
        if differentiable:
            self.reward_model.score_differentiable = types.MethodType(score_differentiable, self.reward_model)

    def __call__(self, img, *args):
        if not self.differentiable:
            with torch.no_grad():
                prompt = self.prompt 
                logr = torch.tensor(self.reward_model.score(prompt, img)).to(self.device)
        else:
            prompt = self.prompt
            logr = self.reward_model.score_differentiable(prompt, img)
        return logr 


class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

# adapted from https://github.com/christophschuhmann/improved-aesthetic-predictor
class AestheticPredictor():
    def __init__(self, device):
        self.device = device

        self.model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14
        s = torch.load("./aesthetic_reward/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
        self.model.load_state_dict(s)

        self.model.to(self.device)
        self.model.eval()
        
        self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device=device)  #RN50x64   

    def normalized(self, a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2 == 0] = 1
        return a / np.expand_dims(l2, axis)

    def __call__(self, img, *args):
        with torch.no_grad():
            #img = self.clip_preprocess(img)
            preprocessed = [self.clip_preprocess(image) for image in img]
            img = torch.stack(preprocessed).to(self.device)
            im_feat = self.clip_model.encode_image(img)
            im_emb_arr = self.normalized(im_feat.cpu().detach().numpy())

            prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        return prediction[:,0]


class CIFARClassifier():
    def __init__(self, device, target_class):
        self.device = device 
        self.classifer = vgg13_bn(pretrained=True).to(device)
        self.classifier_mean = [0.4914, 0.4822, 0.4465]
        self.classifier_std = [0.2471, 0.2435, 0.2616]

        self.target_class = target_class

    def __call__(self, img, *args):
        logits = self.classifer((img.float() / 255 - torch.tensor(self.classifier_mean).cuda()[None, :, None, None]) / torch.tensor(self.classifier_std).cuda()[None, :, None, None])
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        log_r = log_prob[:, self.target_class].to(self.device)
        return log_r 
    
    def get_class_logits(self, img, *args):
        logits = self.classifer((img.float() / 255 - torch.tensor(self.classifier_mean).cuda()[None, :, None, None]) / torch.tensor(self.classifier_std).cuda()[None, :, None, None])
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        return log_prob
    

class SNGANDiscriminatorReward():
    def __init__(self, device):
        self.device = device
        checkpoint = torch.load("./sngan_cifar10/checkpoint.pth")
        args = SNGANConfig()
        self.discriminator = Discriminator(args).to(device)
        self.discriminator.load_state_dict(checkpoint['dis_state_dict'])
        
    def __call__(self, img, *args):
        img_normalized = img.float()/127.5 - 1
        discriminator_score = self.discriminator(img_normalized)
        return discriminator_score[:,0]

# A trainable reward model, takes as input (in_shape), and 
# outputs scalar reward
class TrainableReward(torch.nn.Module):
    def __init__(self, in_shape, device):
        super(TrainableReward, self).__init__()
        
        self.in_shape = in_shape 
        self.device = device

        # in_shape is (C, H, W)
        # conv net from (in_shape) to scalar
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_shape[0], 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * ((in_shape[1] - 4)// 2 - 2)//2 * ((in_shape[2] - 4 )// 2 - 2)//2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        ).to(device)

    def forward(self, x):
        return self.net(x)
    

    
class TrainableClassifierReward(torch.nn.Module):
    def __init__(self, in_shape, device, num_classes = 10):
        super(TrainableClassifierReward, self).__init__()
        
        self.in_shape = in_shape 
        self.device = device
        self.num_classes = num_classes

        # in_shape is (C, H, W)
        # conv net from (in_shape) to scalar
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_shape[0], 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * ((in_shape[1] - 4)// 2 - 2)//2 * ((in_shape[2] - 4 )// 2 - 2)//2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        ).to(device)

    def forward(self, x):
        return self.net(x)