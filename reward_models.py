import torch 
import numpy as np
import ImageReward as RM 
from cifar10_models.vgg import vgg13_bn

from aesthetic_reward.mlp_model import MLP
import clip 

class ImageRewardPrompt():
    def __init__(self, device, prompt):
        self.device = device
        self.prompt = prompt 
        self.reward_model = RM.load("ImageReward-v1.0").to("cuda")

    def __call__(self, img, *args):

        with torch.no_grad():
            prompt = self.prompt 
            logr = torch.tensor(self.reward_model.score(prompt, img)).to(self.device)

        return logr 


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
            img = self.clip_preprocess(img)
            im_feat = self.clip_model.encode_image(img)
            im_emb_arr = self.normalized(im_feat.cpu().detach().numpy() )

            prediction = self.model(torch.from_numpy(im_emb_arr).to(self.device).type(torch.cuda.FloatTensor))
        return prediction 


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