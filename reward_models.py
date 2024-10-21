import torch 
import ImageReward as RM 
from cifar10_models.vgg import vgg13_bn

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

class AestheticPredictor():
    def __init__(self, device):
        self.device = device
        self.aesthetic_model = RM.load("AestheticPredictor-v1.0").to("cuda")

    def __call__(self, img, *args):
        with torch.no_grad():
            logr = torch.tensor(self.aesthetic_model.score(img)).to(self.device)

        return logr

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