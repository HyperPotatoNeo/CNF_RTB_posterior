import os
from torchvision import datasets, transforms
from rtb_model import RTBModel
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()

parser.add_argument('--n_iters', default=50000, type=int, metavar='N', help='Number of training iterations')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Training Batch Size.")
parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float, help='Initial learning rate.')
parser.add_argument('--posterior_class', type=int, default=0, help="Posterior finetuning class")
parser.add_argument('--diffusion_steps', type=int, default=200)
parser.add_argument('--wandb_track', default=False, type=strtobool, help='Whether to track with wandb.')
parser.add_argument('--compute_fid', default=False, type=strtobool, help="Whether to compute FID score during training (every 1k steps)")
parser.add_argument('--fit_gaussian', default=False, type=strtobool, help='Whether to fit a gaussian to posterior samples and compute FID')
parser.add_argument('--prior_sample', default=False, type=strtobool, help="Whether to use off policy samples from prior")
parser.add_argument('--iw_logz', default=False, type=strtobool, help="Whether to use IW log Z estimator")
parser.add_argument('--temperature', default=1.0, type=float, help='(Initial) Temperature for classifier softmax')
parser.add_argument('--anneal', default=False, type=strtobool, help='Whether to anneal temperature (till 0.2)')
parser.add_argument('--anneal_steps', default=15000, type=int, help="Number of steps for temperature annealing")

args = parser.parse_args()

if args.compute_fid:
    # Define the transformation (convert to tensor)
    transform = transforms.Compose([transforms.ToTensor()])
    # Download the CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # Define the class for which we want to save images (class 0 in CIFAR-10 corresponds to 'airplane')
    class_label = args.posterior_class
    # Create a directory to save the images
    output_dir = 'cifar10_class_' + str(class_label)
    os.makedirs(output_dir, exist_ok=True)
    generated_images_dir = 'generated_cifar10_class_' + str(class_label)
    os.makedirs(generated_images_dir, exist_ok=True)

    # Filter and save images of class 0
    for i, (img, label) in enumerate(train_dataset):
        if label == class_label:
            img = transforms.ToPILImage()(img)
            img.save(os.path.join(output_dir, f'airplane_{i}.png'))

rtb_model = RTBModel(posterior_class=args.posterior_class, diffusion_steps=args.diffusion_steps, temperature=args.temperature)

rtb_model.finetune(shape=(args.batch_size, 3, 32, 32), n_iters = args.n_iters, wandb_track=args.wandb_track, learning_rate=args.lr, prior_sample=args.prior_sample, iw_logz=args.iw_logz, compute_fid=args.compute_fid, anneal=args.anneal, anneal_steps=args.anneal_steps)