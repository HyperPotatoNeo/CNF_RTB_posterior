import torch.nn as nn
from .gen_resblock import GenBlock


class Generator(nn.Module):
    def __init__(self, args, activation=nn.ReLU(), n_classes=0):
        super(Generator, self).__init__()
        self.bottom_width = args.bottom_width
        self.activation = activation
        self.n_classes = n_classes
        self.ch = args.gf_dim
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * self.ch)
        self.block2 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block3 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.block4 = GenBlock(self.ch, self.ch, activation=activation, upsample=True, n_classes=n_classes)
        self.b5 = nn.BatchNorm2d(self.ch)
        self.c5 = nn.Conv2d(self.ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z):

        h = z
        h = self.l1(h).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.b5(h)
        h = self.activation(h)
        h = nn.Tanh()(self.c5(h))
        return h


"""Discriminator"""


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, ksize=3, pad=1, activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_channels=None, ksize=3, pad=1,
                 activation=nn.ReLU(), downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=ksize, padding=pad)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=True)
        self.block3 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=False)
        self.block4 = DisBlock(args, self.ch, self.ch, activation=activation, downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output
    
class SNGANConfig:
    def __init__(self,
                 gen_bs=128,
                 dis_bs=64,
                 dataset="cifar10",
                 img_size=32,
                 max_iter=50000,
                 model="sngan_cifar10",
                 latent_dim=128,
                 gf_dim=256,
                 df_dim=128,
                 g_spectral_norm=False,
                 d_spectral_norm=True,
                 g_lr=0.0002,
                 d_lr=0.0002,
                 beta1=0.0,
                 beta2=0.9,
                 init_type="xavier_uniform",
                 n_critic=5,
                 val_freq=20,
                 bottom_width=4,
                 exp_name="sngan_cifar10"):
        self.gen_bs = gen_bs
        self.dis_bs = dis_bs
        self.dataset = dataset
        self.img_size = img_size
        self.max_iter = max_iter
        self.model = model
        self.latent_dim = latent_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.g_spectral_norm = g_spectral_norm
        self.d_spectral_norm = d_spectral_norm
        self.g_lr = g_lr
        self.d_lr = d_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.init_type = init_type
        self.n_critic = n_critic
        self.val_freq = val_freq
        self.bottom_width = bottom_width
        self.exp_name = exp_name