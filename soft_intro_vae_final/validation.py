# standard
import collections
import os
import pickle
import random
import time
from pathlib import Path
from typing import Tuple, Any, Optional, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import uncertify
from IPython import embed
from classification import dice, precision, recall
from dataset import ImageDatasetFromFile, DigitalMonstersDataset
from metrics.fid_score import calculate_fid_given_dataset
from sklearn.metrics import roc_auc_score
from torch.distributions import Normal, Independent, MultivariateNormal
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torchvision.utils import make_grid
from tqdm import tqdm
from uda_release.dset_classes.DsetNoLabel import DsetNoLabel
from uda_release.dset_classes.DsetSSRotRand import DsetSSRotRand
from uncertify.scripts import add_uncertify_to_path
from uncertify.uncertify.data.artificial import BrainGaussBlobDataset
from uncertify.uncertify.data.datasets import CamCanHDF5Dataset, Brats2017HDF5Dataset
from uncertify.uncertify.data.np_transforms import NumpyReshapeTransform, Numpy2PILTransform
from uncertify.uncertify.models.encoder_decoder_baur2020 import BaurEncoder, BaurDecoder
from uncertify.uncertify.utils.custom_types import Tensor
from uncertify.uncertify.visualization.grid import imshow_grid

# writer = SummaryWriter()

"""
Models
"""


class BasicModule(torch.nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path=None):
        if path is None:
            name = 'result/best_model.pth'
            torch.save(self.state_dict(), name)
            return name
        else:
            torch.save(self.state_dict(), path)
            return path


class ResidualBlock(nn.Module):
    """
    https://github.com/hhb072/IntroVAE
    Difference: self.bn2 on output and not on (output + identity)
    """

    def __init__(self, inc=64, outc=64, groups=1, scale=1.0):
        super(ResidualBlock, self).__init__()

        midc = int(outc * scale)

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=midc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(midc)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=midc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(outc)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(torch.add(output, identity_data))
        return output


class Encoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 cond_dim=10):
        super(Encoder, self).__init__()
        self.zdim = zdim
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        self.cond_dim = cond_dim
        cc = channels[0]
        self.main = nn.Sequential(
            nn.Conv2d(cdim, cc, 5, 1, 2, bias=False),
            nn.BatchNorm2d(cc),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
        )

        sz = image_size // 2
        for ch in channels[1:]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('down_to_{}'.format(sz // 2), nn.AvgPool2d(2))
            cc, sz = ch, sz // 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.conv_output_size = self.calc_conv_output_size()
        num_fc_features = torch.zeros(self.conv_output_size).view(-1).shape[0]
        print("conv shape: ", self.conv_output_size)
        print("num fc features: ", num_fc_features)
        if self.conditional:
            self.fc = nn.Linear(num_fc_features + self.cond_dim, 2 * zdim)
        else:
            self.fc = nn.Linear(num_fc_features, 2 * zdim)

    def calc_conv_output_size(self):
        dummy_input = torch.zeros(1, self.cdim, self.image_size, self.image_size)
        dummy_input = self.main(dummy_input)
        return dummy_input[0].shape

    def forward(self, x, o_cond=None):
        y = self.main(x).view(x.size(0), -1)
        if self.conditional and o_cond is not None:
            y = torch.cat([y, o_cond], dim=1)
        y = self.fc(y)
        mu, logvar = y.chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 conv_input_size=None, cond_dim=10):
        super(Decoder, self).__init__()
        self.cdim = cdim
        self.image_size = image_size
        self.conditional = conditional
        cc = channels[-1]
        self.conv_input_size = conv_input_size
        if conv_input_size is None:
            num_fc_features = cc * 4 * 4
        else:
            num_fc_features = torch.zeros(self.conv_input_size).view(-1).shape[0]
        self.cond_dim = cond_dim
        if self.conditional:
            self.fc = nn.Sequential(
                nn.Linear(zdim + self.cond_dim, num_fc_features),
                nn.ReLU(True),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(zdim, num_fc_features),
                nn.ReLU(True),
            )

        sz = 4

        self.main = nn.Sequential()
        for ch in channels[::-1]:
            self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, ch, scale=1.0))
            self.main.add_module('up_to_{}'.format(sz * 2), nn.Upsample(scale_factor=2, mode='nearest'))
            cc, sz = ch, sz * 2

        self.main.add_module('res_in_{}'.format(sz), ResidualBlock(cc, cc, scale=1.0))
        self.main.add_module('predict', nn.Conv2d(cc, cdim, 5, 1, 2))

    def forward(self, z, y_cond=None):
        z = z.view(z.size(0), -1)
        if self.conditional and y_cond is not None:
            y_cond = y_cond.view(y_cond.size(0), -1)
            z = torch.cat([z, y_cond], dim=1)
        y = self.fc(z)
        y = y.view(z.size(0), *self.conv_input_size)
        y = self.main(y)
        return y


class SoftIntroVAE(nn.Module):
    def __init__(self, cdim=3, zdim=512, channels=(64, 128, 256, 512, 512, 512), image_size=256, conditional=False,
                 cond_dim=10):
        super(SoftIntroVAE, self).__init__()

        self.zdim = zdim
        self.conditional = conditional
        self.cond_dim = cond_dim

        self.encoder = Encoder(cdim, zdim, channels, image_size, conditional=conditional, cond_dim=cond_dim)

        self.decoder = Decoder(cdim, zdim, channels, image_size, conditional=conditional,
                               conv_input_size=self.encoder.conv_output_size, cond_dim=cond_dim)

    def forward(self, x, o_cond=None, deterministic=False):
        if self.conditional and o_cond is not None:
            mu, logvar = self.encode(x, o_cond=o_cond)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z, y_cond=o_cond)
        else:
            mu, logvar = self.encode(x)
            if deterministic:
                z = mu
            else:
                z = reparameterize(mu, logvar)
            y = self.decode(z)
        return mu, logvar, z, y

    def sample(self, z, y_cond=None):
        y = self.decode(z, y_cond=y_cond)
        return y

    def sample_with_noise(self, num_samples=1, device=torch.device("cpu"), y_cond=None):
        z = torch.randn(num_samples, self.z_dim).to(device)
        return self.decode(z, y_cond=y_cond)

    def encode(self, x, o_cond=None):
        if self.conditional and o_cond is not None:
            mu, logvar = self.encoder(x, o_cond=o_cond)
        else:
            mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z, y_cond=None):
        if self.conditional and y_cond is not None:
            y = self.decoder(z, y_cond=y_cond)
        else:
            y = self.decoder(z)
        return y


class BaurModel(nn.Module):
    def __init__(self):
        super(BaurModel, self).__init__()
        self.encoder = BaurEncoder()
        self.decoder = BaurDecoder()

    def encode(self, x):
        mu, logvar = self.encoder(x)
        return mu, logvar

    def decode(self, z):
        y = self.decoder(z)
        return y

    def sample(self, z):
        y = self.decode(z)
        return y

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        y = self.decode(z)

        return mu, logvar, z, y


class DCD(BasicModule):
    def __init__(self, h_features=64, input_features=128):
        super(DCD, self).__init__()

        self.fc1 = nn.Linear(input_features, h_features)
        self.fc2 = nn.Linear(h_features, h_features)
        self.fc3 = nn.Linear(h_features, 4)

    def forward(self, inputs):
        out = F.relu(self.fc1(inputs))
        out = self.fc2(out)
        return F.softmax(self.fc3(out), dim=1)


"""
Helpers
"""


def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl


def reparameterize(mu, logvar):
    """
    This function applies the reparameterization trick:
    z = mu(X) + sigma(X)^0.5 * epsilon, where epsilon ~ N(0,I)
    :param mu: mean of x
    :param logvar: log variaance of x
    :return z: the sampled latent variable
    """
    device = mu.device
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std).to(device)
    return mu + eps * std


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='mean'):
    """

    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        # if reduction == 'sum':
        #     recon_error = recon_error.sum()
        # elif reduction == 'mean':
        #     recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction).sum(1)  # for rocau calculation only
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error


def str_to_list(x):
    return [int(xi) for xi in x.split(',')]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg", ".bmp"])


def record_scalar(writer, scalar_list, scalar_name_list, cur_iter):
    scalar_name_list = scalar_name_list[1:-1].split(',')
    for idx, item in enumerate(scalar_list):
        writer.add_scalar(scalar_name_list[idx].strip(' '), item, cur_iter)


def record_image(writer, image_list, cur_iter, num_rows=8):
    image_to_show = torch.cat(image_list, dim=0)
    writer.add_image('visualization', make_grid(image_to_show, nrow=num_rows), cur_iter)


def load_model(model, pretrained, device):
    weights = torch.load(pretrained, map_location=device)
    model.load_state_dict(weights['model'], strict=False)


def save_checkpoint(model, epoch, iteration, prefix=""):
    model_out_path = "./saves/" + prefix + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model.state_dict()}
    if not os.path.exists("./saves/"):
        os.makedirs("./saves/")

    torch.save(state, model_out_path)

    print("model checkpoint saved @ {}".format(model_out_path))


def estimate_log_p_x(model, x, mu, logvar, num_samples=5, x_sigma=1.0):
    """
    Estimates log p(x): log p(x) ~ log sum_i [ exp( log p(x|z_i) + log p(z_i) - log q(z_i|x) )]
    z_i ~ q(z|x)
    """
    batch_size, z_dim = mu.shape
    x = x.view(batch_size, -1)
    sigma = (0.5 * logvar).exp()

    # distributions
    q_z_x = MultivariateNormal(mu, torch.diag_embed(sigma))
    p_z = MultivariateNormal(torch.zeros_like(mu), torch.diag_embed(torch.ones_like(mu)))
    # p_x_z = dists.MultivariateNormal(x, x_sigma * torch.diag_embed(torch.ones_like(x)))

    # get samples
    samples = q_z_x.sample(sample_shape=[num_samples])  # num_samples x batch_size x z_dim
    # reconstruct samples
    # recons = []
    log_p_x_z_s = []
    for i in range(num_samples):
        recon_x = model.decode(samples[i])
        # recons.append(recon_x.view(batch_size, -1))
        log_p_x_z_s.append(gaussian_unit_log_pdf(recon_x.view(batch_size, -1), x, sigma=x_sigma))
    # recons = torch.stack(recons)
    log_p_x_z = torch.stack(log_p_x_z_s)  # num_samples x batch_size x (*img_size)

    # log probs
    log_q_z_x = q_z_x.log_prob(samples)
    log_p_z = p_z.log_prob(samples)
    # log_p_x_z = p_x_z.log_prob(recons)

    # log p(x)
    log_p = torch.logsumexp(log_p_x_z + log_p_z - log_q_z_x, dim=0)
    return log_p


def gaussian_unit_log_pdf(x, mu, sigma=1.0):
    """
    mu: batch_size X x_dim
    sigma: scalar
    """
    batch_size, x_dim = mu.shape
    x = x.view(batch_size, -1)
    return -0.5 * x_dim * np.log(2 * np.pi) - (0.5 / sigma) * ((x - mu) ** 2).sum(-1)


def calc_score_and_auc(model, ds='cifar10', ood_ds='svhn', batch_size=64, rand_state=None, beta_kl=1.0, beta_rec=1.0,
                       device=torch.device("cpu"), loss_type='mse', with_nll=False, type='all'):
    """
    Test ROC AUC of SoftIntroVAE
    """
    # load datasets
    if ds == "cifar10":
        dataset = CIFAR10(root='./cifar10_ds', train=False, download=True, transform=transforms.ToTensor())
    elif ds == 'camcan':
        BRATS_CAMCAN_DEFAULT_TRANSFORM = transforms.Compose([
            NumpyReshapeTransform((200, 200)),
            Numpy2PILTransform(),
            transforms.Resize((128, 128)),
            transforms.PILToTensor()])

        camcan_t2_train_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
                                    "camcan_train_t2_hm_std_bv3.5_xe.hdf5")
        camcan_t2_val_path = Path('/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/'
                                  'camcan_val_t2_hm_std_bv3.5_xe.hdf5')

        train_dataloader, val_dataloader = camcan_data_loader(hdf5_train_path=camcan_t2_train_path,
                                                              hdf5_val_path=camcan_t2_val_path,
                                                              batch_size=batch_size, shuffle_train=True,
                                                              shuffle_val=False,
                                                              num_workers=4, transform=BRATS_CAMCAN_DEFAULT_TRANSFORM,
                                                              uppercase_keys=False, add_gauss_blobs=False)
    elif ds == 'svhn':
        dataset = SVHN(root='./svhn', split='test', transform=transforms.ToTensor(), download=True)
    elif ds == "fmnist":
        dataset = FashionMNIST(root='./fmnist_ds', train=False, download=True, transform=transforms.ToTensor())
    elif ds == "mnist":
        dataset = MNIST(root='./mnist_ds', train=False, download=True, transform=transforms.ToTensor())
    else:
        raise NotImplementedError
    if ood_ds == 'svhn':
        ood_dataset = SVHN(root='./svhn', split='test', transform=transforms.ToTensor(), download=True)
    elif ood_ds == 'camcan':
        BRATS_CAMCAN_DEFAULT_TRANSFORM = transforms.Compose([
            NumpyReshapeTransform((200, 200)),
            Numpy2PILTransform(),
            transforms.Resize((128, 128)),
            transforms.PILToTensor()])

        camcan_t2_train_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
                                    "camcan_train_t2_hm_std_bv0.0_xe.hdf5")
        camcan_t2_val_path = Path('/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/'
                                  'camcan_val_t2_hm_std_bv3.5_xe.hdf5')

        _, ood_dataloader = camcan_data_loader(hdf5_train_path=camcan_t2_train_path, hdf5_val_path=camcan_t2_val_path,
                                               batch_size=batch_size, shuffle_train=True, shuffle_val=False,
                                               num_workers=4, transform=BRATS_CAMCAN_DEFAULT_TRANSFORM,
                                               uppercase_keys=False, add_gauss_blobs=True)
    elif ood_ds == 'brats':
        BRATS_CAMCAN_DEFAULT_TRANSFORM = transforms.Compose([
            NumpyReshapeTransform((200, 200)),
            Numpy2PILTransform(),
            transforms.Resize((128, 128)),
            transforms.PILToTensor()])

        # ood_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/processed/"
        #                     "brats17_t2_bc_std_bv3.5.hdf5")
        ood_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
                            "brats17_t2_hm_bc_std_bv3.5_xe.hdf5")
        # ood_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
        #                     "brats17_t1_bc_std_bv3.5_xe.hdf5")
        # ood_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
        #                     "brats17_t1_hm_bc_std_bv3.5_xe.hdf5")

        ood_dataloader = brats17_val_dataloader(hdf5_path=ood_set_path, batch_size=batch_size, shuffle=False,
                                                num_workers=4,
                                                transform=BRATS_CAMCAN_DEFAULT_TRANSFORM)
    elif ood_ds == 'cifar10':
        ood_dataset = CIFAR10(root='./cifar10_ds', train=False, download=True, transform=transforms.ToTensor())
    elif ood_ds == 'mnist':
        mnist_tranform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        ood_dataset = MNIST(root='./mnist_ds', train=False, download=True, transform=mnist_tranform)
    elif ood_ds == 'fmnist':
        ood_dataset = FashionMNIST(root='./fmnist_ds', train=False, download=True, transform=transforms.ToTensor())
    else:
        raise NotImplementedError

    ds_dl = val_dataloader  # DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ood_dl = ood_dataloader  # DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

    # loop
    idx_label_score = []
    idx_label_score_kl = []
    idx_label_score_recon = []
    idx_label_score_nll = []
    idx_label_score_double_kl = []

    # normal data
    print("calculating for normal data...")
    for batch_i, batch in enumerate(ds_dl):
        # inputs, _ = batch
        mask = batch["mask"]  # first so you get the mask before changing the batch to the actual one
        real_batch = batch["scan"]
        # embed()
        # if type == 'all':
        inputs = real_batch.to(device)
        mask = mask.to(device)
        # elif type == 'healthy':
        #     seg = batch['seg']
        #     healthy = real_batch[torch.sum(seg, dim=(1, 2, 3)) == 0]
        #     inputs = healthy.to(device)
        #     mask = mask[torch.sum(seg, dim=(1, 2, 3)) == 0].to(device)
        # elif type == 'lesions':
        #     seg = batch['seg']
        #     lesions = real_batch[torch.sum(seg, dim=(1, 2, 3)) != 0]
        #     inputs = lesions.to(device)
        #     mask = mask[torch.sum(seg, dim=(1, 2, 3)) != 0].to(device)
        # inputs = batch[0]
        labels = torch.zeros(inputs.size(0))

        with torch.no_grad():
            mu, logvar, _, rec = model(inputs)
            rec_mu, rec_logvar, _, rec_rec = model(rec)
            # loss_rec = calc_reconstruction_loss(inputs*mask, rec*mask, loss_type=loss_type, reduction="none")
            _, _, _, _, loss_rec = loss_function(rec, inputs, mask, mu, logvar)
            loss_kl = calc_kl(logvar, mu, reduce="none")
            elbo = loss_rec + loss_kl
            # embed()
            kl_real_to_rec = calc_kl(logvar, mu, mu_o=rec_mu, logvar_o=rec_logvar.clamp(-20), reduce="none")
            kl_rec_to_real = calc_kl(rec_logvar, rec_mu, mu_o=mu, logvar_o=logvar.clamp(-20), reduce="none")
            loss_double_kl = kl_real_to_rec + kl_rec_to_real
            if with_nll:
                loss_nll = -1 * estimate_log_p_x(model, inputs, mu=mu, logvar=logvar, num_samples=5, x_sigma=1.0)
                idx_label_score_nll += list(zip((labels.data.cpu().numpy().astype(np.bool)).astype(np.int).tolist(),
                                                loss_nll.cpu().data.numpy().tolist()))
        idx_label_score += list(zip((labels.data.cpu().numpy().astype(np.bool)).astype(np.int).tolist(),
                                    elbo.cpu().data.numpy().tolist()))
        idx_label_score_kl += list(zip((labels.data.cpu().numpy().astype(np.bool)).astype(np.int).tolist(),
                                       loss_kl.cpu().data.numpy().tolist()))
        idx_label_score_recon += list(zip((labels.data.cpu().numpy().astype(np.bool)).astype(np.int).tolist(),
                                          loss_rec.cpu().data.numpy().tolist()))

        idx_label_score_double_kl += list(zip((labels.data.cpu().numpy().astype(np.bool)).astype(np.int).tolist(),
                                              loss_double_kl.cpu().data.numpy().tolist()))

    # ood_data
    print("calculating for ood data...")
    ood_idx_label_score = []
    ood_idx_label_score_kl = []
    ood_idx_label_score_recon = []
    ood_idx_label_score_nll = []
    ood_idx_label_score_double_kl = []

    for batch_i, batch in enumerate(ood_dl):
        # inputs, _ = batch
        # inputs = batch[0]
        mask = batch["mask"]  # first so you get the mask before changing the batch to the actual one
        real_batch = batch["scan"]
        seg = batch['seg']
        # embed()
        if type == 'all':
            inputs = real_batch.to(device)
            mask = mask.to(device)
            labels = torch.ones(inputs.size(0))
            # labels[torch.sum(seg, dim=(1, 2, 3)) == 0] = 0
        elif type == 'healthy':
            # seg = batch['seg']
            healthy = real_batch[torch.sum(seg, dim=(1, 2, 3)) == 0]
            inputs = healthy.to(device)
            mask = mask[torch.sum(seg, dim=(1, 2, 3)) == 0].to(device)
            labels = torch.ones(inputs.size(0))
        elif type == 'lesions':
            # seg = batch['seg']
            lesions = real_batch[torch.sum(seg, dim=(1, 2, 3)) != 0]
            inputs = lesions.to(device)
            mask = mask[torch.sum(seg, dim=(1, 2, 3)) != 0].to(device)
            labels = torch.ones(inputs.size(0))

        with torch.no_grad():
            mu, logvar, _, rec = model(inputs)
            rec_mu, rec_logvar, _, rec_rec = model(rec)
            # loss_rec = calc_reconstruction_loss(inputs, rec, loss_type=loss_type, reduction="none")
            _, _, _, _, loss_rec = loss_function(rec, inputs, mask, mu, logvar)
            loss_kl = calc_kl(logvar, mu, reduce="none")
            elbo = loss_rec + loss_kl
            kl_real_to_rec = calc_kl(logvar, mu, mu_o=rec_mu, logvar_o=rec_logvar.clamp(-20), reduce="none")
            kl_rec_to_real = calc_kl(rec_logvar, rec_mu, mu_o=mu, logvar_o=logvar.clamp(-20), reduce="none")
            loss_double_kl = kl_real_to_rec + kl_rec_to_real
            if with_nll:
                loss_nll = -1 * estimate_log_p_x(model, inputs, mu=mu, logvar=logvar, num_samples=5, x_sigma=1.0)
                ood_idx_label_score_nll += list(zip((labels.data.cpu().numpy().astype(np.bool)).astype(np.int).tolist(),
                                                    loss_nll.cpu().data.numpy().tolist()))

        ood_idx_label_score += list(zip((labels.data.cpu().numpy().astype(np.bool)).astype(np.int).tolist(),
                                        elbo.cpu().data.numpy().tolist()))
        ood_idx_label_score_kl += list(zip((labels.data.cpu().numpy().astype(np.bool)).astype(np.int).tolist(),
                                           loss_kl.cpu().data.numpy().tolist()))
        ood_idx_label_score_recon += list(zip((labels.data.cpu().numpy().astype(np.bool)).astype(np.int).tolist(),
                                              loss_rec.cpu().data.numpy().tolist()))
        ood_idx_label_score_double_kl += list(zip((labels.data.cpu().numpy().astype(np.bool)).astype(np.int).tolist(),
                                                  loss_double_kl.cpu().data.numpy().tolist()))

    results_dict = {}
    # calculate ROC AUC score - kl
    # labels, scores = zip(*idx_label_score_kl)
    # labels = np.array(labels)
    # scores = np.array(scores)
    # avg_normal_score = np.mean(scores)
    # print("Stop to check kl for in-data")
    # np.savetxt(f"./in_data_kl.txt", scores)
    # #
    # ood_labels, ood_scores = zip(*ood_idx_label_score_kl)
    # ood_labels = np.array(ood_labels)
    # ood_scores = np.array(ood_scores)
    # avg_ood_score = np.mean(ood_scores)
    # print("Stop to check kl for oodata")
    # np.savetxt(f"./out_data_kl.txt", scores)
    #
    # total_labels = np.concatenate([labels, ood_labels], axis=0)
    # total_scores = np.concatenate([scores, ood_scores], axis=0)
    # test_auc = roc_auc_score(total_labels, total_scores)
    # kl_dict = {'normal': avg_normal_score, 'ood': avg_ood_score, 'auc': test_auc}
    # results_dict['kl'] = kl_dict

    # calculate ROC AUC score - recon
    # labels, scores = zip(*idx_label_score_recon)
    # labels = np.array(labels)
    # scores = np.array(scores)
    # avg_normal_score = np.mean(scores)
    # print("Stop to check recon for in-data")
    # np.savetxt(f"./in_data_recon.txt", scores)
    # #
    # ood_labels, ood_scores = zip(*ood_idx_label_score_recon)
    # ood_labels = np.array(ood_labels)
    # ood_scores = np.array(ood_scores)
    # avg_ood_score = np.mean(ood_scores)
    # print("Stop to check recon for oodata")
    # np.savetxt(f"./out_data_recon.txt", scores)
    #
    # total_labels = np.concatenate([labels, ood_labels], axis=0)
    # total_scores = np.concatenate([scores, ood_scores], axis=0)
    # test_auc = roc_auc_score(total_labels, total_scores)
    # recon_dict = {'normal': avg_normal_score, 'ood': avg_ood_score, 'auc': test_auc}
    # results_dict['recon'] = recon_dict

    if with_nll:
        # calculate ROC AUC score - nll
        labels, scores = zip(*idx_label_score_nll)
        labels = np.array(labels)
        scores = np.array(scores)
        avg_normal_score = np.mean(scores)

        ood_labels, ood_scores = zip(*ood_idx_label_score_nll)
        ood_labels = np.array(ood_labels)
        ood_scores = np.array(ood_scores)
        avg_ood_score = np.mean(ood_scores)

        total_labels = np.concatenate([labels, ood_labels], axis=0)
        total_scores = np.concatenate([scores, ood_scores], axis=0)
        test_auc = roc_auc_score(total_labels, total_scores)
        nll_dict = {'normal': avg_normal_score, 'ood': avg_ood_score, 'auc': test_auc}
        results_dict['nll'] = nll_dict

    # calculate ROC AUC score - double_kl
    # labels, scores = zip(*idx_label_score_double_kl)
    # labels = np.array(labels)
    # scores = np.array(scores)
    # avg_normal_score = np.mean(scores)
    #
    # ood_labels, ood_scores = zip(*ood_idx_label_score_double_kl)
    # ood_labels = np.array(ood_labels)
    # ood_scores = np.array(ood_scores)
    # avg_ood_score = np.mean(ood_scores)
    #
    # total_labels = np.concatenate([labels, ood_labels], axis=0)
    # total_scores = np.concatenate([scores, ood_scores], axis=0)
    # test_auc = roc_auc_score(total_labels, total_scores)
    # dkl_dict = {'normal': avg_normal_score, 'ood': avg_ood_score, 'auc': test_auc}
    # results_dict['dkl'] = dkl_dict

    # calculate ROC AUC score - elbo
    labels, scores = zip(*idx_label_score)
    labels = np.array(labels)
    scores = np.array(scores)
    avg_normal_score = np.mean(scores)

    ood_labels, ood_scores = zip(*ood_idx_label_score)
    ood_labels = np.array(ood_labels)
    ood_scores = np.array(ood_scores)
    avg_ood_score = np.mean(ood_scores)

    total_labels = np.concatenate([labels, ood_labels], axis=0)
    total_scores = np.concatenate([scores, ood_scores], axis=0)

    test_auc = roc_auc_score(total_labels, total_scores)
    elbo_dict = {'normal': avg_normal_score, 'ood': avg_ood_score, 'auc': test_auc}
    results_dict['elbo'] = elbo_dict

    return results_dict


def print_results(r_dict):
    print('#' * 50)
    print("OOD Results:")
    for s_key in r_dict.keys():
        print(
            f'Metric: {s_key} -- Avg. Normal Score: {r_dict[s_key]["normal"]}, Avg. OOD Score: {r_dict[s_key]["ood"]}, AUROC: {r_dict[s_key]["auc"]}')
    print('#' * 50)


def loss_function(reconstruction: Tensor, observation: Tensor, mask: Tensor, mu: Tensor, log_var: Tensor,
                  beta: float = 1.0, train_step: int = None):
    masking_enabled = mask is not None
    mask = mask if masking_enabled else torch.ones_like(observation, dtype=torch.bool)

    # Reconstruction Error
    slice_rec_err = F.l1_loss(observation * mask, reconstruction * mask, reduction='none')
    slice_rec_err = torch.sum(slice_rec_err, dim=(1, 2, 3))
    slice_wise_non_empty = torch.sum(mask, dim=(1, 2, 3))
    # embed()
    slice_wise_non_empty[slice_wise_non_empty == 0] = 1  # for empty masks
    slice_rec_err /= slice_wise_non_empty
    batch_rec_err = torch.mean(slice_rec_err)

    # KL Divergence
    slice_kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    batch_kl_div = torch.mean(slice_kl_div)
    # kld_loss = batch_kl_div * self._calculate_beta(self._train_step_counter)

    # Total Loss
    total_loss = batch_rec_err + batch_kl_div  # changed from initial kld_loss

    return total_loss, batch_kl_div, batch_rec_err, slice_kl_div, slice_rec_err


def camcan_data_loader(hdf5_train_path: Path = None, hdf5_val_path: Path = None,
                       batch_size: int = 64, shuffle_train: bool = True, shuffle_val: bool = False,
                       num_workers: int = 0, transform: Any = None,
                       uppercase_keys: bool = False, add_gauss_blobs: bool = False,
                       domain_adapt: bool = False) -> Tuple[DataLoader, DataLoader]:
    """Create CamCAN train and / or val dataloaders based on paths to hdf5 files."""
    assert not all(path is None for path in {hdf5_train_path, hdf5_val_path}), \
        f'Need to give a train and / or test path!'
    if transform is None:
        transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
    if hdf5_train_path is None:
        train_dataloader = None
    else:
        train_set = CamCanHDF5Dataset(hdf5_file_path=hdf5_train_path, transform=transform,
                                      uppercase_keys=uppercase_keys)
        if domain_adapt:
            train_set = DsetSSRotRand(train_set, digit=False)
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    if hdf5_val_path is None:
        val_dataloader = None
    else:
        val_set = CamCanHDF5Dataset(hdf5_file_path=hdf5_val_path, transform=transform, uppercase_keys=uppercase_keys)
        if add_gauss_blobs:
            val_set = BrainGaussBlobDataset(val_set)
        if domain_adapt:
            val_set = DsetSSRotRand(val_set, digit=False)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers)
    return train_dataloader, val_dataloader


def brats17_val_dataloader(hdf5_path: Path, batch_size: int, shuffle: bool,
                           num_workers: int = 0, transform: Any = None,
                           uppercase_keys: bool = False, add_gauss_blobs: bool = False,
                           domain_adapt: bool = False) -> DataLoader:
    """Create a BraTS dataloader based on a hdf_path."""
    assert hdf5_path.exists(), f'BraTS17 hdf5 file {hdf5_path} does not exist!'
    if transform is None:
        transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
    brats_val_dataset = Brats2017HDF5Dataset(hdf5_file_path=hdf5_path, transform=transform,
                                             uppercase_keys=uppercase_keys)
    if domain_adapt:
        brats_val_dataset = DsetSSRotRand(brats_val_dataset, digit=False)
    if add_gauss_blobs:
        brats_val_dataset = BrainGaussBlobDataset(brats_val_dataset)
    val_dataloader = DataLoader(brats_val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return val_dataloader


def plot_stacked_scan_reconstruction_batches(real_batch, mask, reconstruction,
                                             plot_n_batches: int = 3, nrow: int = 8, show_mask: bool = False,
                                             save_dir_path: Path = None, cut_tensor_to: int = 8,
                                             mask_background: bool = True, close_fig: bool = True, **kwargs) -> None:
    """Plot the scan and reconstruction batches. Horizontally aligned are the samples from one batch.
    Vertically aligned are input image, ground truth segmentation, reconstruction, residual image, residual with
    applied threshold, ground truth and predicted segmentation in same image.
    Args:
        batch_generator: a PyTorch DataLoader as defined in the uncertify dataloaders module
        plot_n_batches: limit plotting to this amount of batches
        save_dir_path: path to directory in which to store the resulting plots - will be created if not existent
        nrow: numbers of samples in one row, default is 8
        show_mask: plots the brain mask
        cut_tensor_to: choose 8 if you need only first 8 samples but for plotting but the original tensor is way larger
        close_fig: if True, will not show in notebook, handy for use in large pipeline
        kwargs: additional keyword arguments for plotting functions
    """
    if save_dir_path is not None:
        save_dir_path.mkdir(exist_ok=True)
    with torch.no_grad():
        # for batch_idx, batch in enumerate(itertools.islice(batch_generator, plot_n_batches)):
        mask = mask
        max_val = torch.max(real_batch)
        min_val = torch.min(real_batch)

        scan = normalize_to_0_1(real_batch, min_val, max_val)
        reconstruction = normalize_to_0_1(reconstruction, min_val, max_val)
        # residual = normalize_to_0_1(batch.residual)
        # thresholded = batch.residuals_thresholded

        if mask_background:
            scan = mask_background_to_zero(scan, mask)
            reconstruction = mask_background_to_zero(reconstruction, mask)
            # residual = mask_background_to_zero(residual, mask)
            # thresholded = mask_background_to_zero(thresholded, mask)

        # if batch.segmentation is not None:
        #     seg = mask_background_to_zero(batch.segmentation, mask)
        #     stacked = torch.cat((scan, seg, reconstruction, residual, thresholded), dim=2)
        stacked = torch.cat((scan, reconstruction), dim=2)
        if show_mask:
            stacked = torch.cat((stacked, mask.type(torch.FloatTensor)), dim=2)
        if cut_tensor_to is not None:
            stacked = stacked[:cut_tensor_to, ...]
        grid = vutils.make_grid(stacked, padding=0, normalize=False, nrow=nrow)
        # describe = scipy.stats.describe(grid.numpy().flatten())
        # print_scipy_stats_description(describe, 'normalized_grid')
        fig, ax = imshow_grid(grid, one_channel=True, vmax=1.0, vmin=0.0, plt_show=True, **kwargs)
        ax.set_axis_off()
        # plt.show()
        # embed()
        if save_dir_path is not None:
            img_file_name = 'batch_1.png'
            plt.savefig(fig, save_dir_path / img_file_name)
            if close_fig:
                plt.close(fig)


def normalize_to_0_1(tensor: Tensor, min_val: Union[float, torch.tensor] = None,
                     max_val: Union[float, torch.tensor] = None) -> Tensor:
    """Takes a pytorch tensor and normalizes the values to the range [0, 1], i.e. largest value gets 1, smallest 0."""
    if min_val is None:
        min_val = tensor.min()
    if max_val is None:
        max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


def mask_background_to_zero(input_tensor: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask, input_tensor, torch.zeros_like(input_tensor))


def mask_background_to_minus_three(input_tensor: Tensor, mask: Tensor) -> Tensor:
    return torch.where(mask, input_tensor, torch.ones_like(input_tensor)*-3.5)


def residual_l1_max(reconstruction: Tensor, original: Tensor) -> Tensor:
    """Construct l1 difference between original and reconstruction.

    Note: Only positive values in the residual are considered, i.e. values below zero are clamped.
    That means only cases where bright pixels which are brighter in the input (likely lesions) are kept."""
    residual = original - reconstruction
    return torch.where(residual > 0.0, residual, torch.zeros_like(residual))


# For Domain Adaptation
def create_target_samples(hdf5_target_path, transform, n=39):
    """
    :param hdf5_target_path: the path to the targer dataset
    :param transform: transforamtion for the dataset
    :param n: number of slices per class (have to think that 1 patient has 155 slices so divided by num of classes)
    :return: target tensor with images and their labels
    """
    brats_val_dataset = Brats2017HDF5Dataset(hdf5_file_path=hdf5_target_path, transform=transform, uppercase_keys=False)
    dataset = DsetSSRotRand(brats_val_dataset, digit=False)

    X, Y = [], []
    classes = 4 * [n]  # Number has to change automatically w/ number of classes w.r.t each auxiliary task

    i = 0
    while True:
        if len(X) == n * 4:
            break
        x, y, _ = dataset[i]
        if classes[y] > 0:
            X.append(x)
            Y.append(y)
            classes[y] -= 1
        i += 1

    assert (len(X) == n * 4)
    return torch.stack(X, dim=0), torch.from_numpy(np.array(Y))


def sample_data(hdf5_source_path, transform):
    source_set = CamCanHDF5Dataset(hdf5_file_path=hdf5_source_path, transform=transform, uppercase_keys=False)
    dataset = DsetSSRotRand(source_set, digit=False)

    n = len(dataset)

    X = torch.Tensor(n, 1, 128, 128)
    mask = torch.Tensor(n, 1, 128, 128)
    Y = torch.LongTensor(n)

    inds = torch.randperm(len(dataset))
    for i, index in enumerate(inds):
        x, y, m = dataset[index]
        X[i] = x
        Y[i] = y
        mask[i] = m
    return X, Y, mask


def create_groups(X_s, Y_s, X_t, Y_t, masks, seed=1):
    """
    G1: a pair of pic comes from same domain, same class label=0
    G3: a pair of pic comes from same domain, different classes label=2
    G2: a pair of pic comes from different domain, same class label=1
    G4: a pair of pic comes from different domain, different classes label=3
    """

    # change seed so every time wo get group data will different in source domain,but in target domain, data not change
    torch.manual_seed(1 + seed)
    torch.cuda.manual_seed(1 + seed)

    n = X_t.shape[0]  # 10*shot

    # shuffle order
    classes = torch.unique(Y_t)
    classes = classes[torch.randperm(len(classes))]  # just to mix up the order of the classes

    class_num = classes.shape[0]
    shot = n // class_num  # it is like having batch_size/amount_of_classes

    def s_idxs(c):
        idx = torch.nonzero(Y_s.eq(int(c)))  # idxs of class c

        return idx[torch.randperm(len(idx))][:shot * 2].squeeze()  # mix the order and take 2*shot of them

    def t_idxs(c):
        return torch.nonzero(Y_t.eq(int(c)))[:shot].squeeze()

    source_idxs = list(map(s_idxs, classes))
    # to get the one with the least amount of classes and have the same amount of images based on it
    min_class = min([source_idxs[i].shape[0] for i in range(len(source_idxs))])
    source_idxs = [source_idxs[i][:min_class] for i in range(len(source_idxs))]

    target_idxs = list(map(t_idxs, classes))

    source_matrix = torch.stack(source_idxs)

    target_matrix = torch.stack(target_idxs)

    G1, G2, G3, G4 = [], [], [], []
    Y1, Y2, Y3, Y4 = [], [], [], []
    masks_G2, masks_G4 = [], []

    for i in range(4):
        for j in range(shot):
            G1.append((X_s[source_matrix[i][j * 2 - 1]], X_s[source_matrix[i][j * 2]]))
            Y1.append((Y_s[source_matrix[i][j * 2 - 1]], Y_s[source_matrix[i][j * 2]]))
            G2.append((X_s[source_matrix[i][j]], X_t[target_matrix[i][j]]))
            Y2.append((Y_s[source_matrix[i][j]], Y_t[target_matrix[i][j]]))
            masks_G2.append(masks[source_matrix[i][j]])
            G3.append((X_s[source_matrix[i % 4][j]], X_s[source_matrix[(i + 1) % 4][j]]))
            Y3.append((Y_s[source_matrix[i % 4][j]], Y_s[source_matrix[(i + 1) % 4][j]]))
            G4.append((X_s[source_matrix[i % 4][j]], X_t[target_matrix[(i + 1) % 4][j]]))
            Y4.append((Y_s[source_matrix[i % 4][j]], Y_t[target_matrix[(i + 1) % 4][j]]))
            masks_G4.append(masks[source_matrix[i % 4][j]])

    groups = [G1, G2, G3, G4]
    groups_y = [Y1, Y2, Y3, Y4]
    masks_G_24 = [masks_G2, masks_G4]

    # make sure we sampled enough samples
    for g in groups:
        assert (len(g) == n)
    return groups, groups_y, masks_G_24


def eval_soft_intro_vae(dataset='cifar10', z_dim=128, lr_e=2e-4, lr_d=2e-4, batch_size=128, num_workers=4,
                        start_epoch=0, exit_on_negative_diff=False,
                        num_epochs=250, num_vae=0, save_interval=50, recon_loss_type="mse",
                        beta_kl=1.0, beta_rec=1.0, beta_neg=1.0, test_iter=1000, seed=-1, pretrained=None,
                        device=torch.device("cpu"), num_row=5, gamma_r=1e-8, with_fid=False):
    """
    :param dataset: dataset to train on: ['cifar10', 'mnist', 'fmnist', 'svhn', 'monsters128', 'celeb128', 'celeb256', 'celeb1024']
    :param z_dim: latent dimensions
    :param lr_e: learning rate for encoder
    :param lr_d: learning rate for decoder
    :param batch_size: batch size
    :param num_workers: num workers for the loading the data
    :param start_epoch: epoch to start from
    :param exit_on_negative_diff: stop run if mean kl diff between fake and real is negative after 50 epochs
    :param num_epochs: total number of epochs to run
    :param num_vae: number of epochs for vanilla vae training
    :param save_interval: epochs between checkpoint saving
    :param recon_loss_type: type of reconstruction loss ('mse', 'l1', 'bce')
    :param beta_kl: beta coefficient for the kl divergence
    :param beta_rec: beta coefficient for the reconstruction loss
    :param beta_neg: beta coefficient for the kl divergence in the expELBO function
    :param test_iter: iterations between sample image saving
    :param seed: seed
    :param pretrained: path to pretrained model, to continue training
    :param device: device to run calculation on - torch.device('cuda:x') or torch.device('cpu')
    :param num_row: number of images in a row gor the sample image saving
    :param gamma_r: coefficient for the reconstruction loss for fake data in the decoder
    :param with_fid: calculate FID during training (True/False)
    :return:
    """
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    # --------------build models -------------------------
    if dataset == 'cifar10':
        image_size = 32
        channels = [64, 128, 256]
        train_set = CIFAR10(root='./cifar10_ds', train=False, download=True, transform=transforms.ToTensor())
        ch = 3
    elif dataset == 'camcan':
        image_size = 128
        ch = 1
        BRATS_CAMCAN_DEFAULT_TRANSFORM = transforms.Compose([
            NumpyReshapeTransform((200, 200)),
            Numpy2PILTransform(),
            transforms.Resize((128, 128)),
            transforms.PILToTensor()])

        # camcan_t2_val_path: Path = "/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/small" \
        #                            "/camcan_val_t2_hm_std_bv3.5_xe.hdf5 "
        # camcan_t2_train_path: Path = '/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/small' \
        #                              '/camcan_train_t2_hm_std_bv3.5_xe.hdf5 '
        camcan_t2_train_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
                                    "camcan_train_t2_hm_std_bv3.5_xe.hdf5")
        camcan_t2_val_path = Path('/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/'
                                  'camcan_val_t2_hm_std_bv3.5_xe.hdf5')
        ood_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/processed/"
                            "brats17_t2_bc_std_bv3.5.hdf5")
        brats17_t2_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/processed/"
                                   "brats17_t2_bc_std_bv3.5.hdf5")
        brats17_t2_hm_set_path = Path(
            "/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
            "brats17_t2_hm_bc_std_bv3.5_xe.hdf5")
        brats17_t1_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
                                   "brats17_t1_bc_std_bv3.5_xe.hdf5")
        brats17_t1_hm_set_path = Path(
            "/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
            "brats17_t1_hm_bc_std_bv3.5_xe.hdf5")

        train_dataloader, val_dataloader = camcan_data_loader(hdf5_train_path=camcan_t2_train_path,
                                                              hdf5_val_path=camcan_t2_val_path, batch_size=128,
                                                              shuffle_train=True, shuffle_val=True, num_workers=4,
                                                              transform=BRATS_CAMCAN_DEFAULT_TRANSFORM,
                                                              uppercase_keys=False, add_gauss_blobs=True,
                                                              domain_adapt=False)

        # X_s, Y_s, masks = sample_data(camcan_t2_val_path, BRATS_CAMCAN_DEFAULT_TRANSFORM)
        # num_patients = 2
        # X_t, Y_t = create_target_samples(brats17_t2_set_path, BRATS_CAMCAN_DEFAULT_TRANSFORM, n=39 * num_patients)
        # print(X_t.shape[0])
        # groups, labels, masks_g = create_groups(X_s, Y_s, X_t, Y_t, masks, seed=1)
        # _, G2, _, G4 = groups
        # _, Y2, _, Y4 = labels
        # masks_g2, masks_g4 = masks_g
        # x21s, x22t = G2[40]
        # y21s, y22t = Y2[40]
        # M2s = masks_g2[40]
        # X2s = torch.stack((x21s, x22t))
        # # M2s = torch.stack(m21s)
        # x41s, x42s = G4[0]
        # vutils.save_image(torch.cat([normalize_to_0_1(X_s[40].squeeze()), masks[40].squeeze(),
        #                              normalize_to_0_1(X2s[0].squeeze()), M2s[0].squeeze(),
        #                              normalize_to_0_1(X2s[1].squeeze())], dim=0).data.cpu(),
        #                   '{}/image_{}.jpg'.format("./Figures_validation",
        #                                            "10_3_20220_camcan_after_gouping_and_choosing"), nrow=1)

        # print("Dataloaders ready with Gauss blobs")
        ood_dataloader = brats17_val_dataloader(hdf5_path=brats17_t1_hm_set_path, batch_size=128, shuffle=True,
                                                num_workers=4,
                                                transform=BRATS_CAMCAN_DEFAULT_TRANSFORM, domain_adapt=False)
    elif dataset == 'celeb128':
        channels = [64, 128, 256, 512, 512]
        image_size = 128
        ch = 3
        output_height = 128
        train_size = 162770
        data_root = '../data/celeb256/img_align_celeba'
        image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        train_list = image_list[:train_size]
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    elif dataset == 'celeb256':
        channels = [64, 128, 256, 512, 512, 512]
        image_size = 256
        ch = 3
        output_height = 256
        train_size = 162770
        data_root = '../data/celeb256/img_align_celeba'
        image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        train_list = image_list[:train_size]
        assert len(train_list) > 0
        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    elif dataset == 'celeb1024':
        channels = [16, 32, 64, 128, 256, 512, 512, 512]
        image_size = 1024
        ch = 3
        output_height = 1024
        train_size = 29000
        data_root = './' + dataset
        image_list = [x for x in os.listdir(data_root) if is_image_file(x)]
        train_list = image_list[:train_size]
        assert len(train_list) > 0

        train_set = ImageDatasetFromFile(train_list, data_root, input_height=None, crop_height=None,
                                         output_height=output_height, is_mirror=True)
    elif dataset == 'monsters128':
        channels = [64, 128, 256, 512, 512]
        image_size = 128
        ch = 3
        data_root = './monsters_ds/'
        train_set = DigitalMonstersDataset(root_path=data_root, output_height=image_size)
    elif dataset == 'svhn':
        image_size = 32
        channels = [64, 128, 256]
        train_set = SVHN(root='./svhn', split='test', transform=transforms.ToTensor(),
                         download=True)  # TODO: change split to test!!
        ch = 3
    elif dataset == 'fmnist':
        image_size = 28
        channels = [64, 128]
        train_set = FashionMNIST(root='./fmnist_ds', train=True, download=True, transform=transforms.ToTensor())
        ch = 1
    elif dataset == 'mnist':
        image_size = 28
        channels = [64, 128]
        train_set = MNIST(root='./mnist_ds', train=True, download=True, transform=transforms.ToTensor())
        ch = 1
    else:
        raise NotImplementedError("dataset is not supported")

    # Eliminate classes 0 and 3 for training without them
    # labels_array = np.array(train_set.targets)  # list needs to be made into np.array
    # labels_0_array = np.array(labels_array != 0)
    # labels_3_array = np.array(labels_array != 3)
    # final_label_array = np.logical_and(labels_0_array, labels_3_array)
    # final_label_array = np.logical_not(final_label_array)
    #
    # train_set.targets = labels_array[final_label_array].tolist()
    # train_set.data = train_set.data[final_label_array]
    # print('Updated Dataloader without classes 0 and 3 is ready')

    # train_data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_data_loader = ood_dataloader
    # start_time = time.time()
    #
    # cur_iter = 0
    # kls_real = []
    # kls_fake = []
    # kls_rec = []
    # rec_errs = []
    # exp_elbos_f = []
    # exp_elbos_r = []
    # best_fid = None
    # for epoch in range(start_epoch, num_epochs):
    #     if with_fid and ((epoch == 0) or (epoch >= 100 and epoch % 20 == 0) or epoch == num_epochs - 1):
    #         with torch.no_grad():
    #             print("calculating fid...")
    #             fid = calculate_fid_given_dataset(train_data_loader, model, batch_size, cuda=True, dims=2048,
    #                                               device=device, num_images=50000)
    #             print("fid:", fid)
    #             if best_fid is None:
    #                 best_fid = fid
    #             elif best_fid > fid:
    #                 print("best fid updated: {} -> {}".format(best_fid, fid))
    #                 best_fid = fid
    #                 # save
    #                 save_epoch = epoch
    #                 prefix = dataset + "_soft_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
    #                     beta_rec) + "_" + "fid_" + str(fid) + "_"
    #                 save_checkpoint(model, save_epoch, cur_iter, prefix)
    #
    #     diff_kls = []
    #     # save models
    #     if epoch % save_interval == 0 and epoch > 0:
    #         save_epoch = (epoch // save_interval) * save_interval
    #         prefix = dataset + "_soft_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
    #             beta_rec) + "_"
    #         save_checkpoint(model, save_epoch, cur_iter, prefix)
    #
    #     model.train()
    #
    #     batch_kls_real = []
    #     batch_kls_fake = []
    #     batch_kls_rec = []
    #     batch_rec_errs = []
    #     batch_exp_elbo_f = []
    #     batch_exp_elbo_r = []
    #
    #     pbar = tqdm(iterable=train_data_loader)
    #
    #     for batch in pbar:
    #         # --------------train------------
    #         if dataset in ["cifar10", "svhn", "fmnist", "mnist"]:
    #             batch = batch[0]
    #         if epoch < num_vae:
    #             if len(batch.size()) == 3:
    #                 batch = batch.unsqueeze(0)
    #
    #             batch_size = batch.size(0)
    #
    #             real_batch = batch.to(device)
    #
    #             # =========== Update E, D ================
    #
    #             real_mu, real_logvar, z, rec = model(real_batch)
    #
    #             loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
    #             loss_kl = calc_kl(real_logvar, real_mu, reduce="mean")
    #
    #             loss = beta_rec * loss_rec + beta_kl * loss_kl
    #
    #             optimizer_d.zero_grad()
    #             optimizer_e.zero_grad()
    #             loss.backward()
    #             optimizer_e.step()
    #             optimizer_d.step()
    #
    #             pbar.set_description_str('epoch #{}'.format(epoch))
    #             pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=loss_kl.data.cpu().item())
    #
    #             if cur_iter % test_iter == 0:
    #                 vutils.save_image(torch.cat([real_batch, rec], dim=0).data.cpu(),
    #                                   '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)
    #
    #         else:
    #             if len(batch.size()) == 3:
    #                 batch = batch.unsqueeze(0)
    #
    #             b_size = batch.size(0)
    #             noise_batch = torch.randn(size=(b_size, z_dim)).to(device)
    #
    #             real_batch = batch.to(device)
    #
    #             # =========== Update E ================
    #             for param in model.encoder.parameters():
    #                 param.requires_grad = True
    #             for param in model.decoder.parameters():
    #                 param.requires_grad = False
    #
    #             fake = model.sample(noise_batch)
    #
    #             real_mu, real_logvar = model.encode(real_batch)
    #             z = reparameterize(real_mu, real_logvar)
    #             rec = model.decoder(z)
    #
    #             loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
    #
    #             lossE_real_kl = calc_kl(real_logvar, real_mu, reduce="mean")
    #
    #             rec_mu, rec_logvar, z_rec, rec_rec = model(rec.detach())
    #             fake_mu, fake_logvar, z_fake, rec_fake = model(fake.detach())
    #
    #             kl_rec = calc_kl(rec_logvar, rec_mu, reduce="none")
    #             kl_fake = calc_kl(fake_logvar, fake_mu, reduce="none")
    #
    #             loss_rec_rec_e = calc_reconstruction_loss(rec, rec_rec, loss_type=recon_loss_type, reduction='none')
    #             while len(loss_rec_rec_e.shape) > 1:
    #                 loss_rec_rec_e = loss_rec_rec_e.sum(-1)
    #             loss_rec_fake_e = calc_reconstruction_loss(fake, rec_fake, loss_type=recon_loss_type, reduction='none')
    #             while len(loss_rec_fake_e.shape) > 1:
    #                 loss_rec_fake_e = loss_rec_fake_e.sum(-1)
    #
    #             expelbo_rec = (-2 * scale * (beta_rec * loss_rec_rec_e + beta_neg * kl_rec)).exp().mean()
    #             expelbo_fake = (-2 * scale * (beta_rec * loss_rec_fake_e + beta_neg * kl_fake)).exp().mean()
    #
    #             lossE_fake = 0.25 * (expelbo_rec + expelbo_fake)
    #             lossE_real = scale * (beta_rec * loss_rec + beta_kl * lossE_real_kl)
    #
    #             lossE = lossE_real + lossE_fake
    #             optimizer_e.zero_grad()
    #             lossE.backward()
    #             optimizer_e.step()
    #
    #             # ========= Update D ==================
    #             for param in model.encoder.parameters():
    #                 param.requires_grad = False
    #             for param in model.decoder.parameters():
    #                 param.requires_grad = True
    #
    #             fake = model.sample(noise_batch)
    #             rec = model.decoder(z.detach())
    #             loss_rec = calc_reconstruction_loss(real_batch, rec, loss_type=recon_loss_type, reduction="mean")
    #
    #             rec_mu, rec_logvar = model.encode(rec)
    #             z_rec = reparameterize(rec_mu, rec_logvar)
    #
    #             fake_mu, fake_logvar = model.encode(fake)
    #             z_fake = reparameterize(fake_mu, fake_logvar)
    #
    #             rec_rec = model.decode(z_rec.detach())
    #             rec_fake = model.decode(z_fake.detach())
    #
    #             loss_rec_rec = calc_reconstruction_loss(rec.detach(), rec_rec, loss_type=recon_loss_type,
    #                                                     reduction="mean")
    #             loss_fake_rec = calc_reconstruction_loss(fake.detach(), rec_fake, loss_type=recon_loss_type,
    #                                                      reduction="mean")
    #
    #             lossD_rec_kl = calc_kl(rec_logvar, rec_mu, reduce="mean")
    #             lossD_fake_kl = calc_kl(fake_logvar, fake_mu, reduce="mean")
    #
    #             lossD = scale * (loss_rec * beta_rec + (
    #                     lossD_rec_kl + lossD_fake_kl) * 0.5 * beta_kl + gamma_r * 0.5 * beta_rec * (
    #                                      loss_rec_rec + loss_fake_rec))
    #
    #             optimizer_d.zero_grad()
    #             lossD.backward()
    #             optimizer_d.step()
    #             if torch.isnan(lossD) or torch.isnan(lossE):
    #                 raise SystemError
    #
    #             dif_kl = -lossE_real_kl.data.cpu() + lossD_fake_kl.data.cpu()
    #             pbar.set_description_str('epoch #{}'.format(epoch))
    #             pbar.set_postfix(r_loss=loss_rec.data.cpu().item(), kl=lossE_real_kl.data.cpu().item(),
    #                              diff_kl=dif_kl.item(), expelbo_f=expelbo_fake.cpu().item())
    #
    #             diff_kls.append(-lossE_real_kl.data.cpu().item() + lossD_fake_kl.data.cpu().item())
    #             batch_kls_real.append(lossE_real_kl.data.cpu().item())
    #             batch_kls_fake.append(lossD_fake_kl.cpu().item())
    #             batch_kls_rec.append(lossD_rec_kl.data.cpu().item())
    #             batch_rec_errs.append(loss_rec.data.cpu().item())
    #             batch_exp_elbo_f.append(expelbo_fake.data.cpu())
    #             batch_exp_elbo_r.append(expelbo_rec.data.cpu())
    #
    #             if cur_iter % test_iter == 0:
    #                 _, _, _, rec_det = model(real_batch, deterministic=True)
    #                 max_imgs = min(batch.size(0), 16)
    #                 vutils.save_image(
    #                     torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
    #                     '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)
    #
    #         cur_iter += 1
    #     e_scheduler.step()
    #     d_scheduler.step()
    #     pbar.close()
    #     if exit_on_negative_diff and epoch > 50 and np.mean(diff_kls) < -1.0:
    #         print(
    #             f'the kl difference [{np.mean(diff_kls):.3f}] between fake and real is negative (no sampling improvement)')
    #         print("try to lower beta_neg hyperparameter")
    #         print("exiting...")
    #         raise SystemError("Negative KL Difference")
    #
    #     if epoch > num_vae - 1:
    #         kls_real.append(np.mean(batch_kls_real))
    #         kls_fake.append(np.mean(batch_kls_fake))
    #         kls_rec.append(np.mean(batch_kls_rec))
    #         rec_errs.append(np.mean(batch_rec_errs))
    #         exp_elbos_f.append(np.mean(batch_exp_elbo_f))
    #         exp_elbos_r.append(np.mean(batch_exp_elbo_r))
    #         # epoch summary
    #         print('#' * 50)
    #         print(f'Epoch {epoch} Summary:')
    #         print(f'beta_rec: {beta_rec}, beta_kl: {beta_kl}, beta_neg: {beta_neg}')
    #         print(
    #             f'rec: {rec_errs[-1]:.3f}, kl: {kls_real[-1]:.3f}, kl_fake: {kls_fake[-1]:.3f}, kl_rec: {kls_rec[-1]:.3f}')
    #         print(
    #             f'diff_kl: {np.mean(diff_kls):.3f}, exp_elbo_f: {exp_elbos_f[-1]:.4e}, exp_elbo_r: {exp_elbos_r[-1]:.4e}')
    #         print(f'time: {time.time() - start_time}')
    #         print('#' * 50)
    #     if epoch == num_epochs - 1:
    #         with torch.no_grad():
    #             _, _, _, rec_det = model(real_batch, deterministic=True)
    #             noise_batch = torch.randn(size=(b_size, z_dim)).to(device)
    #             fake = model.sample(noise_batch)
    #             max_imgs = min(batch.size(0), 16)
    #             vutils.save_image(
    #                 torch.cat([real_batch[:max_imgs], rec_det[:max_imgs], fake[:max_imgs]], dim=0).data.cpu(),
    #                 '{}/image_{}.jpg'.format(fig_dir, cur_iter), nrow=num_row)
    #
    #         # plot graphs
    #         fig = plt.figure()
    #         ax = fig.add_subplot(1, 1, 1)
    #         ax.plot(np.arange(len(kls_real)), kls_real, label="kl_real")
    #         ax.plot(np.arange(len(kls_fake)), kls_fake, label="kl_fake")
    #         ax.plot(np.arange(len(kls_rec)), kls_rec, label="kl_rec")
    #         ax.plot(np.arange(len(rec_errs)), rec_errs, label="rec_err")
    #         ax.legend()
    #         plt.savefig('./soft_intro_train_graphs.jpg')
    #         with open('./soft_intro_train_graphs_data.pickle', 'wb') as fp:
    #             graph_dict = {"kl_real": kls_real, "kl_fake": kls_fake, "kl_rec": kls_rec, "rec_err": rec_errs}
    #             pickle.dump(graph_dict, fp)
    #         # save models
    #         prefix = dataset + "_soft_intro" + "_betas_" + str(beta_kl) + "_" + str(beta_neg) + "_" + str(
    #             beta_rec) + "_"
    #         save_checkpoint(model, epoch, cur_iter, prefix)
    #         model.train()

    # Initialize model with pretrained model

    # model = SoftIntroVAE(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size).to(device)
    model = BaurModel().to(device)

    folder = "./19_4_20220/camcan_bv3.5_maskedrec_soft_intro_betas_1.0_256_3000.0_seed_0"
    models = {}
    i = 0
    for files in os.listdir(folder):
        if files.endswith(".pth"):
            model_epoch = int(files.split('_')[2])
            saved_model = os.path.join(folder, files)
            models[i] = {"model_epoch": model_epoch, "saved_model": saved_model}
            i += 1
        else:
            pass

    ordered_models = collections.OrderedDict(sorted(models.items(), key=lambda t: t[1]["model_epoch"]))
    # for pretrained in models:
    for key in ordered_models:
        epoch = ordered_models[key]["model_epoch"]
        pretrained = ordered_models[key]["saved_model"]
        pretrained_5 = "./19_4_20220/camcan_bv3.5_maskedrec_soft_intro_betas_1.0_256_3000.0_seed_5/model_epoch_52_iter_30940.pth"
        pretrained = "./10_3_20220/camcan_bv3.5_maskedrec_soft_intro_betas_1.0_256_3000.0/model_epoch_51_iter_30345.pth"
        pretrained_0 = "./19_4_20220/camcan_bv3.5_maskedrec_soft_intro_betas_1.0_256_3000.0_seed_0/model_epoch_53_iter_31535.pth"
        pretrained_t2_1 = "./23_4_20220/soft_intro/camcan_domain_adaptation_t2_num_patients_1_beta_dcd_0.01_negative_dcd_loss/model_epoch_100_iter_59500.pth"
        pretrained_t1_1 = "./24_4_20220/soft_intro/camcan_domain_adaptation_t1_num_patients_1_beta_dcd_0.01_negative_dcd_loss/model_epoch_80_iter_47600.pth"
        pretrained_t1hm_1 = "./24_4_20220/soft_intro/camcan_domain_adaptation_t1hm_num_patients_1_beta_dcd_0.01_negative_dcd_loss/model_epoch_80_iter_47600.pth"
        pretrained_t2hm_1 = "./24_4_20220/soft_intro/camcan_domain_adaptation_t2hm_num_patients_1_beta_dcd_0.01_negative_dcd_loss/model_epoch_80_iter_47600.pth"
        pretrained_t2_50 = "./24_4_20220/soft_intro/camcan_domain_adaptation_t2_num_patients_50_beta_dcd_0.01_negative_dcd_loss/model_epoch_56_iter_33320.pth"
        pretrained_t1_50 = "./24_4_20220/soft_intro/camcan_domain_adaptation_t1_num_patients_50_beta_dcd_0.01_negative_dcd_loss/model_epoch_30_iter_17850.pth"
        pretrained_t1hm_50 = "./24_4_20220/soft_intro/camcan_domain_adaptation_t1hm_num_patients_50_beta_dcd_0.01_negative_dcd_loss/model_epoch_30_iter_17850.pth"
        pretrained_t2hm_50 = "./24_4_20220/soft_intro/camcan_domain_adaptation_t2hm_num_patients_50_beta_dcd_0.01_negative_dcd_loss/model_epoch_68_iter_40460.pth"
        pretrained_t2_25 = "./3_5_20220/soft_intro/camcan_domain_adaptation_t2_num_patients_25_beta_dcd_0.01_negative_dcd_loss/model_epoch_50_iter_29750.pth"
        pretrained_t1_25 = "./3_5_20220/soft_intro/camcan_domain_adaptation_t1_num_patients_25_beta_dcd_0.01_negative_dcd_loss/model_epoch_40_iter_23800.pth"
        pretrained_t1hm_25 = "./3_5_20220/soft_intro/camcan_domain_adaptation_t1hm_num_patients_25_beta_dcd_0.01_negative_dcd_loss/model_epoch_60_iter_35700.pth"
        pretrained_t2hm_25 = "./3_5_20220/soft_intro/camcan_domain_adaptation_t2hm_num_patients_25_beta_dcd_0.01_negative_dcd_loss/model_epoch_46_iter_27370.pth"
        pretrained_t2_10 = "./3_5_20220/soft_intro/camcan_domain_adaptation_t2_num_patients_10_beta_dcd_0.01_negative_dcd_loss/model_epoch_46_iter_27370.pth"
        pretrained_t1_10 = "./3_5_20220/soft_intro/camcan_domain_adaptation_t1_num_patients_10_beta_dcd_0.01_negative_dcd_loss/model_epoch_48_iter_28560.pth"
        pretrained_t1hm_10 = "./3_5_20220/soft_intro/camcan_domain_adaptation_t1hm_num_patients_10_beta_dcd_0.01_negative_dcd_loss/model_epoch_54_iter_32130.pth"
        pretrained_t2hm_10 = "./3_5_20220/soft_intro/camcan_domain_adaptation_t2hm_num_patients_10_beta_dcd_0.01_negative_dcd_loss/model_epoch_50_iter_29750.pth"

        pretrained_simple = "./24_4_20220/camcan_Simpple_VAE_beta_rec_1.0_beta_kl_0.0/model_epoch_80_iter_47600.pth"

        if pretrained is not None:
            load_model(model, pretrained_t1hm_25, device)
        model.train()

        # print(f"--------*********  RESULTS FOR MODEL EPOCH: {epoch}  *********--------")
        #
        # types = ['all', 'healthy', 'lesions']
        # for type in types:
        #     results_dict = calc_score_and_auc(model, ds='camcan', ood_ds='brats', batch_size=64, rand_state=None,
        #                                       beta_kl=1.0, beta_rec=3000.0, device=torch.device("cuda"), loss_type='l1',
        #                                       with_nll=False, type=type)
        #     print(f"type: {type}")
        #     print_results(results_dict)
        # break

        # Loop over the whole dataset
        pbar = tqdm(iterable=train_data_loader)
        with torch.no_grad():
            loss = []
            loss_mat = []
            overall_dice = []
            lesional_dice = []
            ovr_recall = []
            ovr_precision = []
            ovr_f_score = []
            for batch in pbar:
                # for batch_i, batch in enumerate(train_data_loader):
                if dataset in ["cifar10", "svhn", "fmnist", "mnist"]:
                    batch = batch[0]
                if dataset == "camcan":
                    # for DA modified camcan
                    # labels = batch[1].to(device)
                    # mask = batch[2].to(device)
                    # real_batch = batch[0].to(device)
                    mask = batch["mask"].to(
                        device)  # first so you get the mask before changing the batch to the actual one
                    seg = batch["seg"].to(device)
                    batch = batch["scan"]

                if len(batch.size()) == 3:
                    batch = batch.unsqueeze(0)

                b_size = batch.size(0)
                noise_batch = torch.randn(size=(b_size, z_dim)).to(device)
                real_batch = batch.to(device)
                _, _, _, rec_det = model(real_batch)
                prediction = residual_l1_max(rec_det, real_batch)
                # masked[mask == False] = -3.5

                fake = model.sample(noise_batch)
                max_imgs = min(real_batch.size(0), 85)
                max_val = torch.max(real_batch)
                min_val = torch.min(real_batch)
                threshold = 2.1
                prediction[prediction <= threshold] = 0
                prediction[prediction > threshold] = 1
                #     batch_precision = precision(prediction[torch.sum(seg, dim=(1, 2, 3)) > 0].cpu().numpy(),
                #                                 seg[torch.sum(seg, dim=(1, 2, 3)) > 0].cpu().numpy())
                #     batch_recall = recall(prediction[torch.sum(seg, dim=(1, 2, 3)) > 0].cpu().numpy(),
                #                           seg[torch.sum(seg, dim=(1, 2, 3)) > 0].cpu().numpy())
                #     if batch_recall + batch_precision == 0:
                #         batch_f_score = 0
                #     else:
                #         batch_f_score = 2 * batch_recall * batch_precision / (batch_recall + batch_precision)
                #     ovr_recall.append(batch_recall)
                #     ovr_precision.append(batch_precision)
                #     ovr_f_score.append(batch_f_score)
                #     # overall_dice.append(dice(prediction.cpu().numpy(), seg.cpu().numpy()))
                #     # lesional_dice.append(dice(prediction[torch.sum(seg, dim=(1, 2, 3)) > 0].cpu().numpy(),
                #     #                           seg[torch.sum(seg, dim=(1, 2, 3)) > 0].cpu().numpy()))
                # print(f"Overall f_score: {np.mean(ovr_f_score)}")
                # print(f"Overall precision: {np.mean(ovr_precision)}")
                # print(f"Overall recall: {np.mean(ovr_recall)}")
                # # print(f"Overall dice: {np.mean(overall_dice)}")
                # # print(f"Lesional dice only: {np.mean(lesional_dice)}")
                # break
                real_batch_norm = normalize_to_0_1(real_batch, min_val, max_val)
                reconstruction_norm = normalize_to_0_1(rec_det)  # have to fix for min,max value
                real_batch_mask = mask_background_to_zero(real_batch_norm, mask)
                a, b, c, d, e, f, g, h = real_batch_mask[(max_imgs - 8):max_imgs].squeeze().data.cpu()
                reconstruction = mask_background_to_minus_three(rec_det, mask)
                a_rec, b_rec, c_rec, d_rec, e_rec, f_rec, g_rec, h_rec = reconstruction[(max_imgs - 8):max_imgs].squeeze().data.cpu()

                prediction_norm = normalize_to_0_1(prediction, min_val, max_val)
                fake_norm = normalize_to_0_1(fake)
                prediction_masked = mask_background_to_zero(prediction, mask)
                aa, bb, cc, dd, ee, ff, gg, hh = prediction_masked[(max_imgs - 8):max_imgs].squeeze().data.cpu()
                aa_seg, bb_seg, cc_seg, dd_seg, ee_seg, ff_seg, gg_seg, hh_seg = seg[(max_imgs - 8):max_imgs].squeeze().data.cpu()

                plt.imsave("./Figures_thesis/after_da/plt_brats_t1_hm_input.png",
                           np.hstack([a, b, c, d, e, f, g, h]), cmap="gray")
                plt.imsave("./Figures_thesis/after_da/plt_brats_t1_hm_recons_after_10.png",
                           np.hstack([a_rec, b_rec, c_rec, d_rec, e_rec, f_rec, g_rec, h_rec]), cmap="gray")
                if pretrained is not None:
                    load_model(model, pretrained, device)
                model.train()
                _, _, _, rec_det = model(real_batch)
                reconstruction = mask_background_to_minus_three(rec_det, mask)
                a_rec, b_rec, c_rec, d_rec, e_rec, f_rec, g_rec, h_rec = reconstruction[
                                                                         (max_imgs - 8):max_imgs].squeeze().data.cpu()
                plt.imsave("./Figures_thesis/after_da/plt_brats_t1_hm_recons_before.png",
                           np.hstack([a_rec, b_rec, c_rec, d_rec, e_rec, f_rec, g_rec, h_rec]), cmap="gray")

                # plt.imsave("./Figures_thesis/plt_before_da_t2hm.png",
                #            np.hstack([c, c_rec, cc_seg, cc]), cmap="gray")

                # vutils.save_image(
                #     torch.cat([real_batch_mask[(max_imgs - num_row)], reconstruction[(max_imgs - num_row)],
                #                seg[(max_imgs - num_row)],
                #                prediction_masked[(max_imgs - num_row)]], dim=0).data.cpu(),
                #     '{}/image_{}.jpg'.format("./Figures_thesis", "before_da_t2hm"), nrow=num_row)
                # # vutils.save_image(
                # #     torch.cat([real_batch_norm[40:max_imgs], reconstruction_norm[40:max_imgs], reconstruction[40:max_imgs],
                # #                prediction_masked[40:max_imgs], seg[40:max_imgs]],
                # #               dim=0).data.cpu(),
                # #     '{}/image_{}.jpg'.format("./Figures_validation/segmentations",
                # #                              "22_4_20220_t2_real_batches_recons_preds_seg_21_da"), nrow=num_row)
                # # vutils.save_image(
                # #     torch.cat([real_batch_norm[40:max_imgs], reconstruction[40:max_imgs],
                # #                fake_norm[40:max_imgs]], dim=0).data.cpu(),
                # #     '{}/image_{}.jpg'.format("./Figures_validation/noise_generated",
                # #                              "19_4_20220_camcan_generated_rec_3000_seed_0_epoch_70"), nrow=num_row)
                # # print(labels[40:max_imgs])
                embed()
        #         # labels_prev = labels

        # b_size = batch.size(0)
        # sample_size = 5
        # latent_z_dim = model.zdim
    #             real_batch = batch.to(device)
    #
    #             real_mu, real_logvar, _, rec = model(real_batch)
    #             # decoder_loss = calc_reconstruction_loss(real_batch*mask, rec*mask, loss_type='l1')
    #             _, _, loss_rec, _, _ = loss_function(rec, real_batch, mask, real_mu, real_logvar)
    #             # loss.append(decoder_loss)
    #             loss_mat.append(loss_rec)
    #             # embed()
    #         # writer.add_scalar("Loss_rec/validation", sum(loss) / len(loss), epoch)
    #         writer.add_scalar("Loss_rec/validation", sum(loss_mat) / len(loss_mat), epoch)
    # writer.close()


def domain_adaptaion(dataset='camcan', target='t2', epochs=100, device=torch.device("cpu"), save_interval=1,
                     pretrained="./10_3_20220/camcan_bv3.5_maskedrec_soft_intro_betas_1.0_256_3000.0/"
                                "model_epoch_51_iter_30345.pth", seed=0, num_patients=50):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print("random seed: ", seed)

    if dataset == 'camcan':
        image_size = 128
        ch = 1
        BRATS_CAMCAN_DEFAULT_TRANSFORM = transforms.Compose([
            NumpyReshapeTransform((200, 200)),
            Numpy2PILTransform(),
            transforms.Resize((128, 128)),
            transforms.PILToTensor()])

        # camcan_t2_val_path: Path = "/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/small" \
        #                            "/camcan_val_t2_hm_std_bv3.5_xe.hdf5 "
        # camcan_t2_train_path: Path = '/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/small' \
        #                              '/camcan_train_t2_hm_std_bv3.5_xe.hdf5 '
        camcan_t2_train_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
                                    "camcan_train_t2_hm_std_bv3.5_xe.hdf5")
        camcan_t2_val_path = Path('/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/'
                                  'camcan_val_t2_hm_std_bv3.5_xe.hdf5')

        if target == 't2':
            brats17_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/"
                                    "processed/brats17_t2_bc_std_bv3.5.hdf5")
        if target == 't2hm':
            brats17_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/"
                                    "processed/brats17_t2_hm_bc_std_bv3.5_xe.hdf5")
        if target == 't1':
            brats17_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
                                    "brats17_t1_bc_std_bv3.5_xe.hdf5")
        if target == 't1hm':
            brats17_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/"
                                    "processed/brats17_t1_hm_bc_std_bv3.5_xe.hdf5")

        # train_dataloader, val_dataloader = camcan_data_loader(hdf5_train_path=camcan_t2_train_path,
        #                                                       hdf5_val_path=camcan_t2_val_path, batch_size=128,
        #                                                       shuffle_train=True, shuffle_val=True, num_workers=4,
        #                                                       transform=BRATS_CAMCAN_DEFAULT_TRANSFORM,
        #                                                       uppercase_keys=False, add_gauss_blobs=False,
        #                                                       domain_adapt=True)
        X_s, Y_s = sample_data(camcan_t2_train_path, BRATS_CAMCAN_DEFAULT_TRANSFORM)
        X_t, Y_t = create_target_samples(brats17_set_path, BRATS_CAMCAN_DEFAULT_TRANSFORM, n=39 * num_patients)

    experiment_dir = './4_4_20220'
    os.makedirs(experiment_dir, exist_ok=True)

    loss_fn = torch.nn.CrossEntropyLoss()

    model = BaurModel().to(device)
    if pretrained is not None:
        load_model(model, pretrained, device)
    model.train()

    discriminator = DCD(input_features=256).to(device)
    discriminator.train()
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.001)
    cur_iter = 0
    start_time = time.time()
    for epoch in range(epochs):

        # save models
        if epoch % save_interval == 0 and epoch > 0:
            save_epoch = (epoch // save_interval) * save_interval
            prefix = "4_4_20220/" + dataset + "_domain_adaptation_t2" + "_batch_40_num_patients_" + str(num_patients) + \
                     "_lr_01_dcd_changes/"
            save_checkpoint(model, save_epoch, cur_iter, prefix)

        # data
        groups, group_labels = create_groups(X_s, Y_s, X_t, Y_t, seed=epoch)

        n_iters = 4 * len(groups[1])
        index_list = torch.randperm(n_iters)
        mini_batch_size = 40  # use mini_batch train can be more stable

        loss_mean = []

        X1 = []
        X2 = []
        ground_truths = []
        # go through groups to make mini_batches for DCD
        for index in range(n_iters):

            ground_truth = index_list[index] // len(groups[1])  # ground_truth group chosen at random

            x1, x2 = groups[ground_truth][
                index_list[index] - len(groups[1]) * ground_truth]  # differrent images from chosen group
            X1.append(x1)
            X2.append(x2)
            ground_truths.append(ground_truth)

            # select data for a mini-batch to train
            if (index + 1) % mini_batch_size == 0:
                X1 = torch.stack(X1)
                X2 = torch.stack(X2)
                ground_truths = torch.LongTensor(ground_truths)
                X1 = X1.to(device)
                X2 = X2.to(device)
                ground_truths = ground_truths.to(device)

                optimizer_D.zero_grad()
                mu1, logvar1 = model.encode(X1)
                z1 = reparameterize(mu1, logvar1)
                mu2, logvar2 = model.encode(X2)
                z2 = reparameterize(mu2, logvar2)
                X_cat = torch.cat([z1, z2], 1)
                y_pred = discriminator(X_cat.detach())
                loss = loss_fn(y_pred, ground_truths)
                loss.backward()
                optimizer_D.step()
                loss_mean.append(loss.item())
                X1 = []
                X2 = []
                ground_truths = []
                cur_iter += 1

        print('Printing losses on tensorboard for curent epoch')
        writer.add_scalar("LossDCD/train", sum(loss_mean) / len(loss_mean), epoch)
    print(f'time: {time.time() - start_time}')
    writer.close()


if __name__ == '__main__':
    eval_soft_intro_vae(dataset='camcan', z_dim=128, batch_size=32, num_workers=4, pretrained=None,
                        device=torch.device("cuda"))
