# standard
import collections
import os
import pickle
import random
import time
from pathlib import Path
from typing import Tuple, Any, Optional, Union, Iterable

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
from classification import false_positive_rate
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
from uncertify.scripts import add_uncertify_to_path
from uncertify.uncertify.data.artificial import BrainGaussBlobDataset
from uncertify.uncertify.data.datasets import CamCanHDF5Dataset, Brats2017HDF5Dataset
from uncertify.uncertify.data.np_transforms import NumpyReshapeTransform, Numpy2PILTransform
from uncertify.uncertify.models.encoder_decoder_baur2020 import BaurEncoder, BaurDecoder
from uncertify.uncertify.utils.custom_types import Tensor
from uncertify.uncertify.visualization.grid import imshow_grid
from uda_release.dset_classes.DsetSSRotRand import DsetSSRotRand
from uda_release.dset_classes.DsetNoLabel import DsetNoLabel

# writer = SummaryWriter()

"""
Models
"""


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
                       uppercase_keys: bool = False, add_gauss_blobs: bool = False) -> Tuple[DataLoader, DataLoader]:
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
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    if hdf5_val_path is None:
        val_dataloader = None
    else:
        val_set = CamCanHDF5Dataset(hdf5_file_path=hdf5_val_path, transform=transform, uppercase_keys=uppercase_keys)
        if add_gauss_blobs:
            val_set = BrainGaussBlobDataset(val_set)
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=shuffle_val, num_workers=num_workers)
    return train_dataloader, val_dataloader


def brats17_val_dataloader(hdf5_path: Path, batch_size: int, shuffle: bool,
                           num_workers: int = 0, transform: Any = None,
                           uppercase_keys: bool = False, add_gauss_blobs: bool = False) -> DataLoader:
    """Create a BraTS dataloader based on a hdf_path."""
    assert hdf5_path.exists(), f'BraTS17 hdf5 file {hdf5_path} does not exist!'
    if transform is None:
        transform = BRATS_CAMCAN_DEFAULT_TRANSFORM
    brats_val_dataset = Brats2017HDF5Dataset(hdf5_file_path=hdf5_path, transform=transform,
                                             uppercase_keys=uppercase_keys)
    if add_gauss_blobs:
        brats_val_dataset = BrainGaussBlobDataset(brats_val_dataset)
    val_dataloader = DataLoader(brats_val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return val_dataloader


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


def residual_l1_max(reconstruction: Tensor, original: Tensor) -> Tensor:
    """Construct l1 difference between original and reconstruction.

    Note: Only positive values in the residual are considered, i.e. values below zero are clamped.
    That means only cases where bright pixels which are brighter in the input (likely lesions) are kept."""
    residual = original - reconstruction
    return torch.where(residual > 0.0, residual, torch.zeros_like(residual))


def eval_soft_intro_vae(dataset='camcan', z_dim=128, lr_e=2e-4, lr_d=2e-4, batch_size=128, num_workers=4,
                        start_epoch=0, exit_on_negative_diff=False,
                        num_epochs=250, num_vae=0, save_interval=50, recon_loss_type="mse",
                        beta_kl=1.0, beta_rec=1.0, beta_neg=1.0, test_iter=1000, seed=-1, pretrained=None,
                        device=torch.device("cpu"), num_row=8, gamma_r=1e-8, with_fid=False):
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

        train_dataloader, val_dataloader = camcan_data_loader(hdf5_train_path=camcan_t2_train_path,
                                                              hdf5_val_path=camcan_t2_val_path, batch_size=128,
                                                              shuffle_train=True, shuffle_val=True, num_workers=4,
                                                              transform=BRATS_CAMCAN_DEFAULT_TRANSFORM,
                                                              uppercase_keys=False, add_gauss_blobs=False)
        # print("Dataloaders ready with Gauss blobs")
        ood_dataloader = brats17_val_dataloader(hdf5_path=ood_set_path, batch_size=128, shuffle=True, num_workers=4,
                                                transform=BRATS_CAMCAN_DEFAULT_TRANSFORM)
    else:
        raise NotImplementedError("dataset is not supported")

    train_data_loader = train_dataloader
    model = BaurModel().to(device)

    # for pretrained in models:
    if pretrained is not None:
        load_model(model, pretrained, device)
    model.train()

    # Loop over the whole dataset
    pbar = tqdm(iterable=train_data_loader)
    with torch.no_grad():
        #  Threshold residual and find lesional samples
        n_lesional_pixels = 50
        thresholds = np.linspace(0.1, 3, 30)
        fprs = []
        for threshold in thresholds:
            fpr = []
            for batch in pbar:
                if dataset in ["cifar10", "svhn", "fmnist", "mnist"]:
                    batch = batch[0]
                if dataset == "camcan":
                    mask = batch["mask"].to(
                        device)  # first so you get the mask before changing the batch to the actual one
                    # seg = batch["seg"].to(device)
                    batch = batch["scan"]

                if len(batch.size()) == 3:
                    batch = batch.unsqueeze(0)
                b_size = batch.size(0)

                # Make map 0 to 1 from residual
                real_batch = batch.to(device)
                _, _, _, rec_det = model(real_batch)
                residual = residual_l1_max(rec_det, real_batch)
                residual[residual <= threshold] = 0
                residual[residual > threshold] = 1

                # Find lesional and false positive rate
                true_labels = np.zeros(b_size)  # Zeros = no lesional samples
                pred_labels = (torch.sum(residual.cpu(), dim=(1, 2, 3)) > n_lesional_pixels).numpy()

                #  For given threshold find FPR per batch
                batch_fpr = false_positive_rate(pred_labels, true_labels)
                fpr.append(batch_fpr)
            fprs.append(np.mean(fpr))
        np.savetxt(f"./19_4_20220/train_fprs_per_threshold_{dataset}_50_01to3.txt", np.array(fprs))


def plot_fpr_vs_residual_threshold(accepted_fpr: float, calculated_threshold: float,
                                   thresholds: Iterable, fpr_train: list, fpr_val: list = None) -> plt.Figure:
    """Plots the training (possibly also validation) set false positive rates vs. associated residual thresholds.

    Arguments:
        accepted_fpr: accepted false positive rate when testing with the training set itself
        calculated_threshold: the threshold which has been calculated based on this accepted false positive rate
        thresholds: a list of residual pixel value thresholds
        fpr_train: the associated false positive rates on the training set
        fpr_val: same but for validation set, not mandatory
    """
    # fig, ax = setup_plt_figure(figsize=(7, 5))
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111)
    ax.plot(thresholds, fpr_train, linewidth=4, linestyle='solid', alpha=0.5, label='Training Set FPR')
    if fpr_val is not None:
        ax.plot(thresholds, fpr_val, linewidth=4, linestyle='dashed', alpha=0.5, label='Validation Set')

    ax.set_ylabel(f'False Positive Rate')
    ax.set_xlabel(f'Residual Pixel Threshold')

    normed_diff = [abs(fpr - accepted_fpr) for fpr in fpr_train]
    ax.plot(thresholds, normed_diff, c='green', alpha=0.7, linewidth=3, label='abs(CamCAN FPR - Accepted FPR)')
    ax.plot(thresholds, [accepted_fpr] * len(thresholds), linestyle='dotted', linewidth=3, color='grey',
            label=f'Accepted FPR ({accepted_fpr:.2f})')
    ax.plot([calculated_threshold, calculated_threshold], [-0.05, 1], linestyle='dotted', color='green', linewidth=3,
            label=f'Threshold ({calculated_threshold:.2f})')
    ax.legend(frameon=False)
    plt.savefig(r'./10_3_20220/threshold_search.png')
    return fig


if __name__ == '__main__':
    # thresholds = np.linspace(0.1, 3, 30)
    # data1 = list(np.loadtxt(r"./10_3_20220/train_fprs_per_threshold_camcan_50_01to3.txt"))
    # figure = plot_fpr_vs_residual_threshold(accepted_fpr=0.05, calculated_threshold=2.1, thresholds=thresholds, fpr_train=
    # data1, fpr_val=None)

    pretrained_s_intro = "./10_3_20220/camcan_bv3.5_maskedrec_soft_intro_betas_1.0_256_3000.0/model_epoch_51_iter_30345.pth"
    pretrained_t2_50 = "./24_4_20220/soft_intro/camcan_domain_adaptation_t2_num_patients_50_beta_dcd_0.01_negative_dcd_loss/model_epoch_56_iter_33320.pth"

    eval_soft_intro_vae(dataset='camcan', z_dim=128, batch_size=32, num_workers=4,
                        pretrained=pretrained_t2_50,
                        device=torch.device("cuda"))
