# standard
import os
import pickle
import random
import time
from pathlib import Path


# import IPython
# from IPython import embed
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from dataset import ImageDatasetFromFile, DigitalMonstersDataset
from metrics.fid_score import calculate_fid_given_dataset
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from torch.distributions import Normal, Independent, MultivariateNormal
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST, SVHN
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from IPython import embed
from uncertify.scripts import add_uncertify_to_path
import uncertify
from uncertify.uncertify.data.datasets import CamCanHDF5Dataset, Brats2017HDF5Dataset
from uncertify.uncertify.data.artificial import BrainGaussBlobDataset
from uncertify.uncertify.data.np_transforms import NumpyReshapeTransform, Numpy2PILTransform
from uncertify.uncertify.models.encoder_decoder_baur2020 import BaurEncoder, BaurDecoder
from uncertify.uncertify.utils.custom_types import Tensor
from typing import Tuple, Any, Optional


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


def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
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
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
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
                       device=torch.device("cpu"), loss_type='mse', with_nll=False):
    """
    Test ROC AUC of SoftIntroVAE
    """
    # load datasets
    if ds == "cifar10":
        dataset = CIFAR10(root='./cifar10_ds', train=False, download=True, transform=transforms.ToTensor())
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
    elif ood_ds == 'cifar10':
        ood_dataset = CIFAR10(root='./cifar10_ds', train=False, download=True, transform=transforms.ToTensor())
    elif ood_ds == 'mnist':
        ood_dataset = MNIST(root='./mnist_ds', train=False, download=True, transform=transforms.ToTensor())
    elif ood_ds == 'fmnist':
        ood_dataset = FashionMNIST(root='./fmnist_ds', train=False, download=True, transform=transforms.ToTensor())
    else:
        raise NotImplementedError

    ds_dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ood_dl = DataLoader(ood_dataset, batch_size=batch_size, shuffle=False)

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
        inputs = batch[0]
        inputs = inputs.to(device)
        labels = torch.zeros(inputs.size(0))

        with torch.no_grad():
            mu, logvar, _, rec = model(inputs)
            rec_mu, rec_logvar, _, rec_rec = model(rec)
            loss_rec = calc_reconstruction_loss(inputs, rec, loss_type=loss_type, reduction="none")
            loss_kl = calc_kl(logvar, mu, reduce="none")
            elbo = loss_rec + loss_kl
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
        inputs = batch[0]
        inputs = inputs.to(device)
        labels = torch.ones(inputs.size(0))

        with torch.no_grad():
            mu, logvar, _, rec = model(inputs)
            rec_mu, rec_logvar, _, rec_rec = model(rec)
            loss_rec = calc_reconstruction_loss(inputs, rec, loss_type=loss_type, reduction="none")
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
    labels, scores = zip(*idx_label_score_kl)
    labels = np.array(labels)
    scores = np.array(scores)
    avg_normal_score = np.mean(scores)

    ood_labels, ood_scores = zip(*ood_idx_label_score_kl)
    ood_labels = np.array(ood_labels)
    ood_scores = np.array(ood_scores)
    avg_ood_score = np.mean(ood_scores)

    total_labels = np.concatenate([labels, ood_labels], axis=0)
    total_scores = np.concatenate([scores, ood_scores], axis=0)
    test_auc = roc_auc_score(total_labels, total_scores)
    kl_dict = {'normal': avg_normal_score, 'ood': avg_ood_score, 'auc': test_auc}
    results_dict['kl'] = kl_dict

    # calculate ROC AUC score - recon
    labels, scores = zip(*idx_label_score_recon)
    labels = np.array(labels)
    scores = np.array(scores)
    avg_normal_score = np.mean(scores)

    ood_labels, ood_scores = zip(*ood_idx_label_score_recon)
    ood_labels = np.array(ood_labels)
    ood_scores = np.array(ood_scores)
    avg_ood_score = np.mean(ood_scores)

    total_labels = np.concatenate([labels, ood_labels], axis=0)
    total_scores = np.concatenate([scores, ood_scores], axis=0)
    test_auc = roc_auc_score(total_labels, total_scores)
    recon_dict = {'normal': avg_normal_score, 'ood': avg_ood_score, 'auc': test_auc}
    results_dict['recon'] = recon_dict

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
    labels, scores = zip(*idx_label_score_double_kl)
    labels = np.array(labels)
    scores = np.array(scores)
    avg_normal_score = np.mean(scores)

    ood_labels, ood_scores = zip(*ood_idx_label_score_double_kl)
    ood_labels = np.array(ood_labels)
    ood_scores = np.array(ood_scores)
    avg_ood_score = np.mean(ood_scores)

    total_labels = np.concatenate([labels, ood_labels], axis=0)
    total_scores = np.concatenate([scores, ood_scores], axis=0)
    test_auc = roc_auc_score(total_labels, total_scores)
    dkl_dict = {'normal': avg_normal_score, 'ood': avg_ood_score, 'auc': test_auc}
    results_dict['dkl'] = dkl_dict

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
    total_loss = batch_rec_err + batch_kl_div # changed from initial kld_loss

    return total_loss, batch_kl_div, batch_rec_err, slice_kl_div, slice_rec_err


#  Dataloaders Medical
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


def eval_soft_intro_vae(dataset='cifar10', z_dim=128, lr_e=2e-4, lr_d=2e-4, batch_size=128, num_workers=4,
                        start_epoch=0, exit_on_negative_diff=False,
                        num_epochs=250, num_vae=0, save_interval=50, recon_loss_type="mse",
                        beta_kl=1.0, beta_rec=2500.0, beta_neg=1.0, test_iter=1000, seed=-1, pretrained=None,
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
        train_set = CIFAR10(root='./cifar10_ds', train=True, download=True, transform=transforms.ToTensor())
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
                                    "camcan_train_t2_hm_std_bv0.0_xe.hdf5")
        camcan_t2_val_path = Path('/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/'
                                  'camcan_val_t2_hm_std_bv3.5_xe.hdf5')

        brats17_t2_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/processed/"
                            "brats17_t2_bc_std_bv3.5.hdf5")
        brats17_t2_hm_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
                            "brats17_t2_hm_bc_std_bv3.5_xe.hdf5")
        brats17_t1_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
                            "brats17_t1_bc_std_bv3.5_xe.hdf5")
        brats17_t1_hm_set_path = Path("/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Skovolas/datasets/processed/"
                            "brats17_t1_hm_bc_std_bv3.5_xe.hdf5")

        # ood_set_path = Path("itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/processed/"
        #                     "brats17_t2_bc_std_bv3.5.hdf5")

        _, val_dataloader = camcan_data_loader(hdf5_train_path=camcan_t2_train_path,
                                               hdf5_val_path=camcan_t2_val_path, batch_size=128,
                                               shuffle_train=True, shuffle_val=False, num_workers=4,
                                               transform=BRATS_CAMCAN_DEFAULT_TRANSFORM,
                                               uppercase_keys=False, add_gauss_blobs=False)

        _, gauss_dataloader = camcan_data_loader(hdf5_train_path=camcan_t2_train_path,
                                                 hdf5_val_path=camcan_t2_val_path, batch_size=128,
                                                 shuffle_train=True, shuffle_val=False, num_workers=4,
                                                 transform=BRATS_CAMCAN_DEFAULT_TRANSFORM,
                                                 uppercase_keys=False, add_gauss_blobs=True)

        brats17_t1_hm_dataloader = brats17_val_dataloader(brats17_t1_hm_set_path, 128, False, 4, BRATS_CAMCAN_DEFAULT_TRANSFORM, False, False)
        brats17_t1_dataloader = brats17_val_dataloader(brats17_t1_set_path, 128, False, 4, BRATS_CAMCAN_DEFAULT_TRANSFORM, False, False)
        brats17_t2_hm_dataloader = brats17_val_dataloader(brats17_t2_hm_set_path, 128, False, 4, BRATS_CAMCAN_DEFAULT_TRANSFORM, False, False)
        brats17_t2_dataloader = brats17_val_dataloader(brats17_t2_set_path, 128, False, 4, BRATS_CAMCAN_DEFAULT_TRANSFORM, False, False)

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

    # model = SoftIntroVAE(cdim=ch, zdim=z_dim, channels=channels, image_size=image_size).to(device)
    # if pretrained is not None:
    #     load_model(model, pretrained, device)
    # print(model)
    #
    # fig_dir = './figures_' + dataset
    # os.makedirs(fig_dir, exist_ok=True)
    #
    # optimizer_e = optim.Adam(model.encoder.parameters(), lr=lr_e)
    # optimizer_d = optim.Adam(model.decoder.parameters(), lr=lr_d)
    #
    # e_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_e, milestones=(350,), gamma=0.1)
    # d_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=(350,), gamma=0.1)

    scale = 1 / (ch * image_size ** 2)  # normalize by images size (channels * height * width)

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

    # Test and ood datasets
    # svhn_set = SVHN(root='./svhn', split='test', transform=transforms.ToTensor(), download=True)

    mnist_tranform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    test_set = MNIST(root='./mnist_ds', train=False, download=True, transform=mnist_tranform)
    mnist_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    camcan_dataloader = val_dataloader
    test_data_loader = gauss_dataloader
    # ood_data_loader = ood_dataloader

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
    if pretrained is not None:
        load_model(model, pretrained, device)
    # print(model)
    model.train()
    splits = pretrained.split('/')
    model_epoch = splits[-1].split('.')[0]

    # Loop over the whole dataset
    pbar = tqdm(iterable=camcan_dataloader)
    pbart1hm = tqdm(iterable=brats17_t1_hm_dataloader)
    pbart1 = tqdm(iterable=brats17_t1_dataloader)
    pbart2hm = tqdm(iterable=brats17_t2_hm_dataloader)
    pbart2 = tqdm(iterable=brats17_t2_dataloader)
    pbartest = tqdm(iterable=test_data_loader)
    pbarmnist = tqdm(iterable=mnist_data_loader)
    first_batch = 1
    with torch.no_grad():
        cifar10_train_log_p = []
        cifar10_test_log_p = []
        mnist_test_log_p = []
        t1_hm_healthy_log_p = []
        t1_hm_lesion_log_p = []
        t1_healthy_log_p = []
        t1_lesion_log_p = []
        t2_hm_lesion_log_p = []
        t2_hm_healthy_log_p = []
        t2_healthy_log_p = []
        t2_lesion_log_p = []

        # for batch in pbarmnist:
        #     # if dataset in ["cifar10", "svhn", "fmnist", "mnist"]: # Due to medical data
        #     batch = batch[0]
        #
        #     if len(batch.size()) == 3:
        #         batch = batch.unsqueeze(0)
        #
        #     # b_size = batch.size(0)
        #     # sample_size = 5
        #     # latent_z_dim = model.zdim
        #     real_batch = batch.to(device)
        #
        #     # # Forward pass to get all necessary variables for POSTERIOR DISTRIBUTION OF ENCODER
        #     # real_mu, real_logvar = model.encode(real_batch)
        #     # encoder_posterior = Independent(Normal(real_mu, torch.exp(0.5 * real_logvar)), 1)
        #     # z_sampled = encoder_posterior.sample([sample_size])
        #     #
        #     # # PRIOR DISTRIBUTION for LATENT SPACE
        #     # mu_o = torch.zeros(b_size, latent_z_dim, dtype=float).to(device)
        #     # var_o = torch.ones(b_size, latent_z_dim, dtype=float).to(device)
        #     # var_d = torch.ones([b_size, 3, 32, 32], dtype=float).to(device)
        #     # latent_prior = Independent(Normal(mu_o, var_o), 1)
        #     #
        #     # # Find PROBABILITIES from DISTRIBUTIONS above (They are log_prob) [5x32]
        #     # z_prob = latent_prior.log_prob(z_sampled)
        #     # # change type to float64 (cus otherwise you have division w/ 0.)
        #     # encoder_prob = encoder_posterior.log_prob(z_sampled).type(torch.float64)
        #     #
        #     # # for loop for the reconstruction loss to take each batched sample at once
        #     # # log_det = torch.log(((2 * np.pi) ** 16) * torch.ones(b_size)).to(device)
        #     # for i in range(sample_size):
        #     #     z = z_sampled[i]
        #     #     rec = model.decoder(z)
        #     #
        #     #     # DISTRIBUTION FOR DECODER LIKELIHOOD INSTEAD OF TAKING THE RECONSTRUCTION LOSS
        #     #     decoder_likelihood = Independent(Normal(rec.type(torch.float64), var_d), 3)
        #     #     decoder_likelihood_prob = decoder_likelihood.log_prob(real_batch.type(torch.float64))
        #     #
        #     #     # decoder_likelihood = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
        #     #     # decoder_likelihood_prob = - (decoder_likelihood / 2) - log_det
        #     #     # decoder_likelihood_prob = torch.exp(decoder_likelihood_prob)
        #     #     # decoder_likelihood.append(decoder_likelihood_prob)
        #     #
        #     #     alpha = z_prob[i] + decoder_likelihood_prob - encoder_prob[i]
        #     #     # alpha = decoder_likelihood_prob * torch.div(z_prob[i], encoder_prob[i])
        #     #     if i == 0:
        #     #         alphas = alpha
        #     #     else:
        #     #         alphas = torch.cat((alphas, alpha), 0)
        #     # alphas = alphas.reshape(-1, b_size)
        #     # # batch_likelihood = torch.log(sum(alphas))
        #     #
        #     # exponentials = torch.zeros(b_size).to(device)
        #     # for j in range(sample_size - 1):
        #     #     exponentials += torch.exp(alphas[j + 1] - alphas[0])
        #     # batch_likelihood = alphas[0] + torch.log(1 + exponentials)
        #     #
        #     # # Monte Carlo accumulation of likelihoods per batch
        #     # if first_batch:
        #     #     mc_likelihood = batch_likelihood
        #     # else:
        #     #     mc_likelihood = torch.cat((mc_likelihood, batch_likelihood))
        #
        #     mu, logvar, _, rec = model(real_batch)
        #     # decoder_loss = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
        #     decoder_loss = calc_reconstruction_loss(real_batch, rec, loss_type='l1')
        #     loss_kl = calc_kl(logvar, mu, reduce="none")
        #     elbo = decoder_loss + loss_kl
        #     # log_p = estimate_log_p_x(model, real_batch, mu=mu, logvar=logvar, num_samples=5, x_sigma=1.0)
        #     mnist_test_log_p.append(elbo)
        #     # cifar10_test_log_p.append(decoder_loss)
        #     first_batch += 1
        # mnist_test_log_p = torch.cat(mnist_test_log_p, dim=0)
        # print('Finished MNIST')

        for batch in pbar:
            if dataset in ["cifar10", "svhn", "fmnist", "mnist"]:
                batch = batch[0]
            if dataset == "camcan":
                mask = batch["mask"].to(device)  # first so you get the mask before changing the batch to the actual one
                batch = batch["scan"]

            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)

            # b_size = batch.size(0)
            # sample_size = 5
            # latent_z_dim = model.zdim
            real_batch = batch.to(device)

            # # Forward pass to get all necessary variables for POSTERIOR DISTRIBUTION OF ENCODER
            # real_mu, real_logvar = model.encode(real_batch)
            # encoder_posterior = Independent(Normal(real_mu, torch.exp(0.5 * real_logvar)), 1)
            # z_sampled = encoder_posterior.sample([sample_size])
            #
            # # PRIOR DISTRIBUTION for LATENT SPACE
            # mu_o = torch.zeros(b_size, latent_z_dim, dtype=float).to(device)
            # var_o = torch.ones(b_size, latent_z_dim, dtype=float).to(device)
            # var_d = torch.ones([b_size, 3, 32, 32], dtype=float).to(device)
            # latent_prior = Independent(Normal(mu_o, var_o), 1)
            #
            # # Find PROBABILITIES from DISTRIBUTIONS above (They are log_prob) [5x32]
            # z_prob = latent_prior.log_prob(z_sampled)
            # # change type to float64 (cus otherwise you have division w/ 0.)
            # encoder_prob = encoder_posterior.log_prob(z_sampled).type(torch.float64)
            #
            # # for loop for the reconstruction loss to take each batched sample at once
            # # log_det = torch.log(((2 * np.pi) ** 16) * torch.ones(b_size)).to(device)
            # for i in range(sample_size):
            #     z = z_sampled[i]
            #     rec = model.decoder(z)
            #
            #     # DISTRIBUTION FOR DECODER LIKELIHOOD INSTEAD OF TAKING THE RECONSTRUCTION LOSS
            #     decoder_likelihood = Independent(Normal(rec.type(torch.float64), var_d), 3)
            #     decoder_likelihood_prob = decoder_likelihood.log_prob(real_batch.type(torch.float64))
            #
            #     # decoder_likelihood = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
            #     # decoder_likelihood_prob = - (decoder_likelihood / 2) - log_det
            #     # decoder_likelihood_prob = torch.exp(decoder_likelihood_prob)
            #     # decoder_likelihood.append(decoder_likelihood_prob)
            #
            #     alpha = z_prob[i] + decoder_likelihood_prob - encoder_prob[i]
            #     # alpha = decoder_likelihood_prob * torch.div(z_prob[i], encoder_prob[i])
            #     if i == 0:
            #         alphas = alpha
            #     else:
            #         alphas = torch.cat((alphas, alpha), 0)
            # alphas = alphas.reshape(-1, b_size)
            # # batch_likelihood = torch.log(sum(alphas))
            #
            # exponentials = torch.zeros(b_size).to(device)
            # for j in range(sample_size - 1):
            #     exponentials += torch.exp(alphas[j + 1] - alphas[0])
            # batch_likelihood = alphas[0] + torch.log(1 + exponentials)
            #
            # # Monte Carlo accumulation of likelihoods per batch
            # if first_batch:
            #     mc_likelihood = batch_likelihood
            # else:
            #     mc_likelihood = torch.cat((mc_likelihood, batch_likelihood))

            mu, logvar, _, rec = model(real_batch)
            # decoder_loss = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
            _, _, loss_rec, _, decoder_loss = loss_function(rec, real_batch, mask, mu, logvar)
            loss_kl = calc_kl(logvar, mu, reduce="none")
            elbo = decoder_loss*beta_rec + loss_kl
            # log_p = estimate_log_p_x(model, real_batch, mu=mu, logvar=logvar, num_samples=5, x_sigma=1.0)
            cifar10_train_log_p.append(elbo)
            # cifar10_test_log_p.append(decoder_loss)
            first_batch += 1
        cifar10_train_log_p = torch.cat(cifar10_train_log_p, dim=0)
        print('Finished CamCAN')

        for batch in pbartest:
            if dataset in ["cifar10", "svhn", "fmnist", "mnist"]:
                batch = batch[0]
            if dataset == "camcan":
                mask = batch["mask"].to(device)  # first so you get the mask before changing the batch to the actual one
                batch = batch["scan"]

            if len(batch.size()) == 3:
                batch = batch.unsqueeze(0)

            # b_size = batch.size(0)
            # sample_size = 5
            # latent_z_dim = model.zdim
            real_batch = batch.to(device)

            # # Forward pass to get all necessary variables for POSTERIOR DISTRIBUTION OF ENCODER
            # real_mu, real_logvar = model.encode(real_batch)
            # encoder_posterior = Independent(Normal(real_mu, torch.exp(0.5 * real_logvar)), 1)
            # z_sampled = encoder_posterior.sample([sample_size])
            #
            # # PRIOR DISTRIBUTION for LATENT SPACE
            # mu_o = torch.zeros(b_size, latent_z_dim, dtype=float).to(device)
            # var_o = torch.ones(b_size, latent_z_dim, dtype=float).to(device)
            # var_d = torch.ones([b_size, 3, 32, 32], dtype=float).to(device)
            # latent_prior = Independent(Normal(mu_o, var_o), 1)
            #
            # # Find PROBABILITIES from DISTRIBUTIONS above (They are log_prob) [5x32]
            # z_prob = latent_prior.log_prob(z_sampled)
            # # change type to float64 (cus otherwise you have division w/ 0.)
            # encoder_prob = encoder_posterior.log_prob(z_sampled).type(torch.float64)
            #
            # # for loop for the reconstruction loss to take each batched sample at once
            # # log_det = torch.log(((2 * np.pi) ** 16) * torch.ones(b_size)).to(device)
            # for i in range(sample_size):
            #     z = z_sampled[i]
            #     rec = model.decoder(z)
            #
            #     # DISTRIBUTION FOR DECODER LIKELIHOOD INSTEAD OF TAKING THE RECONSTRUCTION LOSS
            #     decoder_likelihood = Independent(Normal(rec.type(torch.float64), var_d), 3)
            #     decoder_likelihood_prob = decoder_likelihood.log_prob(real_batch.type(torch.float64))
            #
            #     # decoder_likelihood = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
            #     # decoder_likelihood_prob = - (decoder_likelihood / 2) - log_det
            #     # decoder_likelihood_prob = torch.exp(decoder_likelihood_prob)
            #     # decoder_likelihood.append(decoder_likelihood_prob)
            #
            #     alpha = z_prob[i] + decoder_likelihood_prob - encoder_prob[i]
            #     # alpha = decoder_likelihood_prob * torch.div(z_prob[i], encoder_prob[i])
            #     if i == 0:
            #         alphas = alpha
            #     else:
            #         alphas = torch.cat((alphas, alpha), 0)
            # alphas = alphas.reshape(-1, b_size)
            # # batch_likelihood = torch.log(sum(alphas))
            #
            # exponentials = torch.zeros(b_size).to(device)
            # for j in range(sample_size - 1):
            #     exponentials += torch.exp(alphas[j + 1] - alphas[0])
            # batch_likelihood = alphas[0] + torch.log(1 + exponentials)
            #
            # # Monte Carlo accumulation of likelihoods per batch
            # if first_batch:
            #     mc_likelihood = batch_likelihood
            # else:
            #     mc_likelihood = torch.cat((mc_likelihood, batch_likelihood))

            mu, logvar, _, rec = model(real_batch)
            # decoder_loss = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
            _, _, loss_rec, _, decoder_loss = loss_function(rec, real_batch, mask, mu, logvar)
            loss_kl = calc_kl(logvar, mu, reduce="none")
            elbo = decoder_loss*beta_rec + loss_kl
            # log_p = estimate_log_p_x(model, real_batch, mu=mu, logvar=logvar, num_samples=5, x_sigma=1.0)
            cifar10_test_log_p.append(elbo)
            # cifar10_test_log_p.append(decoder_loss)
            first_batch += 1
        cifar10_test_log_p = torch.cat(cifar10_test_log_p, dim=0)
        print('Finished CamCAN lesional')

        # for batch in pbart1hm:
        #     if dataset == "camcan":
        #         mask = batch["mask"].to(device)  # first so you get the mask before changing the batch to the actual one
        #         seg = batch['seg'].to(device)
        #         batch = batch["scan"]
        #
        #
        #     # b_size = batch.size(0)
        #     # sample_size = 5
        #     # latent_z_dim = model.zdim
        #     real_batch = batch.to(device)
        #     # healthy = real_batch[torch.sum(seg, dim=(1, 2, 3)) == 0]
        #     # lesions = real_batch[torch.sum(seg, dim=(1, 2, 3)) != 0]
        #
        #     # # Forward pass to get all necessary variables for POSTERIOR DISTRIBUTION OF ENCODER
        #     # real_mu, real_logvar = model.encode(real_batch)
        #     # encoder_posterior = Independent(Normal(real_mu, torch.exp(0.5 * real_logvar)), 1)
        #     # z_sampled = encoder_posterior.sample([sample_size])
        #     #
        #     # # PRIOR DISTRIBUTION for LATENT SPACE
        #     # mu_o = torch.zeros(b_size, latent_z_dim, dtype=float).to(device)
        #     # var_o = torch.ones(b_size, latent_z_dim, dtype=float).to(device)
        #     # var_d = torch.ones([b_size, 3, 32, 32], dtype=float).to(device)
        #     # latent_prior = Independent(Normal(mu_o, var_o), 1)
        #     #
        #     # # Find PROBABILITIES from DISTRIBUTIONS above (They are log_prob) [5x32]
        #     # z_prob = latent_prior.log_prob(z_sampled)
        #     # # change type to float64 (cus otherwise you have division w/ 0.)
        #     # encoder_prob = encoder_posterior.log_prob(z_sampled).type(torch.float64)
        #     #
        #     # # for loop for the reconstruction loss to take each batched sample at once
        #     # # log_det = torch.log(((2 * np.pi) ** 16) * torch.ones(b_size)).to(device)
        #     # for i in range(sample_size):
        #     #     z = z_sampled[i]
        #     #     rec = model.decoder(z)
        #     #
        #     #     # DISTRIBUTION FOR DECODER LIKELIHOOD INSTEAD OF TAKING THE RECONSTRUCTION LOSS
        #     #     decoder_likelihood = Independent(Normal(rec.type(torch.float64), var_d), 3)
        #     #     decoder_likelihood_prob = decoder_likelihood.log_prob(real_batch.type(torch.float64))
        #     #
        #     #     # decoder_likelihood = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
        #     #     # decoder_likelihood_prob = - (decoder_likelihood / 2) - log_det
        #     #     # decoder_likelihood_prob = torch.exp(decoder_likelihood_prob)
        #     #     # decoder_likelihood.append(decoder_likelihood_prob)
        #     #
        #     #     alpha = z_prob[i] + decoder_likelihood_prob - encoder_prob[i]
        #     #     # alpha = decoder_likelihood_prob * torch.div(z_prob[i], encoder_prob[i])
        #     #     if i == 0:
        #     #         alphas = alpha
        #     #     else:
        #     #         alphas = torch.cat((alphas, alpha), 0)
        #     # alphas = alphas.reshape(-1, b_size)
        #     # # batch_likelihood = torch.log(sum(alphas))
        #     #
        #     # exponentials = torch.zeros(b_size).to(device)
        #     # for j in range(sample_size - 1):
        #     #     exponentials += torch.exp(alphas[j + 1] - alphas[0])
        #     # batch_likelihood = alphas[0] + torch.log(1 + exponentials)
        #     #
        #     # # Monte Carlo accumulation of likelihoods per batch
        #     # if first_batch:
        #     #     mc_likelihood = batch_likelihood
        #     # else:
        #     #     mc_likelihood = torch.cat((mc_likelihood, batch_likelihood))
        #
        #     mu, logvar, _, rec = model(real_batch)
        #     # decoder_loss = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
        #     _, _, loss_rec, _, decoder_loss = loss_function(rec, real_batch, mask, mu, logvar)
        #     loss_kl = calc_kl(logvar, mu, reduce="none")
        #     elbo = decoder_loss*beta_rec + loss_kl
        #     # log_p = estimate_log_p_x(model, real_batch, mu=mu, logvar=logvar, num_samples=5, x_sigma=1.0)
        #     t1_hm_healthy_log_p.append(elbo[torch.sum(seg, dim=(1, 2, 3)) == 0])
        #     t1_hm_lesion_log_p.append(elbo[torch.sum(seg, dim=(1, 2, 3)) != 0])
        #     first_batch += 1
        # t1_hm_healthy_log_p = torch.cat(t1_hm_healthy_log_p, dim=0)
        # t1_hm_lesion_log_p = torch.cat(t1_hm_lesion_log_p, dim=0)
        # print('Finished BraTs T1 + HM')
        #
        # for batch in pbart1:
        #     if dataset == "camcan":
        #         mask = batch["mask"].to(device)  # first so you get the mask before changing the batch to the actual one
        #         seg = batch['seg'].to(device)
        #         batch = batch["scan"]
        #
        #
        #     # b_size = batch.size(0)
        #     # sample_size = 5
        #     # latent_z_dim = model.zdim
        #     real_batch = batch.to(device)
        #     # healthy = real_batch[torch.sum(seg, dim=(1, 2, 3)) == 0]
        #     # lesions = real_batch[torch.sum(seg, dim=(1, 2, 3)) != 0]
        #
        #     # # Forward pass to get all necessary variables for POSTERIOR DISTRIBUTION OF ENCODER
        #     # real_mu, real_logvar = model.encode(real_batch)
        #     # encoder_posterior = Independent(Normal(real_mu, torch.exp(0.5 * real_logvar)), 1)
        #     # z_sampled = encoder_posterior.sample([sample_size])
        #     #
        #     # # PRIOR DISTRIBUTION for LATENT SPACE
        #     # mu_o = torch.zeros(b_size, latent_z_dim, dtype=float).to(device)
        #     # var_o = torch.ones(b_size, latent_z_dim, dtype=float).to(device)
        #     # var_d = torch.ones([b_size, 3, 32, 32], dtype=float).to(device)
        #     # latent_prior = Independent(Normal(mu_o, var_o), 1)
        #     #
        #     # # Find PROBABILITIES from DISTRIBUTIONS above (They are log_prob) [5x32]
        #     # z_prob = latent_prior.log_prob(z_sampled)
        #     # # change type to float64 (cus otherwise you have division w/ 0.)
        #     # encoder_prob = encoder_posterior.log_prob(z_sampled).type(torch.float64)
        #     #
        #     # # for loop for the reconstruction loss to take each batched sample at once
        #     # # log_det = torch.log(((2 * np.pi) ** 16) * torch.ones(b_size)).to(device)
        #     # for i in range(sample_size):
        #     #     z = z_sampled[i]
        #     #     rec = model.decoder(z)
        #     #
        #     #     # DISTRIBUTION FOR DECODER LIKELIHOOD INSTEAD OF TAKING THE RECONSTRUCTION LOSS
        #     #     decoder_likelihood = Independent(Normal(rec.type(torch.float64), var_d), 3)
        #     #     decoder_likelihood_prob = decoder_likelihood.log_prob(real_batch.type(torch.float64))
        #     #
        #     #     # decoder_likelihood = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
        #     #     # decoder_likelihood_prob = - (decoder_likelihood / 2) - log_det
        #     #     # decoder_likelihood_prob = torch.exp(decoder_likelihood_prob)
        #     #     # decoder_likelihood.append(decoder_likelihood_prob)
        #     #
        #     #     alpha = z_prob[i] + decoder_likelihood_prob - encoder_prob[i]
        #     #     # alpha = decoder_likelihood_prob * torch.div(z_prob[i], encoder_prob[i])
        #     #     if i == 0:
        #     #         alphas = alpha
        #     #     else:
        #     #         alphas = torch.cat((alphas, alpha), 0)
        #     # alphas = alphas.reshape(-1, b_size)
        #     # # batch_likelihood = torch.log(sum(alphas))
        #     #
        #     # exponentials = torch.zeros(b_size).to(device)
        #     # for j in range(sample_size - 1):
        #     #     exponentials += torch.exp(alphas[j + 1] - alphas[0])
        #     # batch_likelihood = alphas[0] + torch.log(1 + exponentials)
        #     #
        #     # # Monte Carlo accumulation of likelihoods per batch
        #     # if first_batch:
        #     #     mc_likelihood = batch_likelihood
        #     # else:
        #     #     mc_likelihood = torch.cat((mc_likelihood, batch_likelihood))
        #
        #     mu, logvar, _, rec = model(real_batch)
        #     # decoder_loss = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
        #     _, _, loss_rec, _, decoder_loss = loss_function(rec, real_batch, mask, mu, logvar)
        #     loss_kl = calc_kl(logvar, mu, reduce="none")
        #     elbo = decoder_loss*beta_rec + loss_kl
        #     # log_p = estimate_log_p_x(model, real_batch, mu=mu, logvar=logvar, num_samples=5, x_sigma=1.0)
        #     t1_healthy_log_p.append(elbo[torch.sum(seg, dim=(1, 2, 3)) == 0])
        #     t1_lesion_log_p.append(elbo[torch.sum(seg, dim=(1, 2, 3)) != 0])
        #     first_batch += 1
        # t1_healthy_log_p = torch.cat(t1_healthy_log_p, dim=0)
        # t1_lesion_log_p = torch.cat(t1_lesion_log_p, dim=0)
        # print('Finished BraTs T1')
        #
        # for batch in pbart2hm:
        #     if dataset == "camcan":
        #         mask = batch["mask"].to(device)  # first so you get the mask before changing the batch to the actual one
        #         seg = batch['seg'].to(device)
        #         batch = batch["scan"]
        #
        #
        #     # b_size = batch.size(0)
        #     # sample_size = 5
        #     # latent_z_dim = model.zdim
        #     real_batch = batch.to(device)
        #     # healthy = real_batch[torch.sum(seg, dim=(1, 2, 3)) == 0]
        #     # lesions = real_batch[torch.sum(seg, dim=(1, 2, 3)) != 0]
        #
        #     # # Forward pass to get all necessary variables for POSTERIOR DISTRIBUTION OF ENCODER
        #     # real_mu, real_logvar = model.encode(real_batch)
        #     # encoder_posterior = Independent(Normal(real_mu, torch.exp(0.5 * real_logvar)), 1)
        #     # z_sampled = encoder_posterior.sample([sample_size])
        #     #
        #     # # PRIOR DISTRIBUTION for LATENT SPACE
        #     # mu_o = torch.zeros(b_size, latent_z_dim, dtype=float).to(device)
        #     # var_o = torch.ones(b_size, latent_z_dim, dtype=float).to(device)
        #     # var_d = torch.ones([b_size, 3, 32, 32], dtype=float).to(device)
        #     # latent_prior = Independent(Normal(mu_o, var_o), 1)
        #     #
        #     # # Find PROBABILITIES from DISTRIBUTIONS above (They are log_prob) [5x32]
        #     # z_prob = latent_prior.log_prob(z_sampled)
        #     # # change type to float64 (cus otherwise you have division w/ 0.)
        #     # encoder_prob = encoder_posterior.log_prob(z_sampled).type(torch.float64)
        #     #
        #     # # for loop for the reconstruction loss to take each batched sample at once
        #     # # log_det = torch.log(((2 * np.pi) ** 16) * torch.ones(b_size)).to(device)
        #     # for i in range(sample_size):
        #     #     z = z_sampled[i]
        #     #     rec = model.decoder(z)
        #     #
        #     #     # DISTRIBUTION FOR DECODER LIKELIHOOD INSTEAD OF TAKING THE RECONSTRUCTION LOSS
        #     #     decoder_likelihood = Independent(Normal(rec.type(torch.float64), var_d), 3)
        #     #     decoder_likelihood_prob = decoder_likelihood.log_prob(real_batch.type(torch.float64))
        #     #
        #     #     # decoder_likelihood = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
        #     #     # decoder_likelihood_prob = - (decoder_likelihood / 2) - log_det
        #     #     # decoder_likelihood_prob = torch.exp(decoder_likelihood_prob)
        #     #     # decoder_likelihood.append(decoder_likelihood_prob)
        #     #
        #     #     alpha = z_prob[i] + decoder_likelihood_prob - encoder_prob[i]
        #     #     # alpha = decoder_likelihood_prob * torch.div(z_prob[i], encoder_prob[i])
        #     #     if i == 0:
        #     #         alphas = alpha
        #     #     else:
        #     #         alphas = torch.cat((alphas, alpha), 0)
        #     # alphas = alphas.reshape(-1, b_size)
        #     # # batch_likelihood = torch.log(sum(alphas))
        #     #
        #     # exponentials = torch.zeros(b_size).to(device)
        #     # for j in range(sample_size - 1):
        #     #     exponentials += torch.exp(alphas[j + 1] - alphas[0])
        #     # batch_likelihood = alphas[0] + torch.log(1 + exponentials)
        #     #
        #     # # Monte Carlo accumulation of likelihoods per batch
        #     # if first_batch:
        #     #     mc_likelihood = batch_likelihood
        #     # else:
        #     #     mc_likelihood = torch.cat((mc_likelihood, batch_likelihood))
        #
        #     mu, logvar, _, rec = model(real_batch)
        #     # decoder_loss = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
        #     _, _, loss_rec, _, decoder_loss = loss_function(rec, real_batch, mask, mu, logvar)
        #     loss_kl = calc_kl(logvar, mu, reduce="none")
        #     elbo = decoder_loss*beta_rec + loss_kl
        #     # log_p = estimate_log_p_x(model, real_batch, mu=mu, logvar=logvar, num_samples=5, x_sigma=1.0)
        #     t2_hm_healthy_log_p.append(elbo[torch.sum(seg, dim=(1, 2, 3)) == 0])
        #     t2_hm_lesion_log_p.append(elbo[torch.sum(seg, dim=(1, 2, 3)) != 0])
        #     first_batch += 1
        # t2_hm_healthy_log_p = torch.cat(t2_hm_healthy_log_p, dim=0)
        # t2_hm_lesion_log_p = torch.cat(t2_hm_lesion_log_p, dim=0)
        # print('Finished BraTs T2 + HM')

        for batch in pbart2:
            if dataset == "camcan":
                mask = batch["mask"].to(device)  # first so you get the mask before changing the batch to the actual one
                seg = batch['seg'].to(device)
                batch = batch["scan"]


            # b_size = batch.size(0)
            # sample_size = 5
            # latent_z_dim = model.zdim
            real_batch = batch.to(device)
            # healthy = real_batch[torch.sum(seg, dim=(1, 2, 3)) == 0]
            # lesions = real_batch[torch.sum(seg, dim=(1, 2, 3)) != 0]

            # # Forward pass to get all necessary variables for POSTERIOR DISTRIBUTION OF ENCODER
            # real_mu, real_logvar = model.encode(real_batch)
            # encoder_posterior = Independent(Normal(real_mu, torch.exp(0.5 * real_logvar)), 1)
            # z_sampled = encoder_posterior.sample([sample_size])
            #
            # # PRIOR DISTRIBUTION for LATENT SPACE
            # mu_o = torch.zeros(b_size, latent_z_dim, dtype=float).to(device)
            # var_o = torch.ones(b_size, latent_z_dim, dtype=float).to(device)
            # var_d = torch.ones([b_size, 3, 32, 32], dtype=float).to(device)
            # latent_prior = Independent(Normal(mu_o, var_o), 1)
            #
            # # Find PROBABILITIES from DISTRIBUTIONS above (They are log_prob) [5x32]
            # z_prob = latent_prior.log_prob(z_sampled)
            # # change type to float64 (cus otherwise you have division w/ 0.)
            # encoder_prob = encoder_posterior.log_prob(z_sampled).type(torch.float64)
            #
            # # for loop for the reconstruction loss to take each batched sample at once
            # # log_det = torch.log(((2 * np.pi) ** 16) * torch.ones(b_size)).to(device)
            # for i in range(sample_size):
            #     z = z_sampled[i]
            #     rec = model.decoder(z)
            #
            #     # DISTRIBUTION FOR DECODER LIKELIHOOD INSTEAD OF TAKING THE RECONSTRUCTION LOSS
            #     decoder_likelihood = Independent(Normal(rec.type(torch.float64), var_d), 3)
            #     decoder_likelihood_prob = decoder_likelihood.log_prob(real_batch.type(torch.float64))
            #
            #     # decoder_likelihood = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
            #     # decoder_likelihood_prob = - (decoder_likelihood / 2) - log_det
            #     # decoder_likelihood_prob = torch.exp(decoder_likelihood_prob)
            #     # decoder_likelihood.append(decoder_likelihood_prob)
            #
            #     alpha = z_prob[i] + decoder_likelihood_prob - encoder_prob[i]
            #     # alpha = decoder_likelihood_prob * torch.div(z_prob[i], encoder_prob[i])
            #     if i == 0:
            #         alphas = alpha
            #     else:
            #         alphas = torch.cat((alphas, alpha), 0)
            # alphas = alphas.reshape(-1, b_size)
            # # batch_likelihood = torch.log(sum(alphas))
            #
            # exponentials = torch.zeros(b_size).to(device)
            # for j in range(sample_size - 1):
            #     exponentials += torch.exp(alphas[j + 1] - alphas[0])
            # batch_likelihood = alphas[0] + torch.log(1 + exponentials)
            #
            # # Monte Carlo accumulation of likelihoods per batch
            # if first_batch:
            #     mc_likelihood = batch_likelihood
            # else:
            #     mc_likelihood = torch.cat((mc_likelihood, batch_likelihood))

            mu, logvar, _, rec = model(real_batch)
            # decoder_loss = calc_reconstruction_loss(real_batch, rec, loss_type='mse')
            _, _, loss_rec, _, decoder_loss = loss_function(rec, real_batch, mask, mu, logvar)
            loss_kl = calc_kl(logvar, mu, reduce="none")
            elbo = decoder_loss*beta_rec + loss_kl
            # log_p = estimate_log_p_x(model, real_batch, mu=mu, logvar=logvar, num_samples=5, x_sigma=1.0)
            t2_healthy_log_p.append(elbo[torch.sum(seg, dim=(1, 2, 3)) == 0])
            t2_lesion_log_p.append(elbo[torch.sum(seg, dim=(1, 2, 3)) != 0])
            first_batch += 1
        t2_healthy_log_p = torch.cat(t2_healthy_log_p, dim=0)
        t2_lesion_log_p = torch.cat(t2_lesion_log_p, dim=0)
        print('Finished BraTs T2')

        # writer.add_scalar("Loss_rec/validation", sum(cifar10_test_log_p)/len(cifar10_test_log_p), first_batch * 20)


            # mc_likelihood = decoder_likelihood + z_prob - encoder_prob
            # log_likelihood = sum(mc_likelihood, 1)

    # np.savetxt(f"./cifar10_likelihood/simple_data_{dataset}_epoch_250_gaussian_decoder.txt", mc_likelihood.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_{dataset}_{model_epoch}.txt", cifar10_train_log_p.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_{dataset}_gauss_{model_epoch}.txt", cifar10_test_log_p.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_mnist_test_{model_epoch}.txt", mnist_test_log_p.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_t1_hm_healthy_{model_epoch}.txt", t1_hm_healthy_log_p.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_t1_hm_lesion_{model_epoch}.txt", t1_hm_lesion_log_p.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_t1_healthy_{model_epoch}.txt", t1_healthy_log_p.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_t1_lesion_{model_epoch}.txt", t1_lesion_log_p.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_t2_hm_healthy_{model_epoch}.txt", t2_hm_healthy_log_p.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_t2_hm_lesion_{model_epoch}.txt", t2_hm_lesion_log_p.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_t2_healthy_{model_epoch}.txt", t2_healthy_log_p.cpu().numpy())
    np.savetxt(f"./10_3_20220/likelihoods/3000/weighted_elbo_data_t2_lesion_{model_epoch}.txt", t2_lesion_log_p.cpu().numpy())

    # with open(r"./data.txt", 'a') as f:
    #     f.write(str(mc_likelihood))

    # os.makedirs("./ood_figures", exist_ok=True)
    # os.mkdir("/run/user/519383/ood_figures", 0o700)
    # fig = plt.figure(figsize=(8, 6))
    # plt.hist(mc_likelihood.cpu(), bins=10)
    # plt.savefig(r'./ood_figures/cifar_histogram.png')
    # plt.show()


if __name__ == '__main__':
    # for files in os.listdir("./camcan_soft_intro_betas_1.0_256_1.0"):
    #     if files.endswith(".pth"):
    #         pretrained = files
    pretrained = "./10_3_20220/camcan_bv3.5_maskedrec_soft_intro_betas_1.0_256_2500.0/model_epoch_108_iter_64260.pth"
    # pretrained = "./10_3_20220/camcan_bv3.5_maskedrec_soft_intro_betas_1.0_256_3000.0/model_epoch_51_iter_30345.pth"
    print(pretrained)
    eval_soft_intro_vae(dataset='camcan', z_dim=128, batch_size=32, num_workers=4,
                                pretrained=pretrained,
                                device=torch.device("cuda"))
        # else:
        #     pass
