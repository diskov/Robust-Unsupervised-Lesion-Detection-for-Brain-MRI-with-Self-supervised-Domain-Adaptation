import torch
import pytorch_lightning as pl
from uncertify.uncertify.data.datasets import CamCanHDF5Dataset, Brats2017HDF5Dataset
from uncertify.uncertify.models.encoder_decoder_baur2020 import BaurEncoder, BaurDecoder
from uncertify.uncertify.visualization.grid import imshow_grid


#  Make a full model with the encoder and decoder given above


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

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = reparameterize(mu, logvar)
        y = self.decode(z)

        return mu, logvar, z, y


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


def loss_function(self, reconstruction: Tensor, observation: Tensor, mask: Tensor, mu: Tensor, log_var: Tensor,
                  beta: float = 1.0, train_step: int = None):
    masking_enabled = mask is not None
    mask = mask if masking_enabled else torch.ones_like(observation, dtype=torch.bool)

    # Reconstruction Error
    slice_rec_err = torch_functional.l1_loss(observation * mask, reconstruction * mask, reduction='none')
    slice_rec_err = torch.sum(slice_rec_err, dim=(1, 2, 3))
    slice_wise_non_empty = torch.sum(mask, dim=(1, 2, 3))
    slice_wise_non_empty[slice_wise_non_empty == 0] = 1  # for empty masks
    slice_rec_err /= slice_wise_non_empty
    batch_rec_err = torch.mean(slice_rec_err)

    # KL Divergence
    slice_kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    batch_kl_div = torch.mean(slice_kl_div)
    kld_loss = batch_kl_div * self._calculate_beta(self._train_step_counter)

    # Total Loss
    total_loss = batch_rec_err + kld_loss

    return total_loss, batch_kl_div, batch_rec_err, slice_kl_div, slice_rec_err


#  Dataloaders and transofrms and preprocessing

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


BRATS_CAMCAN_DEFAULT_TRANSFORM = torchvision.transforms.Compose([
    NumpyReshapeTransform((200, 200)),
    Numpy2PILTransform(),
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor()])
transform = BRATS_CAMCAN_DEFAULT_TRANSFORM

camcan_t2_val_path: Path = "/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/small" \
                           "/camcan_val_t2_hm_std_bv3.5_xe.hdf5 "
camcan_t2_train_path: Path = '/itet-stor/kskovolas/bmicdatasets_bmicnas01/Processed/Matthaeus/datasets/small' \
                             '/camcan_train_t2_hm_std_bv3.5_xe.hdf5 '

train_dataloader, val_dataloader = camcan_data_loader(train_set_path, val_set_path, batch_size, shuffle_train,
                                                      shuffle_val, num_workers, transform, uppercase_keys,
                                                      add_gauss_blobs)


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
        grid = torchvision.utils.make_grid(stacked, padding=0, normalize=False, nrow=nrow)
        # describe = scipy.stats.describe(grid.numpy().flatten())
        # print_scipy_stats_description(describe, 'normalized_grid')
        fig, ax = imshow_grid(grid, one_channel=True, vmax=1.0, vmin=0.0, plt_show=False, **kwargs)
        ax.set_axis_off()
        plt.show()
        if save_dir_path is not None:
            img_file_name = 'batch_1.png'
            fig.save(fig, save_dir_path / img_file_name)
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
