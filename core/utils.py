import torch
import torch.nn.functional as F

from typing import Union

def otsu_thresholding(tensor : torch.Tensor, nbins : int = 20, weight : Union[torch.Tensor, None] = None):

    """

    Args:
        tensor: (torch.Tensor) shape [N x D] of N batches of D scalar data points
        nbins : (int) number of bins
    Returns:

    """

    if weight is None:
        weight = tensor.new_ones(tensor.shape)
    else:
        assert weight.shape == tensor.shape

    mu = torch.sum(tensor * weight, dim=1, keepdim=True) / (torch.sum(weight, dim=1, keepdim=True) + 1e-4)

    tmax, _ = torch.max(tensor - (1 - weight) * 1e5, dim=1, keepdim=True)
    tmin, _ = torch.min(tensor + (1 - weight) * 1e5, dim=1, keepdim=True)
    grid_size = (tmax - tmin) / nbins

    base_grid = torch.linspace(0, nbins - 1, nbins).cuda().unsqueeze(0)
    lattice = grid_size * base_grid + tmin

    cluster = (tensor.unsqueeze(2) <= lattice.unsqueeze(1)).float() # N x D x B
    weight = weight.unsqueeze(2)

    dat = tensor.unsqueeze(2)
    less = torch.sum(cluster * dat * weight, dim=1) / (torch.sum(cluster * weight, dim=1) + 1e-4)
    more = torch.sum((1 - cluster) * dat * weight, dim=1) / (torch.sum((1 - cluster) * weight, dim=1) + 1e-4)

    sigma = torch.sum(cluster * weight, dim=1) * (less - mu) ** 2 + torch.sum((1 - cluster) * weight, dim=1) * (more - mu) ** 2

    _, index = torch.max(sigma, dim=1)

    threshold = torch.gather(lattice, dim=1, index=index.unsqueeze(1))

    return threshold


