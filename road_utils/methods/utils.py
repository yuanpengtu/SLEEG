from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from zipfile import ZipFile
from PIL import Image

import wget
import gdown
import tarfile
import yaml

from .image_segmentation.network.deepv3 import DeepWV3Plus
from .image_segmentation.network.mynn import Norm2d
from .image_dissimilarity.models.dissimilarity_model import DissimNetPrior, DissimNet
from .image_synthesis.models.pix2pix_model import Pix2PixModel
from .image_segmentation.optimizer import restore_snapshot
from .image_dissimilarity.models.vgg_features import VGG19_difference


def init_pytorch_DeepWV3Plus(ckpt_path=None, num_classes=19):
    print("Load PyTorch model", end="", flush=True)
    torch.cuda.empty_cache()
    network = nn.DataParallel(DeepWV3Plus(num_classes))
    print("... ok")
    if ckpt_path is not None:
        print("Checkpoint file: %s" % ckpt_path, end="", flush=True)
        network.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
        print("... ok\n")
    network = network.cuda().eval()
    return network


def download_checkpoint(url, save_dir):
    print("Download PyTorch checkpoint")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = wget.download(url, out=str(save_dir))
    return filename


def download_tar(url, save_dir):
    print("Download .tar and de-compress")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = wget.download(url, out=str(save_dir))
    with tarfile.open(Path(save_dir) / filename, 'r') as tar_file:
        tar_file.extractall(save_dir)
    return filename


def download_zip(url, save_dir):
    print("Download .zip and de-compress")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    filename = wget.download(url, out=str(save_dir))
    with ZipFile(Path(save_dir) / filename, 'r') as zip_ref:
        zip_ref.extractall(save_dir)
    return filename


def load_gdrive_file(file_id, save_dir, ending='zip'):
    print("Downloads files from google drive, caches files that are already downloaded.")
    filename = '{}.{}'.format(file_id, ending) if ending else file_id
    filename = os.path.join(os.path.expanduser('~/.keras/datasets'), filename)
    if not os.path.exists(filename):
        gdown.download('https://drive.google.com/uc?id={}'.format(file_id), filename, quiet=False)
    ZipFile(filename).extractall(save_dir)
    return filename


def get_softmax(network, image, transform=None, as_numpy=True):
    if transform is None:
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = transform(image)
    x = x.unsqueeze_(0).cuda()
    with torch.no_grad():
        y = network(x)
    probs = F.softmax(y, 1)
    if as_numpy:
        probs = probs.data.cpu().numpy()[0].astype("float32")
    return probs


def get_entropy(network, image, transform=None, as_numpy=True):
    probs = get_softmax(network, image, transform, as_numpy=False)
    entropy = torch.div(torch.sum(-probs * torch.log(probs), dim=1), torch.log(torch.tensor(probs.shape[1])))
    if as_numpy:
        entropy = entropy.data.cpu().numpy()[0].astype("float32")
    return entropy

def get_calibrated_softmax(network, image, magnitude, temperature, transform=None, as_numpy=False):
    if transform is None:
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = transform(image)
    x = x.unsqueeze_(0).cuda()
    p = get_gradient_wrt_input(network, image, transform)
    x = torch.sub(x, p, alpha=magnitude)
    with torch.no_grad():
        y = network(x)
    y = y / temperature
    probs = F.softmax(y, 1)
    if as_numpy:
        probs = probs.data.cpu().numpy()[0].astype("float32")
    return probs


def get_gradient_wrt_input(network, image, transform=None, as_numpy=False):
    if transform is None:
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = transform(image)
    x = x.unsqueeze_(0).requires_grad_().cuda()
    y = network(x)
    softmax = nn.Softmax(dim=1)(y)
    pred = torch.argmax(softmax, dim=1).detach()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(softmax, pred.cuda())
    loss.backward(retain_graph=True)
    grad = -torch.autograd.grad(outputs=loss, inputs=x, retain_graph=True)[0]
    grad = torch.sign(grad)
    if as_numpy:
        grad = grad.cpu().detach().numpy()[0].astype("float32")
    return grad


def mahalanobis_modification(network, ckpt_path):
    network.module.final = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True))
    print("Checkpoint file: %s" % ckpt_path, end="", flush=True)
    network.load_state_dict(torch.load(ckpt_path)['state_dict'], strict=False)
    print("... ok\n")
    return network.cuda().eval()


def get_activations(network, image, transform=None, as_numpy=True):
    if transform is None:
        transform = Compose([ToTensor(), Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x = transform(image)
    x = x.unsqueeze_(0).cuda()
    with torch.no_grad():
        y = network(x)
    if as_numpy:
        y = y.data.cpu().numpy()[0].astype("float32")
    return y


def get_segmentation(checkpoint_dir, opt):
    net = init_pytorch_DeepWV3Plus()
    print('Segmentation Net Built.')
    snapshot = os.path.join(checkpoint_dir, opt.snapshot)
    segmentation_net, _ = restore_snapshot(net, optimizer=None, snapshot=snapshot, restore_optimizer_bool=False)
    segmentation_net.eval()
    print('Segmentation Net Restored.')
    return segmentation_net


def get_synthesis(checkpoints_dir, opt):
    # Get Synthesis Net
    print('Synthesis Net Built.')
    opt.checkpoints_dir = checkpoints_dir
    synthesis_net = Pix2PixModel(opt)
    synthesis_net.eval()
    print('Synthesis Net Restored')
    return synthesis_net


def get_dissimilarity(checkpoint_dir, ours=True):
    # Get Dissimilarity Net
    if ours:
        config_diss = os.path.join(os.getcwd(), os.path.dirname(__file__),
                                   'image_dissimilarity/configs/test/ours_configuration.yaml')
    else:
        config_diss = os.path.join(os.getcwd(), os.path.dirname(__file__),
                                   'image_dissimilarity/configs/test/baseline_configuration.yaml')

    with open(config_diss, 'r') as stream:
        config_diss = yaml.load(stream, Loader=yaml.FullLoader)

    prior = config_diss['model']['prior']
    ensemble = config_diss['ensemble']

    if prior:
        diss_model = DissimNetPrior(**config_diss['model']).cuda()
    else:
        diss_model = DissimNet(**config_diss['model']).cuda()

    print('Dissimilarity Net Built.')
    save_folder = os.path.join(checkpoint_dir, config_diss['save_folder'])
    model_path = os.path.join(save_folder, '%s_net_%s.pth' % (config_diss['which_epoch'], config_diss['experiment_name']))
    model_weights = torch.load(model_path)
    diss_model.load_state_dict(model_weights)
    diss_model.eval()
    print('Dissimilarity Net Restored')
    return diss_model, prior, ensemble


def get_synboost_transformations():
    # Transform images to Tensor based on ImageNet Mean and STD
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    img_transform = Compose([ToTensor(), Normalize(*mean_std)])

    # synthesis necessary pre-process
    transform_semantic = Compose([Resize(size=(256, 512), interpolation=Image.NEAREST), ToTensor()])
    transform_image_syn = Compose([Resize(size=(256, 512), interpolation=Image.BICUBIC), ToTensor(),
                                   Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # dissimilarity pre-process
    vgg_diff = VGG19_difference().cuda()
    base_transforms_diss = Compose([Resize(size=(256, 512), interpolation=Image.NEAREST), ToTensor()])
    norm_transform_diss = Compose([Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # imageNet normalization

    return img_transform, transform_semantic, transform_image_syn, vgg_diff, base_transforms_diss, norm_transform_diss