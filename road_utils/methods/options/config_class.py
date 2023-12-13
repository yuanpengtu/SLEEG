
class Config():
    aspect_ratio = 2.0
    checkpoints_dir = 'models'
    contain_dontcare_label = False
    crop_size = 512
    dataset_mode = 'cityscapes'
    gpu = 0
    init_type = 'xavier'
    init_variance = 0.02
    isTrain = False
    label_nc = 35
    mpdist = False
    name = 'image-synthesis'
    netG = 'condconv'
    ngf = 64
    no_instance = False
    norm_E = 'spectralinstance'
    norm_G = 'spectralsync_batch'
    semantic_nc = 36
    snapshot = 'image-segmentation/cityscapes_best.pth'
    use_vae = True
    which_epoch = 'latest'
    z_dim = 256
