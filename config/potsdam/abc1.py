from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.losses.useful_loss import ABCLoss
from geoseg.datasets.vaihingen_dataset import *
# from geoseg.models.MyUnet import MyUNet
from geoseg.models.ABCNet import ABCNet
from catalyst.contrib.nn import Lookahead
from catalyst import utils
# 0 v3
# training hparam
max_epoch = 70
ignore_index = len(CLASSES)
train_batch_size = 8
val_batch_size = 4
lr = 3e-4
weight_decay = 2.5e-4
backbone_lr = 6e-5
backbone_weight_decay = 2.5e-4
accumulate_n = 1  # accumulate gradients of n batches
num_classes = len(CLASSES)
classes = CLASSES
weights_name = "abc222-newww-e70"
weights_path = "model_weights/potsdam/{}".format(weights_name)
test_weights_name = "abc222-newww-e70-v1"
log_name = 'potsdam/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = False
check_val_every_n_epoch = 1
gpus = [0]
strategy = None
pretrained_ckpt_path = None
# pretrained_ckpt_path = '/home/caoyiwen/slns/NewNet/model_weights/potsdam/be1-pre-e100/be1-pre-e100-v1.ckpt'
resume_ckpt_path = None
# define the network
# net = BESNet(nclass=num_classes, aux=False, pretrained=True)
net = ABCNet(n_classes=num_classes)

# define the loss 两函数加权和
loss = ABCLoss(ignore_index=ignore_index)
# loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index), DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
# loss = JointLoss(CriterionDSN(), BoundaryLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)

use_aux_loss = True

# define the dataloader
def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=len(CLASSES), nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [albu.Normalize()]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


train_dataset = VaihingenDataset(data_root='/home/caoyiwen/data/potsdam_new/train', mode='train',
                                 img_dir='images_1024', mask_dir='masks_1024', img_size=(1024, 1024),
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(data_root='/home/caoyiwen/data/potsdam_new/val', img_dir='images_1024', mask_dir='masks_1024', img_size=(1024, 1024),
                               transform=val_aug)
test_dataset = VaihingenDataset(data_root='/home/caoyiwen/data/potsdam_new/test', img_dir='images_1024', mask_dir='masks_1024', img_size=(1024, 1024),
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=2,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=2,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
# lr_scheduler = PolyLRScheduler(optimizer, t_initial=max_epoch, power=0.9, lr_min=1e-6,
#                                warmup_t=5, warmup_lr_init=1e-6)
