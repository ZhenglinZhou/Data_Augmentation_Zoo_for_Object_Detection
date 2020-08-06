import config
import numpy as np
from tools import easy_visualization
import torch



def _mixup(img1, img2):
    """
    input
        img = [3, W, H]
    output
        img = [3, W, H]
    """
    alpha = config.alpha
    lam = np.random.beta(alpha, alpha)

    height = max(img1.shape[1], img2.shape[1])
    width = max(img1.shape[2], img2.shape[2])
    mix_img = np.zeros(shape=(3, height, width), dtype=np.float32)
    mix_img[:, :img1.shape[1], :img1.shape[2]] = img1.astype('float32') * lam
    mix_img[:, :img2.shape[1], :img2.shape[2]] += img2.astype('float32') * (1 - lam)
    mix_img = mix_img.astype('int8')
    return mix_img, lam

def mixup(data):
    """
    input
        data = dict['img': img, 'annot': annot]
    """
    img = data['img'].numpy()
    annot = data['annot']
    batch_size = len(img)
    new_img = []
    lam = []
    """
        order = [(0,1), (2,3), (4,5), ...]
        so that img[0] is same as img[1], 
        img[2] is same as img[3], ..., 
        which is the same mix img    
    """
    order = list(range(batch_size))
    order = [[order[x % len(order)] for x in range(i, i + 2)] for i in
             range(0, len(order), 2)]
    for _order in order:
        img1, img2 = img[_order[0]], img[_order[1]]
        mix_img, _lam = _mixup(img1, img2)
        def permute_numpy(img):
            print(img.shape)
            img = torch.Tensor(img).permute(1, 2, 0)
            return img.numpy()
        easy_visualization({'img': permute_numpy(img1), 'annot': annot[_order[0]]})
        easy_visualization({'img':permute_numpy(mix_img), 'annot': annot[_order[0]]})
        easy_visualization({'img': permute_numpy(img2), 'annot': annot[_order[1]]})
        easy_visualization({'img':permute_numpy(mix_img), 'annot': annot[_order[1]]})
        for i in range(2):
            new_img.append(mix_img)
        lam.append(_lam)
        lam.append(1 - _lam)
    new_data = {'img': torch.Tensor(new_img), 'annot': annot}
    return new_data, lam

def _mix_loss(loss, batch_size):
    new_loss = torch.stack([loss[i] + loss[i+1] for i in range(0, batch_size, 2)])
    return new_loss

def mix_loss(cls_loss, reg_loss, lam):
    """
    :param loss: len(cls_loss) = len(reg_loss) = batch_size
    :param lam:  len(lam) = batch_size
    :return: mix_loss
    """
    batch_size = len(lam)
    if torch.cuda.is_available():
        lam = torch.Tensor(lam).cuda().float()
    else:
        lam = torch.Tensor(lam).float()
    cls_loss = torch.mul(cls_loss, lam)
    reg_loss = torch.mul(reg_loss, lam)
    cls_loss = _mix_loss(cls_loss, batch_size)
    reg_loss = _mix_loss(reg_loss, batch_size)
    return cls_loss, reg_loss
