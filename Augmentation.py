import config
import numpy as np
from augmentation_zoo.autoaugment_utils import distort_image_with_autoaugment
from picture_visualization import easy_visualization
import torch

class retinanet_augmentater(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, flip_x=0.5):
        self.flip_x = flip_x

    def __call__(self, sample):
        if np.random.rand() < self.flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample

class autoaugmenter(object):
    """Applies the AutoAugment policy to `image` and `bboxes`.
    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.
      bboxes: `Tensor` of shape [N, 4] representing ground truth boxes that are
        normalized between [0, 1].
      augmentation_name: The name of the AutoAugment policy to use. The available
        options are `v0`, `v1`, `v2`, `v3` and `test`. `v0` is the policy used for
        all of the results in the paper and was found to achieve the best results
        on the COCO dataset. `v1`, `v2` and `v3` are additional good policies
        found on the COCO dataset that have slight variation in what operations
        were used during the search procedure along with how many operations are
        applied in parallel to a single image (2 vs 3).
    Returns:
      A tuple containing the augmented versions of `image` and `bboxes`.
    """
    def __init__(self, augmentation_name = 'v2'):
        self.augmentation_name = augmentation_name

    def normalizer(self, image, annots):
        h, w = image.shape[0], image.shape[1]

        ratio = np.array([w, h, w, h], dtype=int)
        annots[:, 0:4] = annots[:, 0:4] / ratio

        copy_annots = annots.copy()
        annots[:,0], annots[:,1] = copy_annots[:,1], copy_annots[:,0]
        annots[:,2], annots[:,3] = copy_annots[:,3], copy_annots[:,2]
        return annots

    def unnormalizer(self, image, annots):
        h, w = image.shape[0], image.shape[1]

        ratio = np.array([w, h, w, h], dtype=int)
        annots[:, 0:4] = annots[:, 0:4] * ratio

        copy_annots = annots.copy()
        annots[:,0], annots[:,1] = copy_annots[:,1], copy_annots[:,0]
        annots[:,2], annots[:,3] = copy_annots[:,3], copy_annots[:,2]

        return annots.astype(int)

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        print(annots)
        easy_visualization(sample)
        annots = self.normalizer(image, annots)
        bboxes = annots[:, 0:4]
        print(annots)
        image, bboxes = distort_image_with_autoaugment(image, bboxes, self.augmentation_name)


        annots[:, 0:4] = bboxes
        annots = self.unnormalizer(image, annots)


        print(annots)
        sample = {'img': image, 'annot': annots}
        return sample



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
    lam = torch.Tensor(lam)
    cls_loss = torch.mul(cls_loss, lam)
    reg_loss = torch.mul(reg_loss, lam)
    cls_loss = _mix_loss(cls_loss, batch_size)
    reg_loss = _mix_loss(reg_loss, batch_size)
    return cls_loss, reg_loss



class Augmenter(object):

    def __call__(self, sample):
        # sample = retinanet_augmentater(sample)
        # sample = autoaugmenter(sample)
        return sample