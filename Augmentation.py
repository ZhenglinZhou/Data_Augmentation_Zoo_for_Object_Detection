import numpy as np
from augmentation_zoo.autoaugment_utils import distort_image_with_autoaugment
from picture_visualization import easy_visualization

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
