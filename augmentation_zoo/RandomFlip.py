import numpy as np

class RandomFlip(object):
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