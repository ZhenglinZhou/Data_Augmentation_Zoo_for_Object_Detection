from prepare_data import VocDataset, KittiDataset, Normalizer, Resizer
from tools import SplitKittiDataset
from Augmentation import autoaugmenter, retinanet_augmentater
from config import voc_root_dir, kitti_root_dir
from picture_visualization import visualization, easy_visualization
from torchvision import transforms

if __name__ == '__main__':
    voc_train = VocDataset(voc_root_dir, 'train',
                           transform=transforms.Compose([autoaugmenter('test')]))
    sample = voc_train[2]
    visualization(voc_train, sample)
    # sample = autoaugment(sample, 'v2')
    # visualization(voc_train, sample)

