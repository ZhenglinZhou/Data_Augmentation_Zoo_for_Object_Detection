from prepare_data import VocDataset, KittiDataset, Normalizer, Resizer
from tools import SplitKittiDataset
from Augmentation import autoaugmenter, retinanet_augmentater
from config import voc_root_dir, kitti_root_dir
from torchvision import transforms

if __name__ == '__main__':
    voc_train = VocDataset(voc_root_dir, 'train',
                           transform=transforms.Compose([autoaugmenter('test')]))
    # for i in range(voc_train.__len__()):
    #     print(i)
    #     sample = voc_train[i]
    sample = voc_train[11]
    # SplitKittiDataset(kitti_root_dir, 0.5)  # 分割KITTI数据集，50%训练集，50%测试集
    #
    # kitti_train = KittiDataset(kitti_root_dir, 'train', transform=transforms.Compose([autoaugmenter('test')]))

    # sample = kitti_train[2]
    # for i in range():
    #     sample = kitti_train[i]


