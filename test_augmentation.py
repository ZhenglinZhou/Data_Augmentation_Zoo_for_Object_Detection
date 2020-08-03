from prepare_data import VocDataset, KittiDataset, Normalizer, Resizer
from tools import SplitKittiDataset
from Augmentation import autoaugmenter, retinanet_augmentater
from config import VOC_ROOT_DIR, KITTI_ROOT_DIR
from torchvision import transforms
from tools import easy_visualization
from augmentation_zoo.grid_official import Grid
import config as cfg
from tools import easy_visualization
from augmentation_zoo.small_object_augmentation import copysmallobjects

if __name__ == '__main__':
    # voc_train = VocDataset(VOC_ROOT_DIR, 'train')

    # for i in range(voc_train.__len__()):
    #     print(i)
    #     sample = voc_train[i]
    #     easy_visualization(sample)
    # sample = voc_train[11]
    # print(sample['annot'])
    # SplitKittiDataset(kitti_root_dir, 0.5)  # 分割KITTI数据集，50%训练集，50%测试集
    #
    kitti_train = KittiDataset(KITTI_ROOT_DIR, 'train')
    sample = kitti_train[2]
    # easy_visualization(sample)
    # for i in range(kitti_train.__len__()):
    #     sample = kitti_train[i]
    #     easy_visualization(sample)
    cs = copysmallobjects()
    sample = cs(sample)
    easy_visualization(sample)


