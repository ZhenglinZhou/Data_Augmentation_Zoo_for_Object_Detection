from prepare_data import VocDataset, KittiDataset, Normalizer, Resizer
from tools import SplitKittiDataset
from config import VOC_ROOT_DIR, KITTI_ROOT_DIR
from torchvision import transforms
from tools import easy_visualization
from augmentation_zoo.MyGridMask import GridMask
import config as cfg
from tools import easy_visualization
from augmentation_zoo.SmallObjectAugmentation import SmallObjectAugmentation
from augmentation_zoo.Myautoaugment_utils import AutoAugmenter
from augmentation_zoo.RandomFlip import RandomFlip

def _make_transform():
    transform_list = list()
    if cfg.AUTOAUGMENT:
        transform_list.append(AutoAugmenter(cfg.AUTO_POLICY))
    if cfg.GRID:
        transform_list.append(GridMask(True, True, cfg.GRID_ROTATE,cfg.GRID_OFFSET,cfg.GRID_RATIO,cfg.GRID_MODE,cfg.GRID_PROB))
    if cfg.RANDOM_FLIP:
        transform_list.append(RandomFlip())
    if cfg.SMALL_OBJECT_AUGMENTATION:
        transform_list.append(SmallObjectAugmentation(cfg.SOA_THRESH, cfg.SOA_PROB, cfg.SOA_COPY_TIMES, cfg.SOA_EPOCHS, cfg.SOA_ALL_OBJECTS, cfg.SOA_ONE_OBJECT))
    return transform_list

if __name__ == '__main__':
    transform_list = _make_transform()
    voc_train = VocDataset(VOC_ROOT_DIR, 'train', transform=transforms.Compose(transform_list))
    kitti_train = KittiDataset(KITTI_ROOT_DIR, 'train', transforms.Compose(transform_list))

    # for i in range(voc_train.__len__()):
    #     print(i)
    #     sample = voc_train[i]
    #     easy_visualization(sample)

    sample = voc_train[5]
    easy_visualization(sample)

    # for i in range(kitti_train.__len__()):
    #     sample = kitti_train[i]
    #     easy_visualization(sample)

    # sample = kitti_train[0]
    # easy_visualization(sample)


