"""
    dataset_type = 1 => voc dataset
    dataset_type = 2 => kitti dataset
"""
dataset_type = 1
epochs = 100
CUDA_DEVICES = '6'

voc_batch_size = 12
kitti_batch_size = 32


# voc_root_dir = '/home/CN/zhenglin.zhou/Documents/VOC/VOCdevkit/'
# kitti_root_dir = '/home/CN/zhenglin.zhou/Documents/Kitti/training/'

voc_root_dir = 'D:/VOC/VOCdevkit'
kitti_root_dir = 'D:/KITTI/training'

"""    Mixup    """
use_mixup = 0
alpha = 1

"""    Autoaugment     """
use_autoaugment = 0

"""   Mixup   """

GRID_GRID = False
GRID_ROTATE = 1
GRID_OFFSET = 0
GRID_RATIO = 0.5
GRID_MODE = 1
GRID_PROB = 0.5
