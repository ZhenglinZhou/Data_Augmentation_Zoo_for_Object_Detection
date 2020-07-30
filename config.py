"""
    dataset_type = 1 => voc dataset
    dataset_type = 2 => kitti dataset
"""
dataset_type = 2
epochs = 100
CUDA_DEVICES = '6'

kitti_batch_size = 24
voc_batch_size = 8

# voc_root_dir = '/home/CN/zhenglin.zhou/Documents/VOC/VOCdevkit/'
# kitti_root_dir = '/home/CN/zhenglin.zhou/Documents/Kitti/training/'

voc_root_dir = 'D:\VOC\VOCdevkit'
kitti_root_dir = 'D:/KITTI/training'

"""    Mixup    """
use_mixup = 0
alpha = 1

"""    Autoaugment     """
use_autoaugment = 1
