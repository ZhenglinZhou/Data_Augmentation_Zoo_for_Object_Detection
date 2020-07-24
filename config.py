"""
    dataset_type = 1 => voc dataset
    dataset_type = 2 => kitti dataset
"""
dataset_type = 1
epochs = 100

voc_root_dir = 'D:\VOC\VOCdevkit'
voc_batch_size = 64

kitti_root_dir = 'D:/KITTI/training'
kitti_batch_size = 2


"""
    Mixup
"""
use_mixup = 1
alpha = 1