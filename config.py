"""
    DATASET_TYPE = 1 => voc dataset
    DATASET_TYPE = 2 => kitti dataset
"""
DATASET_TYPE = 1
EPOCHS = 100
CUDA_DEVICES = '6'

VOC_BATCH_SIZE = 8
KITTI_BATCH_SIZE = 24


# VOC_ROOT_DIR = '/home/CN/zhenglin.zhou/Documents/VOC/VOCdevkit/'
# KITTI_ROOT_DIR = '/home/CN/zhenglin.zhou/Documents/Kitti/training/'

VOC_ROOT_DIR = 'D:/VOC/VOCdevkit'
KITTI_ROOT_DIR = 'D:/KITTI/training'

"""   RANDOM FLIP   """
RANDOM_FLIP = False

"""    AUTOAUGMENT    """
AUTOAUGMENT = False
AUTO_POLICY = 'v4'

"""   GRIDMASK   """
GRID = False
GRID_ROTATE = 1
GRID_OFFSET = 0
GRID_RATIO = 0.5
GRID_MODE = 1
GRID_PROB = 0.5

"""   SMALL OBJECT AUGMENTATION   """
SMALL_OBJECT_AUGMENTATION = True
SOA_THRESH = 64*64
SOA_PROB = 1
SOA_COPY_TIMES = 3
SOA_EPOCHS = 30
SOA_ONE_OBJECT = False
SOA_ALL_OBJECTS = False

"""    MIXUP(TODO)    """
MIXUP = False
MIXUP_ALPHA = 1




