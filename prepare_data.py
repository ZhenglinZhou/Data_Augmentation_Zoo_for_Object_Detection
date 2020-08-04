import numpy as np
import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset, Sampler
import random
import cv2


VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

KITTI_CLASSES = [
    'car',
    'van',
    'truck',
    'pedestrian',
    'person_sitting',
    'cyclist',
    'tram',
    'misc',
]

class KittiDataset(Dataset):
    def __init__(self,
                 root_dir,
                 sets,
                 transform=None,
                 keep_difficult=False
                 ):
        self.root_dir = root_dir
        self.sets = sets
        self.transform = transform
        self.keep_difficult = keep_difficult

        self.categories = KITTI_CLASSES

        self.name_2_label = dict(
            zip(self.categories, range(len(self.categories)))
        )
        self.label_2_name = {
            v: k
            for k, v in self.name_2_label.items()
        }
        self.ids = list()
        self.find_file_list()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, image_index):
        img = self.load_image(image_index)
        annot = self.load_annotations(image_index)
        sample = {'img':img, 'annot':annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def find_file_list(self):
        file_path = os.path.join(self.root_dir, self.sets + '.txt')
        for line in open(file_path):
            self.ids.append(line.strip())

    def load_image(self, image_index):
        img_idx = self.ids[image_index]
        image_path = os.path.join(self.root_dir,
                                 'image_2', img_idx + '.png')
        img = cv2.imread(image_path)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        img_idx = self.ids[image_index]
        anna_path = os.path.join(self.root_dir,
                                'label_2', img_idx + '.txt')
        annotations = []
        with open(anna_path) as file:
            lines = file.readlines()
            for line in lines:
                items = line.split(" ")
                name = items[0].lower().strip()
                if name == 'dontcare':
                    continue
                else:
                    bndbox = [float(items[i+4]) for i in range(4)]
                    if (bndbox[2] - bndbox[0]) <= 0 or (bndbox[3] - bndbox[1]) <= 0:
                        continue
                    label = self.name_2_label[name]
                    bndbox.append(int(label))
                annotations.append(bndbox)
        annotations = np.array(annotations)
        return annotations

    def label_to_name(self, voc_label):
        return self.label_2_name[voc_label]

    def name_to_label(self, voc_name):
        return self.name_2_label[voc_name]

    def image_aspect_ratio(self, image_index):
        img_idx = self.ids[image_index]
        image_path = os.path.join(self.root_dir,
                                  'image_2', img_idx + '.png')
        img = cv2.imread(image_path)
        return float(img.shape[1] / float(img.shape[0]))

    def num_classes(self):
        return 8

class VocDataset(Dataset):
    def __init__(self,
                 root_dir,
                 image_set='train',         # train/val/test
                 years=['2007', '2012'],    # 默认2007+2012
                 transform=None,
                 keep_difficult=False
                 ):
        self.root_dir = root_dir
        self.years = years

        self.image_set = image_set
        self.transform = transform
        self.keep_difficult = keep_difficult

        self.categories = VOC_CLASSES

        self.name_2_label = dict(
            zip(self.categories, range(len(self.categories)))
        )
        self.label_2_name = {
            v: k
            for k, v in self.name_2_label.items()
        }
        self.ids = list()
        self.find_file_list()


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, image_index):

        img = self.load_image(image_index)
        annots = self.load_annotations(image_index)
        sample = {'img': img, 'annot': annots}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def find_file_list(self):
        for year in self.years:
            if not (year == '2012' and self.image_set == 'test'):
                root_path = os.path.join(self.root_dir, 'VOC' + year)
                file_path = os.path.join(root_path, 'ImageSets', 'Main', self.image_set + '.txt')
                for line in open(file_path):
                    self.ids.append((root_path, line.strip()))

    def load_image(self, image_index):
        image_root_dir, img_idx = self.ids[image_index]
        image_path = os.path.join(image_root_dir,
                                 'JPEGImages', img_idx + '.jpg')
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        image_root_dir, img_idx = self.ids[image_index]
        anna_path = os.path.join(image_root_dir,
                                'Annotations', img_idx + '.xml')
        annotations = []
        target = ET.parse(anna_path).getroot()
        for obj in target.iter("object"):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            bndbox = []
            for pt in pts:
                cut_pt = bbox.find(pt).text
                bndbox.append(np.float32(cut_pt))
            if (bndbox[2] - bndbox[0]) <= 0 or (bndbox[3] - bndbox[1]) <= 0:
                continue
            name = obj.find('name').text.lower().strip()
            label = self.name_2_label[name]
            bndbox.append(label)
            annotations += [bndbox]
        annotations = np.array(annotations)

        return annotations

    def label_to_name(self, voc_label):
        return self.label_2_name[voc_label]

    def name_to_label(self, voc_name):
        return self.name_2_label[voc_name]

    def image_aspect_ratio(self, image_index):
        image_root_dir, img_idx = self.ids[image_index]
        image_path = os.path.join(image_root_dir,
                                  'JPEGImages', img_idx + '.jpg')
        img = cv2.imread(image_path)
        return float(img.shape[1] / float(img.shape[0]))

    def num_classes(self):
        return 20

class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    """
        output
        img = [batch_size x 3 x W x H]
    """
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        """
            image = [H * W * 3]
            cv2.resize(image, (resize_W, resize_H))
        """
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))

        # image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots = annots.astype(np.float32)
        annots[:, :4] *= scale
        annots = annots.astype(np.int)

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}

class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]



