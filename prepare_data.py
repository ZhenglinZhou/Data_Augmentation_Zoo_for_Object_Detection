import numpy as np
import os
import skimage.io
import skimage.color
import xml.etree.ElementTree as ET
import picture_visualization as pv
import matplotlib.pyplot as plt

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
class VocDataset:
    def __init__(self,
                 root_dir,
                 years=['2007', '2012'],    # 默认2007+2012
                 image_set='train',               # train/val/test
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

    def __getitem__(self, image_index): #sets: train/val/test
        image_root_dir, img_idx = self.ids[image_index]
        image_dir = os.path.join(image_root_dir,
                                 'JPEGImages', img_idx + '.jpg')
        anna_dir = os.path.join(image_root_dir,
                                'Annotations', img_idx + '.xml')
        img = self.load_image(image_dir)
        annot = self.load_annotations(anna_dir)
        sample = {'img':img, 'annot':annot}
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

    def load_image(self, image_path):
        img = skimage.io.imread(image_path)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, anna_path):
        annotations = []
        target = ET.parse(anna_path).getroot()
        for obj in target.iter("object"):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            bndbox = []
            for pt in pts:
                cut_pt = bbox.find(pt).text
                bndbox.append(cut_pt)
            categray = obj.find('name').text.lower().strip()
            label = self.name_2_label[categray]
            bndbox.append(label)
            annotations += [bndbox]
        annotations = np.array(annotations)

        return annotations

    def find_category_id_from_voc_label(self, voc_label):
        return self.label_2_name[voc_label]

    def image_aspect_ratio(self, image_path):
        img = skimage.io.imread(image_path)
        return float(img.shape[1] / float(img.shape[0]))

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
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class AspectRatioBasedSampler():
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def group_images(self):
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]


if __name__ == '__main__':
    root_dir = 'D:\VOC\VOCdevkit'
    voc = VocDataset(root_dir)
    sample = voc.__getitem__('train', 2)
    name = []
    for i in range(len(sample['annot'])):
        name.append(
            voc.find_category_id_from_voc_label(
                int(sample['annot'][i][4])
            )
        )
    pv.visualization(sample, name)
    image = sample['img'][:, ::-1, :]
    fig = plt.imshow(image)
    plt.show()


