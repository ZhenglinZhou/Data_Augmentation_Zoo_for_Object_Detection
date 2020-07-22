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
                 years = ['2007', '2012'],
                 image_sets = ['train', 'val', 'test'],
                 transform=None,
                 keep_difficult=False
                 ):
        self.root_dir = root_dir
        self.years = years
        self.image_sets = image_sets
        self.transform = transform
        self.keep_difficult = keep_difficult

        self.categories = VOC_CLASSES
        self.voc_categories_to_voc_label = dict(
            zip(self.categories, range(len(self.categories)))
        )
        self.voc_label_to_voc_category = {
            v: k
            for k, v in self.voc_categories_to_voc_label.items()
        }
        self.ids = self.find_file_list()
        self.train_list, self.val_list, self.test_list = [v for k,v in self.ids.items()]

    def __getitem__(self, sets, image_index): #sets: train/val/test
        image_root_dir, img_idx = self.ids[sets][image_index]
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
        file_list = dict()
        for sets in self.image_sets:
            ids = list()
            for year in self.years:
                if not (year == '2012' and sets == 'test'):
                    root_path = os.path.join(self.root_dir, 'VOC' + year)
                    file_path = os.path.join(root_path, 'ImageSets', 'Main', sets + '.txt')
                    for line in open(file_path):
                        ids.append((root_path, line.strip()))
            file_list[sets] = ids.copy()
        return file_list

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
            label = self.voc_categories_to_voc_label[categray]
            bndbox.append(label)
            annotations += [bndbox]
        annotations = np.array(annotations)

        return annotations

    def find_category_id_from_voc_label(self, voc_label):
        return self.voc_label_to_voc_category[voc_label]

    def image_aspect_ratio(self, image_path):
        img = skimage.io.imread(image_path)
        return float(img.shape[1] / float(img.shape[0]))

class Augmenter(object):
    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :] #random flip



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


