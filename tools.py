import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image

colors = [
    (39, 129, 113),
    (164, 80, 133),
    (83, 122, 114),
    (99, 81, 172),
    (95, 56, 104),
    (37, 84, 86),
    (14, 89, 122),
    (80, 7, 65),
    (10, 102, 25),
    (90, 185, 109),
    (106, 110, 132),
    (169, 158, 85),
    (188, 185, 26),
    (103, 1, 17),
    (82, 144, 81),
    (92, 7, 184),
    (49, 81, 155),
    (179, 177, 69),
    (93, 187, 158),
    (13, 39, 73),
    (12, 50, 60),
    (16, 179, 33),
    (112, 69, 165),
    (15, 139, 63),
    (33, 191, 159),
    (182, 173, 32),
    (34, 113, 133),
    (90, 135, 34),
    (53, 34, 86),
    (141, 35, 190),
    (6, 171, 8),
    (118, 76, 112),
    (89, 60, 55),
    (15, 54, 88),
    (112, 75, 181),
    (42, 147, 38),
    (138, 52, 63),
    (128, 65, 149),
    (106, 103, 24),
    (168, 33, 45),
    (28, 136, 135),
    (86, 91, 108),
    (52, 11, 76),
    (142, 6, 189),
    (57, 81, 168),
    (55, 19, 148),
    (182, 101, 89),
    (44, 65, 179),
    (1, 33, 26),
    (122, 164, 26),
    (70, 63, 134),
    (137, 106, 82),
    (120, 118, 52),
    (129, 74, 42),
    (182, 147, 112),
    (22, 157, 50),
    (56, 50, 20),
    (2, 22, 177),
    (156, 100, 106),
    (21, 35, 42),
    (13, 8, 121),
    (142, 92, 28),
    (45, 118, 33),
    (105, 118, 30),
    (7, 185, 124),
    (46, 34, 146),
    (105, 184, 169),
    (22, 18, 5),
    (147, 71, 73),
    (181, 64, 91),
    (31, 39, 184),
    (164, 179, 33),
    (96, 50, 18),
    (95, 15, 106),
    (113, 68, 54),
    (136, 116, 112),
    (119, 139, 130),
    (31, 139, 34),
    (66, 6, 127),
    (62, 39, 2),
    (49, 99, 180),
    (49, 119, 155),
    (153, 50, 183),
    (125, 38, 3),
    (129, 87, 143),
    (49, 87, 40),
    (128, 62, 120),
    (73, 85, 148),
    (28, 144, 118),
    (29, 9, 24),
    (175, 45, 108),
    (81, 175, 64),
    (178, 19, 157),
    (74, 188, 190),
    (18, 114, 2),
    (62, 128, 96),
    (21, 3, 150),
    (0, 6, 95),
    (2, 20, 184),
    (122, 37, 185),
]


def bbox_to_rect(bbox, color):
    return plt.Rectangle(
        xy=(bbox[0],bbox[1]), width=bbox[2]-bbox[0],height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )

def visualization(set, sample):
    image, annots = sample['img'], sample['annot']
    fig = plt.imshow(image)
    for i in range(len(annots)):
        annot = [int(x) for x in annots[i]]
        label = annot[4]
        name = set.label_to_name(label)
        color = [c/255.0 for c in colors[label]]
        rect = bbox_to_rect(annot, color)
        fig.axes.add_patch(rect)
        fig.axes.text(rect.xy[0]+24, rect.xy[1]+10,
                      name, va='center', ha='center', fontsize=6, color='blue',
                      bbox=dict(facecolor='m'))
    plt.show()

def easy_visualization(sample):
    image, annots = sample['img'], sample['annot']
    fig = plt.imshow(image)
    for i in range(len(annots)):
        annot = [int(x) for x in annots[i]]
        label = annot[4]
        color = [c/255.0 for c in colors[label]]
        rect = bbox_to_rect(annot, color)
        fig.axes.add_patch(rect)
    plt.show()

def rotate_visualzation(sample):
    image, annots = sample['img'], sample['annot']
    image = Image.fromarray(np.int8(image * 255))
    draw = ImageDraw.Draw(image)
    for annot in annots:
        xmin, ymin, xmax, ymax = annot[0], annot[1], annot[2], annot[3]
        draw.line([(xmin, ymin), (xmax, ymax)])
        """
            需要四个点，然后依次给五个点，形成一个闭合矩形
        """
    image.show()

class SplitKittiDataset():
    def __init__(self,
                 root_dir,
                 ratio=0.5):
        self.ratio = ratio
        self.root_dir = root_dir
        self.ids = list()
        self.find_file_list()
        self.split()

    def find_file_list(self):
        file_path = os.path.join(self.root_dir, 'image_2')
        for _, _, files in os.walk(file_path):
            for file in files:
                self.ids.append(file[:-4])

    def __len__(self):
        return len(self.ids)

    def split(self):
        dataset_size = self.__len__()
        indices = list(range(dataset_size))
        split = int(np.floor(self.ratio * dataset_size))
        np.random.seed(42)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_file = self.root_dir + '/train.txt'
        val_file = self.root_dir + '/val.txt'
        self.write_file(train_file, [self.ids[ind] for ind in train_indices])
        self.write_file(val_file, [self.ids[ind] for ind in val_indices])

    def write_file(self, file_name, write_data):
        f = open(file_name, "w")
        for data in write_data:
            if data != 'Thumb':
                f.write(data + '\n')



if __name__ == '__main__':
    root_path = 'D:/KITTI/training'
    spkitti = SplitKittiDataset(root_path)
