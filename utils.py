import os
import numpy as np


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
        f = open(file_name, "a+")
        for data in write_data:
            f.write(data + '\n')

if __name__ == '__main__':
    root_path = 'D:/KITTI/training'
    spkitti = SplitKittiDataset(root_path)
