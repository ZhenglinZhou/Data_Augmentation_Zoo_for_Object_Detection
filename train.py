from prepare_data import VocDataset, collater, Resizer, AspectRatioBasedSampler, Normalizer
from torch.utils.data import DataLoader
import torch
from Augmentation import RetinaNet_Augmenter
from torchvision import transforms
from picture_visualization import visualization

def main():
    root_dir = 'D:\VOC\VOCdevkit'
    dataset_train = VocDataset(root_dir, 'train',
                               transform=transforms.Compose([
                                   Normalizer(),
                                   RetinaNet_Augmenter(),
                                   Resizer()]))

    dataset_val = VocDataset(root_dir, 'val', transform=transforms.Compose([
                                   Normalizer(),
                                   RetinaNet_Augmenter(),
                                   Resizer()]))
    sample = dataset_val.__getitem__(2)
    visualization(dataset_train, sample)

if __name__ == '__main__':
    main()