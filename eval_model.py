import torch
from retinanet import model
from prepare_data import VocDataset, Normalizer, Resizer, AspectRatioBasedSampler, UnNormalizer, collater
from Augmentation import AutoAugmenter
from torchvision import transforms
import config
from retinanet import csv_eval
from torch.utils.data import DataLoader
import numpy as np
import cv2

def show_image():
    net_path = 'voc_retinanet_19.pt'
    dataset_val = VocDataset(config.voc_root_dir, 'val', transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
    retinanet = torch.load(net_path, map_location=torch.device('cpu'))
    retinanet = torch.nn.DataParallel(retinanet)
    retinanet.eval()

    # mAP = csv_eval.evaluate(dataset_val, retinanet)

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):
        with torch.no_grad():
            scores, classification, transformed_anchors = retinanet(data['img'].float())
            print(scores.cpu())
            idxs = np.where(scores.cpu() > 0.1)
            print(idxs)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            print(idxs[0].shape[0])
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                print(bbox)
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.label_2_name[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)

            cv2.imshow('img', img)
            cv2.waitKey(0)

if __name__ == '__main__':
    show_image()