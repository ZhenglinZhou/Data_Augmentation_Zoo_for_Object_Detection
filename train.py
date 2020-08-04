from prepare_data import KittiDataset, VocDataset, collater, Resizer, AspectRatioBasedSampler, Normalizer
from torch.utils.data import DataLoader
import torch
from Augmentation import mixup, mix_loss, RandomFlip, AutoAugmenter
from torchvision import transforms
import collections
import torch.optim as optim
from retinanet import model
import numpy as np
from tools import SplitKittiDataset
from retinanet import csv_eval
import config as cfg
import os
from augmentation_zoo.MyGridMask import GridMask
from augmentation_zoo.SmallObjectAugmentation import SmallObjectAugmentation
"""
    author: zhenglin.zhou
    date: 20200724
"""

# os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CUDA_DEVICES

print('CUDA available: {}'.format(torch.cuda.is_available()))

def _make_transform():
    transform_list = list()
    if cfg.AUTOAUGMENT:
        transform_list.append(AutoAugmenter(cfg.AUTO_POLICY))
    if cfg.GRID:
        transform_list.append(GridMask(True, True, cfg.GRID_ROTATE,cfg.GRID_OFFSET,cfg.GRID_RATIO,cfg.GRID_MODE,cfg.GRID_PROB))
    if cfg.RANDOM_FLIP:
        transform_list.append(RandomFlip())
    if cfg.SMALL_OBJECT_AUGMENTATION:
        transform_list.append(SmallObjectAugmentation(cfg.SOA_THRESH, cfg.SOA_PROB, cfg.SOA_COPY_TIMES, cfg.SOA_EPOCHS, cfg.SOA_ALL_OBJECTS, cfg.SOA_ONE_OBJECT))
    transform_list.append(Normalizer())
    transform_list.append(Resizer())
    return transform_list

def _make_dataset():
    transform_list = _make_transform()
    if cfg.DATASET_TYPE == 1:
        batch_size = cfg.VOC_BATCH_SIZE
        dataset_train = VocDataset(cfg.VOC_ROOT_DIR, 'train', transform=transforms.Compose(transform_list))
        dataset_val = VocDataset(cfg.VOC_ROOT_DIR, 'val', transform=transforms.Compose([Normalizer(), Resizer()]))
    elif cfg.DATASET_TYPE == 2:
        root_dir = cfg.KITTI_ROOT_DIR
        batch_size = cfg.KITTI_BATCH_SIZE
        SplitKittiDataset(root_dir, 0.5)  # 分割KITTI数据集，50%训练集，50%测试集
        dataset_train = KittiDataset(root_dir, 'train', transform=transforms.Compose(transform_list))
        dataset_val = KittiDataset(root_dir, 'val', transform=transforms.Compose([Normalizer(), Resizer()]))
    return batch_size, dataset_train, dataset_val


def main():

    batch_size, dataset_train, dataset_val = _make_dataset()
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist =collections.deque(maxlen=500)

    retinanet.train()
    # retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    BEST_MAP = 0
    BEST_MAP_EPOCH = 0
    for epoch_num in range(cfg.EPOCHS):

        retinanet.train()
        # retinanet.freeze_bn()
        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            # try:
            optimizer.zero_grad()

            if cfg.MIXUP:
                data, lam = mixup(data)

            if torch.cuda.is_available():
                classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
            else:
                classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

            if cfg.MIXUP:
                classification_loss, regression_loss = mix_loss(classification_loss, regression_loss, lam)

            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss
            if bool(loss == 0):
                continue
            loss.backward()

            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
            optimizer.step()

            loss_hist.append(float(loss))
            epoch_loss.append(float(loss))

            print(
                'Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                    epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)))

            del classification_loss
            del regression_loss

        # except Exception as e:
        #     print(e)
        #     continue

        """ validation part """
        print('Evaluating dataset')
        average_precisions, mAP = csv_eval.evaluate(dataset_val, retinanet)
        if mAP > BEST_MAP:
            best_average_precisions = average_precisions
            BEST_MAP = mAP
            BEST_MAP_EPOCH = epoch_num
        scheduler.step(np.mean(epoch_loss))
        # torch.save(retinanet.module, '{}_retinanet_{}.pt'.format('voc', epoch_num)))
    retinanet.eval()

    print('\nBest_mAP:', BEST_MAP_EPOCH)
    for label in range(retinanet.num_classes()):
        label_name = retinanet.label_to_name(label)
        print('{}: {}'.format(label_name, best_average_precisions[label][0]))
    print('BEST MAP: ', BEST_MAP)
    # torch.save(retinanet, 'model_final.pt')

if __name__ == '__main__':

    main()