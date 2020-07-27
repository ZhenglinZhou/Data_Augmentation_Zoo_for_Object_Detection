"""
    author: zhenglin.zhou
    date: 20200724
"""
from prepare_data import KittiDataset, VocDataset, collater, Resizer, AspectRatioBasedSampler, Normalizer
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from Augmentation import Augmenter, mixup
from torchvision import transforms
from picture_visualization import visualization
import collections
import torch.optim as optim
from retinanet import model
import numpy as np
from tools import SplitKittiDataset
from retinanet import csv_eval
import config

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main():
    use_mixup = config.use_mixup
    epochs = config.epochs
    transform = transforms.Compose([Normalizer(),
                                    Augmenter(),
                                    Resizer()])

    if config.dataset_type == 1:
        root_dir = config.voc_root_dir
        batch_size = config.voc_batch_size
        dataset_train = VocDataset(root_dir, 'train', transform=transform)

        dataset_val = VocDataset(root_dir, 'val', transform=transform)

    elif config.dataset_type == 2:
        root_dir = config.kitti_root_dir
        batch_size = config.kitti_batch_size
        SplitKittiDataset(root_dir, 0.5)  # 分割KITTI数据集，50%训练集，50%测试集

        dataset_train = KittiDataset(root_dir, 'train', transform=transform)

        dataset_val = KittiDataset(root_dir, 'val', transform=transform)

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3,
                                  collate_fn=collater, batch_sampler=sampler)
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=batch_size, drop_last=False)
    dataloader_val = DataLoader(dataset_train, num_workers=3,
                                  collate_fn=collater, batch_sampler=sampler)


    retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3,verbose=True)
    loss_hist =collections.deque(maxlen=500)

    retinanet.train()
    # retinanet.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))


    for epoch_num in range(epochs):

        retinanet.train()
        # retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            # try:
            optimizer.zero_grad()

            if use_mixup:
                # data = data.numpy()
                order = list(range(data['img'].shape[0]))
                np.random.seed(36)
                np.random.shuffle(order)

                order = [[order[x % len(order)] for x in range(i, i + 2)] for i in
                        range(0, len(order), 2)]
                img = data['img'].numpy()
                annot = data['annot'].numpy()
                for _order in order:
                    sample1 = {'img':img[_order[0]], 'annot': annot[_order[0]]}
                    sample2 = {'img':img[_order[1]], 'annot': annot[_order[1]]}
                    mix_img, lam = mixup(sample1, sample2)
                    new_img = torch.Tensor([mix_img, mix_img])
                    new_annot = torch.Tensor([annot[_order[0]], annot[_order[1]]])

                    new_data = {'img': new_img, 'annot': new_annot}

                    if torch.cuda.is_available():
                        classification_loss, regression_loss = retinanet(
                            [new_data['img'].cuda().float(), new_data['annot']])
                    else:
                        classification_loss, regression_loss = retinanet([new_data['img'].float(), new_data['annot']])
                    classification_loss_0, classification_loss_1 = classification_loss.chunk(2)
                    regression_loss_0, regression_loss_1 = regression_loss.chunk(2)
                    classification_loss = classification_loss_0 * lam + classification_loss_1 * (1 - lam)
                    regression_loss = regression_loss_0 * lam + regression_loss_1 * (1 - lam)

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
                            epoch_num, iter_num, float(classification_loss), float(regression_loss),
                            np.mean(loss_hist)))

                    del classification_loss
                    del regression_loss
            else:
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

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
        mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format('voc', epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')

if __name__ == '__main__':
    main()