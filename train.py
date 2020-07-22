from prepare_data import KittiDataset, VocDataset, collater, Resizer, AspectRatioBasedSampler, Normalizer
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch
from Augmentation import RetinaNet_Augmenter
from torchvision import transforms
from picture_visualization import visualization
import collections
import torch.optim as optim
from retinanet import model
import numpy as np
from utils import SplitKittiDataset


from retinanet import csv_eval
print('CUDA available: {}'.format(torch.cuda.is_available()))

def main():
    epochs = 100
    voc_root_dir = 'D:\VOC\VOCdevkit'
    kitti_root_dir = 'D:/KITTI/training'
    type = 2

    if type == 1:
        dataset_train = VocDataset(voc_root_dir, 'train',transform=transforms.Compose([
                                       Normalizer(),
                                       RetinaNet_Augmenter(),
                                       Resizer()]))

        dataset_val = VocDataset(voc_root_dir, 'val', transform=transforms.Compose([
                                       Normalizer(),
                                       RetinaNet_Augmenter(),
                                       Resizer()]))
    elif type == 2:
        SplitKittiDataset(kitti_root_dir, 0.5)
        dataset_train = KittiDataset(kitti_root_dir, 'train', transform=transforms.Compose([
            Normalizer(),
            RetinaNet_Augmenter(),
            Resizer()]))

        dataset_val = KittiDataset(kitti_root_dir, 'val', transform=transforms.Compose([
            Normalizer(),
            RetinaNet_Augmenter(),
            Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3,
                                  collate_fn=collater, batch_sampler=sampler)
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)
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
            try:
                optimizer.zero_grad()

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
            except Exception as e:
                print(e)
                continue

        """ validation part """
        print('Evaluating dataset')
        mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format('voc', epoch_num))

    retinanet.eval()

    torch.save(retinanet, 'model_final.pt')

if __name__ == '__main__':
    main()