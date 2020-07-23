import sys
sys.path.insert(0, "efficientdet-pytorch")
import json
import zipfile
import gluoncv as gcv
import torch
import os
from datetime import datetime
import time
import random
import cv2
import pandas as pd
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from glob import glob
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet
import warnings
warnings.filterwarnings("ignore")


IMG_SIZE = 1024
fold_number = 0
csv_path = 'train_adjusted_v2.csv'
#csv_path = 'train.csv'
image_dir = 'all_images/trainval'
weight_path = 'efficientdet_d5-ef44aea8.pth'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
SEED = 42
AREA_SMALL = 100


class TrainGlobalConfig:
    num_workers = 4
    batch_size = 2
    n_epochs = 75  # n_epochs = 40
    lr = 0.001
    grad_accumulation_steps = 4

    folder = f'effdet5-cutmix-augmix-1024-fold{fold_number}'

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=4,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def Kfold():
    if os.path.splitext(csv_path)[0] == 'train':
        marking = pd.read_csv(csv_path)
    else:
        marking = pd.read_csv(csv_path, header=None, names=['image_id', 'width', 'height', 'bbox', 'source'])

    bboxs = np.stack(marking['bbox'].apply(lambda x: np.fromstring(x[1:-1], sep=',')))
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:,i]

    marking['area'] = marking.w * marking.h
    marking['class'] = 1
    marking['size'] = (marking.area > AREA_SMALL).astype(int)
    marking['source_path'] = image_dir
    marking.drop(columns=['bbox'], inplace=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = marking[['image_id', 'source', 'class']].groupby('image_id').min()['source']
    df_folds.loc[:, 'class'] = marking[['image_id', 'class']].groupby('image_id').min()['class']
    df_folds.loc[:, 'stratify_group'] = np.char.add(df_folds['class'].apply(lambda x: f'{x}_').values.astype(str), np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    ))
    df_folds.loc[:, 'fold'] = 0

    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number
    return df_folds, marking


def get_train_transforms():
    return A.Compose(
        [
            A.ShiftScaleRotate(scale_limit=(-0.5, 0.5), rotate_limit=0, shift_limit=0., p=0.5, border_mode=0),
            A.RandomRotate90(p=0.5),
            A.Resize(IMG_SIZE, IMG_SIZE, p=1),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                         val_shift_limit=0.2, p=0.9),
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2, p=0.9),
                    A.RandomGamma(p=0.9),
            ],p=0.25),
            A.OneOf([
                A.IAASharpen(alpha=(0.1, 0.3), p=0.5),
                A.CLAHE(p=0.8),
                #A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                #A.GaussianBlur(blur_limit=3, p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            ], p=0.0),
            A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


class DatasetRetriever(Dataset):

    def __init__(self, marking, image_ids, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.transforms = transforms
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        r = random.random()
        if self.test:
            image, boxes, labels = self.load_image_and_boxes(index)
        elif r < 0.50:
            image, boxes, labels = self.load_cutmix_image_and_boxes(index)
        else:
            image, boxes, labels = self.load_mixup_iamge_and_boxes(index)

        assert len(boxes) == len(labels)
        target = {}
        target['boxes'] = boxes
        target['labels'] = torch.tensor(labels.astype(np.uint8))
        target['image_id'] = torch.tensor([index])
        target['img_size'] = torch.tensor([(IMG_SIZE, IMG_SIZE)])
        target['img_scale'] = torch.tensor([1.])
        image = image.astype(np.uint8)

        if self.transforms:
            for i in range(10):
                sample = self.transforms(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })

                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]  # yxyx: be warning
                    target['labels'] = torch.tensor(sample['labels'])
                    break

        assert len(target['boxes']) == len(target['labels'])
        image = image.float() / 255.0

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        image_id = self.image_ids[index]
        records = self.marking[self.marking['image_id'] == image_id]
        source_path = records['source_path'].iloc[0]
        image = cv2.imread(f'{source_path}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        # image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0,0) , 10), -4 ,128)
        # image = (image - IMAGENET_DEFAULT_MEAN) / IMAGENET_DEFAULT_STD
        boxes = records[['x', 'y', 'w', 'h']].values
        labels = records['class'].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = np.clip(boxes, 0, 1024)
        return image, boxes, labels

    def load_mixup_iamge_and_boxes(self, index):
        image, boxes, labels = self.load_image_and_boxes(index)
        r_image, r_boxes, r_labels = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        mixup_image = (image + r_image) / 2
        mixup_boxes = np.concatenate([boxes, r_boxes], axis=0)
        mixup_labels = np.concatenate([labels, r_labels], axis=0)
        return mixup_image, mixup_boxes, mixup_labels

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2

        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.float32)
        result_boxes = []
        result_labels = []

        for i, index in enumerate(indexes):
            image, boxes, labels = self.load_image_and_boxes(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
            result_labels.append(labels)

        result_boxes = np.concatenate(result_boxes, 0)
        result_labels = np.concatenate(result_labels, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        condition = np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)
        result_boxes = result_boxes[condition]
        result_labels = result_labels[condition]
        return result_image, result_boxes, result_labels


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Fitter:

    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0

        self.base_dir = f'./{config.folder}'
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.log_path = f'{self.base_dir}/log.txt'
        self.best_summary_loss = 10 ** 5
        self.best_summary_score = 0.0

        self.model = model
        self.device = device

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')

    def fit(self, train_loader, validation_loader):
        for e in range(self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                self.log(f'\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, summary_score = self.validation(validation_loader)
            self.log(f'[RESULT]: Validation. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, summary_score: {summary_score.avg:.5f}, time: {(time.time() - t):.5f}')

            '''
            if summary_score.avg > self.best_summary_score:
                self.best_summary_score = summary_score.avg
            '''
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_los = summary_loss.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1


    def validation(self, val_loader):
        self.model.eval()
        summary_loss = AverageMeter()
        summary_score = AverageMeter()
        t = time.time()
        metrics = [gcv.utils.metrics.VOCMApMetric(iou_thresh=iou) for iou in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]]

        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}, ' + \
                        f'score: {summary_score.avg:.5f} ', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                targets2 = {}
                targets2['bbox'] = [target['boxes'].to(self.device).float() for target in
                                    targets]  # variable number of instances, so the entire structure can be forced to tensor
                targets2['cls'] = [target['labels'].to(self.device).float() for target in targets]
                targets2['image_id'] = torch.tensor([target['image_id'] for target in targets]).to(self.device).float()
                targets2['img_scale'] = torch.tensor([target['img_scale'] for target in targets]).to(
                    self.device).float()
                targets2['img_size'] = torch.tensor([(IMG_SIZE, IMG_SIZE) for target in targets]).to(
                    self.device).float()
                output = self.model(images, targets2)
                loss = output['loss']
                det = output['detections']
                summary_loss.update(loss.detach().item(), batch_size)

                # update(pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels, gt_difficults=None)

                for i in range(0, len(det)):
                    pred_scores = det[i, :, 4].cpu().unsqueeze_(0).numpy()
                    condition = (pred_scores > 0.25)[0]
                    gt_boxes = targets2['bbox'][i].cpu().unsqueeze_(0).numpy()[
                        ..., [1, 0, 3, 2]]  # move to PASCAL VOC from yxyx format
                    gt_labels = targets2['cls'][i].cpu().unsqueeze_(0).numpy()
                    pred_bboxes = det[i, :, 0:4].cpu().unsqueeze_(0).numpy()[:, condition,
                                  :]  # move from (x,y,w,h) to (x,y,x,y)
                    pred_bboxes[..., 2] = pred_bboxes[..., 0] + pred_bboxes[..., 2]
                    pred_bboxes[..., 3] = pred_bboxes[..., 1] + pred_bboxes[..., 3]
                    pred_labels = det[i, :, 5].cpu().unsqueeze_(0).numpy()[:, condition]
                    pred_scores = pred_scores[:, condition]

                    for metric in metrics:
                        metric.update(
                            pred_bboxes=pred_bboxes,
                            pred_labels=pred_labels,
                            pred_scores=pred_scores,
                            gt_bboxes=gt_boxes,
                            gt_labels=gt_labels)

                summary_score.update(np.mean([metric.get()[1] for metric in metrics]), batch_size)

        return summary_loss, summary_score

    def train_one_epoch(self, train_loader):
        self.model.train()
        summary_loss = AverageMeter()
        t = time.time()
        for step, (images, targets, image_ids) in enumerate(train_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Train Step {step}/{len(train_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )

            images = torch.stack(images)
            images = images.to(self.device).float()
            batch_size = images.shape[0]
            targets2 = {}
            targets2['bbox'] = [target['boxes'].to(self.device).float() for target in
                                targets]  # variable number of instances, so the entire structure can be forced to tensor
            targets2['cls'] = [target['labels'].to(self.device).float() for target in targets]
            targets2['image_id'] = torch.tensor([target['image_id'] for target in targets]).to(self.device).float()
            targets2['img_scale'] = torch.tensor([target['img_scale'] for target in targets]).to(self.device).float()
            targets2['img_size'] = torch.tensor([(IMG_SIZE, IMG_SIZE) for target in targets]).to(self.device).float()

            output = self.model(images, targets2)
            loss = output['loss'] # / self.config.grad_accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (step + 1) % self.config.grad_accumulation_steps == 0:
                summary_loss.update(loss.detach().item(), batch_size*self.config.grad_accumulation_steps)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.config.step_scheduler:
                    self.scheduler.step()

        return summary_loss

    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'best_summary_score': self.best_summary_score,
            'epoch': self.epoch
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.best_summary_score = checkpoint['best_summary_score']
        self.epoch = checkpoint['epoch'] + 1

    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


def collate_fn(batch):
    return tuple(zip(*batch))


def run_training(train_dataset, valid_dataset, path=''):
    device = torch.device('cuda:0')
    net.to(device)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        sampler=RandomSampler(train_dataset),
        pin_memory=False,
        drop_last=True,
        num_workers=TrainGlobalConfig.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    if path:
        fitter.load(path=path)
    fitter.fit(train_loader, val_loader)


def get_net(config='', checkpoint='', n_classes=1):
    config = get_efficientdet_config(config)
    net = EfficientDet(config, pretrained_backbone=False)
    config.num_classes = n_classes
    config.image_size = IMG_SIZE
    head = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    if checkpoint:
      metadata = torch.load(checkpoint)
      if os.path.splitext(checkpoint)[1] == '.pth':
        net.load_state_dict(metadata)
        net.class_net = head
      else:
        net.class_net = head
        net.load_state_dict(metadata['model_state_dict'])

    return DetBenchTrain(net, config)


if __name__ == "__main__":
    seed_everything(SEED)

    net = get_net('tf_efficientdet_d5', weight_path, 1)

    df_folds, marking = Kfold()
    train_dataset = DatasetRetriever(
        image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
        marking=marking,
        transforms=get_train_transforms(),
        test=False,
    )
    validation_dataset = DatasetRetriever(
        image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
        marking=marking,
        transforms=get_valid_transforms(),
        test=True,
    )

    run_training(train_dataset, validation_dataset)
