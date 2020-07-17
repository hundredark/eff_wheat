# 导入依赖的库
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from utils.dataset import DatasetRetriever
from utils.aug import get_train_transforms, get_valid_transforms
from utils.train_utils import Fitter
from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain
from effdet.efficientdet import HeadNet


fold = 0
csv_path = r"./train_adjusted_v2.csv"
TRAIN_ROOT_PATH = r'./all_images/trainval'
weight = "./effdet5-cutmix-augmix0/last-checkpoint.bin"
gpu = 0


class TrainGlobalConfig:
    num_workers = 4
    batch_size = 6
    n_epochs = 50  # n_epochs = 40
    lr = 0.0004

    folder = 'effdet5-cutmix-augmix-fold{}'.format(fold)

    # -------------------
    verbose = True
    verbose_step = 1
    # -------------------

    # --------------------
    # 我们只在每次 epoch 完，验证完后，再执行学习策略。
    step_scheduler = False  # do scheduler.step after optimizer.step
    validation_scheduler = True  # do scheduler.step after validation stage loss

    # 当指标变化小时，减少学习率
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )


def Kfold(csv_path, k=5):
    marking = pd.read_csv(csv_path, header=None, names=['image_id', 'width', 'height', 'bbox', 'source'])
    bboxs = np.stack(marking['bbox'].apply(lambda x: eval(x)))

    for i, column in enumerate(['x', 'y', 'w', 'h']):
        marking[column] = bboxs[:,i]
    marking.drop(columns=['bbox'], inplace=True)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    df_folds = marking[['image_id']].copy()
    df_folds.loc[:, 'bbox_count'] = 1
    df_folds = df_folds.groupby('image_id').count()
    df_folds.loc[:, 'source'] = marking[['image_id', 'source']].groupby('image_id').min()['source']
    df_folds.loc[:, 'stratify_group'] = np.char.add(
        df_folds['source'].values.astype(str),
        df_folds['bbox_count'].apply(lambda x: f'_{x // 15}').values.astype(str)
    )
    a = marking[['x']].max()
    print(a)

    df_folds.loc[:, 'fold'] = 0
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df_folds.index, y=df_folds['stratify_group'])):
        df_folds.loc[df_folds.iloc[val_index].index, 'fold'] = fold_number

    return df_folds, marking


def collate_fn(batch):
    return tuple(zip(*batch))


def get_net():
    # 模型的配置，这个返回的是一个字典
    config = get_efficientdet_config('tf_efficientdet_d5')
    # 根据上面的配置生成网络
    net = EfficientDet(config, pretrained_backbone=False)

    #'''
    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    checkpoint = torch.load(weight)
    net.load_state_dict(checkpoint['model_state_dict'])
    #'''

    '''
    # 加载预训练模型
    checkpoint = torch.load(weight)
    net.load_state_dict(checkpoint)
    config.num_classes = 1
    config.image_size = 512
    # norm_kwargs 设置的是 BATCHNORM2D 的参数
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    '''

    return DetBenchTrain(net, config)


def train(fold_number = 0):
    device = torch.device('cuda:{}'.format(gpu))

    net = get_net()
    net.to(device)

    df_folds, marking = Kfold(csv_path, k=5)

    train_dataset = DatasetRetriever(
        path=TRAIN_ROOT_PATH,
        image_ids=df_folds[df_folds['fold'] != fold_number].index.values,
        marking=marking,
        transforms=get_train_transforms(),
        test=False,
    )

    validation_dataset = DatasetRetriever(
        path=TRAIN_ROOT_PATH,
        image_ids=df_folds[df_folds['fold'] == fold_number].index.values,
        marking=marking,
        transforms=get_valid_transforms(),
        test=True,
    )

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
        validation_dataset,
        batch_size=TrainGlobalConfig.batch_size,
        num_workers=TrainGlobalConfig.num_workers,
        shuffle=False,
        sampler=SequentialSampler(validation_dataset),
        pin_memory=False,
        collate_fn=collate_fn,
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)


if __name__ == "__main__":
    print("{} fold training start".format(fold))
    train(fold)
