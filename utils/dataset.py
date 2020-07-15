import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset


class DatasetRetriever(Dataset):

    def __init__(self, path, marking, image_ids, transforms=None, test=False):
        super().__init__()
        self.image_dir = path
        # 图片的 ID 列表
        self.image_ids = image_ids
        # 图片的标签和基本信息
        self.marking = marking
        # 图像增强
        self.transforms = transforms
        # 测试集
        self.test = test

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        # 百分之 50 的概率会做 mix up
        if self.test or random.random() > 0.5:
            # 具体定义在后面
            image, boxes = self.load_image_and_boxes(index)
        else:
            # 具体定义在后面
            image, boxes = self.load_mixup_image_and_boxes(index)

        # 这里只有一类的目标定位问题，标签数量就是 bbox 的数量
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([index])

        # 多做几次图像增强，防止有图像增强失败，如果成功，则直接返回。
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
                    break

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]

    def load_image_and_boxes(self, index):
        # 加载 image_id 名字
        image_id = self.image_ids[index]
        # 加载图片
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        # 转换图片通道 从 BGR 到 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # 0,1 归一化
        image /= 255.0
        # 获取对应 image_id 的信息
        records = self.marking[self.marking['image_id'] == image_id]
        # 获取 bbox
        boxes = records[['x', 'y', 'w', 'h']].values
        # 转换成模型输入需要的格式
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = np.clip(boxes, 0, 1024)
        return image, boxes

    def load_mixup_image_and_boxes(self, index, imsize=1024):
        # 加载图片和 bbox
        image, boxes = self.load_image_and_boxes(index)
        # 随机加载另外一张图片和 bbox
        r_image, r_boxes = self.load_image_and_boxes(random.randint(0, self.image_ids.shape[0] - 1))
        # 进行 mixup 图片的融合，这里简单的利用 0.5 权重
        mixup_image = (image + r_image) / 2
        # 进行 mixup bbox的融合
        mixup_boxes = np.concatenate((boxes, r_boxes), 0)
        return mixup_image, mixup_boxes