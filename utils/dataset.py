import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset


class DatasetRetriever(Dataset):
    def __init__(self, marking, image_ids, path, transforms=None, test=False):
        super().__init__()

        self.image_ids = image_ids
        self.marking = marking
        self.path = path
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
        target['img_size'] = torch.tensor([(1024, 1024)])
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
        image = cv2.imread(f'{self.path}/{image_id}.jpg', cv2.IMREAD_COLOR)
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
