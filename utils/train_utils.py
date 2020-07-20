import os
import numpy as np
import time
import torch
import datetime
import glob
import gluoncv as gcv
from .utils import AverageMeter

# 模型训练类
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
                timestamp = datetime.utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')

            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            summary_loss, summary_score = self.validation(validation_loader)
            self.log(f'[RESULT]: Validation. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, summary_score: {summary_score.avg:.5f}, time: {(time.time() - t):.5f}')

            if summary_score.avg > self.best_summary_score:
                self.best_summary_score = summary_score.avg
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint.bin')
                for path in sorted(glob(f'{self.base_dir}/best-checkpoint.bin'))[:-3]:
                    os.remove(path)

            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_score.avg)

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
                targets2['img_size'] = torch.tensor([(1024, 1024) for target in targets]).to(
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
            targets2['img_size'] = torch.tensor([(1024, 1024) for target in targets]).to(self.device).float()

            output = self.model(images, targets2)
            loss = output['loss'] / self.config.grad_accumulation_steps
            loss.backward()

            summary_loss.update(loss.detach().item(), batch_size)

            # Gradient accumulation
            if (step + 1) % self.config.grad_accumulation_steps == 0:
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
            'best_summary_score': self.summary_score,
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