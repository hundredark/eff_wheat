import os
import time
import torch
import datetime
from .utils import AverageMeter

# 模型训练类
class Fitter:
    # 初始化
    def __init__(self, model, device, config):
        # 模型各类参数
        self.config = config
        # epoch的初始值
        self.epoch = 0
        # 保存模型的地址
        self.base_dir = f'./{config.folder}'
        # 如果不存在则新增对应目录
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        # 打印 log 的地址，保存模型的训练信息
        self.log_path = f'{self.base_dir}/log.txt'
        # 设定一个比较大的 best_summary_loss 值，为了保存最优的模型
        self.best_summary_loss = 10 ** 5

        self.model = model
        self.device = device

        # 确定哪些值需要加weight_decay （正则项值）
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        # 优化算法使用 RMS
        # 学习策略
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(self.optimizer, **config.scheduler_params)
        self.log(f'Fitter prepared. Device is {self.device}')

    # 模型训练
    def fit(self, train_loader, validation_loader):
        # 训练 n_epochs 次
        for e in range(self.config.n_epochs):
            # 在日志中记录信息
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]['lr']
                timestamp = datetime.datetime.now().utcnow().isoformat()
                self.log(f'\n{timestamp}\nLR: {lr}')

            # 开始训练一个 epoch
            t = time.time()
            summary_loss = self.train_one_epoch(train_loader)

            self.log(
                f'[RESULT]: Train. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            self.save(f'{self.base_dir}/last-checkpoint.bin')

            t = time.time()
            # 得到验证集合的损失
            summary_loss = self.validation(validation_loader)

            self.log(
                f'[RESULT]: Val. Epoch: {self.epoch}, summary_loss: {summary_loss.avg:.5f}, time: {(time.time() - t):.5f}')
            # 如果验证的损失比最优的好，则保存最优的模型
            if summary_loss.avg < self.best_summary_loss:
                self.best_summary_loss = summary_loss.avg
                # 切换到模型的验证模式
                self.model.eval()
                self.save(f'{self.base_dir}/best-checkpoint-{str(self.epoch).zfill(3)}epoch.bin')

            # 执行学习策略（相当于 callback 函数）
            if self.config.validation_scheduler:
                self.scheduler.step(metrics=summary_loss.avg)

            self.epoch += 1

    # 获得验证集的结果
    def validation(self, val_loader):
        # 切换到模型的验证模式
        self.model.eval()
        # 初始化损失计算器
        summary_loss = AverageMeter()
        t = time.time()
        # 开始遍历验证集
        for step, (images, targets, image_ids) in enumerate(val_loader):
            if self.config.verbose:
                if step % self.config.verbose_step == 0:
                    print(
                        f'Val Step {step}/{len(val_loader)}, ' + \
                        f'summary_loss: {summary_loss.avg:.5f}, ' + \
                        f'time: {(time.time() - t):.5f}', end='\r'
                    )
            with torch.no_grad():
                images = torch.stack(images)
                batch_size = images.shape[0]
                images = images.to(self.device).float()
                boxes = [target['boxes'].to(self.device).float() for target in targets]
                labels = [target['labels'].to(self.device).float() for target in targets]

                loss, _, _ = self.model(images, boxes, labels)
                summary_loss.update(loss.detach().item(), batch_size)

        return summary_loss

    def train_one_epoch(self, train_loader):
        # 切换到模型的训练模式
        self.model.train()
        # 初始化损失计算器
        summary_loss = AverageMeter()
        t = time.time()
        # 开始遍历训练集
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
            boxes = [target['boxes'].to(self.device).float() for target in targets]
            labels = [target['labels'].to(self.device).float() for target in targets]

            self.optimizer.zero_grad()
            # 前向传播计算 loss
            loss, _, _ = self.model(images, boxes, labels)
            # 反向传播计算 grad
            loss.backward()
            # 更新 loss
            summary_loss.update(loss.detach().item(), batch_size)
            # 根据优化算法更新 parameter
            self.optimizer.step()
            # 执行学习策略
            if self.config.step_scheduler:
                self.scheduler.step()

        return summary_loss

    # 保存模型
    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_summary_loss,
            'epoch': self.epoch,
        }, path)

    # 加载模型
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_summary_loss = checkpoint['best_summary_loss']
        self.epoch = checkpoint['epoch'] + 1

    # 打印日志
    def log(self, message):
        if self.config.verbose:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')


