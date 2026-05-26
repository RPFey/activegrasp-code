import os
import re
import trimesh
import mcubes
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from torch.utils import tensorboard
from pathlib import Path

from .VGN_architecture import ConvNet
from .GIGA_architecture import get_GIGA_model
from .ACE_architecture import ACENet

from .misc import EasyDict
from .timer import CudaTimer


class TrainingWriter():
    def __init__(self, log_dir):
        """write training log to tensorboard
        """
        train_path = log_dir / "training"
        self.writer = tensorboard.SummaryWriter(train_path, flush_secs=60)

    def log(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def log_images(self, tag, images, step, **kwargs):
        self.writer.add_images(tag, images, step, **kwargs)

class ValidationWriter():
    def __init__(self, log_dir):
        """write validation log to tensorboard
        """
        val_path = log_dir / "validation"
        self.writer = tensorboard.SummaryWriter(val_path, flush_secs=60)

    def reset(self, loss_dict):
        # sum, count, average
        self.TP, self.FP, self.FN, self.TN = 0, 0, 0, 0
        self.count = 0
        self.loss_dict_sum = loss_dict
        for key in self.loss_dict_sum.keys():
            self.loss_dict_sum[key] = 0

    def update(self, batch_dict, prediction, loss_dict):
        if "loss_dict_sum" not in dir(self):
            self.reset(loss_dict)
        # save grasp result
        grasp_result = batch_dict.grasp_label
        grasp_result_pred = torch.round(prediction.grasp_label).to(grasp_result)
        for k, v in loss_dict.items():
            self.loss_dict_sum[k] += v
        for g_p, g in zip(grasp_result_pred, grasp_result):
            if g_p == 1 and g == 1:
                self.TP += 1
            elif g_p == 1 and g == 0:
                self.FP += 1
            elif g_p == 0 and g == 1:
                self.FN += 1
            else:
                self.TN += 1
        self.count += 1

    def log(self, iteration_count):
        # accuracy, precision, recall
        if self.TP + self.FN + self.FP + self.TN != 0:
            self.writer.add_scalar("Grasp/accuracy", (self.TP + self.TN) / (self.TP + self.FN + self.FP + self.TN), iteration_count)
        if self.TP + self.FP != 0:
            self.writer.add_scalar("Grasp/precision", self.TP / (self.TP + self.FP), iteration_count)
        if self.TP + self.FN != 0:
            self.writer.add_scalar("Grasp/recall", self.TP / (self.TP + self.FN), iteration_count)
        # average loss
        for k, v in self.loss_dict_sum.items():
            self.loss_dict_sum[k] = v / self.count
            self.writer.add_scalar('Loss/' + k, self.loss_dict_sum[k], iteration_count)
        del self.loss_dict_sum  # reset


class Runner():
    '''主要包含训练阶段的网络构建、训练、验证、可视化、保存等功能
    '''
    def __init__(self, cfg):
        self.device = torch.device(cfg.device)
        self.cfg = cfg
        self.timer = CudaTimer(enabled=False)

        # build the network or load from checkpoint
        if cfg.load_path == '':
            self.net = self.build_network(cfg)
            self.epoch = 0
            print('Built network from scratch.')
        else:
            self.net = self.load_network(cfg)
            print(f'Loaded network from {cfg.load_path}.')

    def load_network(self, cfg):
        """Construct the neural network, load parameters from the specified file.
        """
        path = cfg.load_path
        net = self.build_network(cfg)
        checkpoint = torch.load(path, map_location=self.device)
        net.load_state_dict(checkpoint['net'])
        self.epoch = checkpoint['epoch']
        return net

    def build_network(self, cfg):
        """Construct the neural network from scratch.
        """
        net_type = cfg.net
        models = {
            "vgn": ConvNet,
            "giga": get_GIGA_model,
            "ace": ACENet,
        }
        return models[net_type.lower()](cfg).to(self.device)

    def save_network(self, net, path, epoch, name=''):
        """save parameters to [log_dir]/ckpts/[ckpt_epoch_n].pt
        允许 epoch 中间保存
        """
        time_stamp = datetime.now().strftime("%y-%m-%d-%H-%M")
        checkpoint = {
            'net': net.state_dict(),
            'time': time_stamp,
            'epoch': epoch
        }
        ckpt_path = path / "ckpts"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, ckpt_path / f"ckpt_epoch_{epoch:02d}{name}.pt")

    def train(self, train_loader, val_loader):
        # define optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.cfg.learning_rate)

        # 判断是否指定了 hydra=debug (对应 config 文件会新建并更改当前目录)
        self.debug = not bool(re.search(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', os.getcwd()))

        # define tensorboard writer
        if not self.debug:
            self.logdir = Path(os.getcwd())
            print(f"logdir: {self.logdir}")
            self.training_writer = TrainingWriter(self.logdir)
            self.validation_writer = ValidationWriter(self.logdir)
        else:
            print('\033[1;31m' + 'Debug mode. Logging is disabled.' + '\033[0m')

        # deal with fractional epochs
        if type(self.cfg.epochs) == float:
            self.epochs_fractional = self.cfg.epochs - int(self.cfg.epochs)
            self.cfg.epochs = int(self.cfg.epochs) + 1
        else:
            self.epochs_fractional = 1e6

        print('\033[1;34m' + 'Start training...' + '\033[0m')
        iteration_count = 0
        total_epochs = self.cfg.epochs + self.epoch  # 针对从 ckpt 继续训练的情况
        for _ in range(self.cfg.epochs):
            # progress bar
            loop = tqdm(train_loader, total=len(train_loader))
            prefix = self.logdir.name[-8:] if not self.debug else "Debug"
            loop.set_description(f"[{prefix}] Epoch {self.epoch + 1}/{total_epochs}")

            # read training batch
            for batch_idx, batch in enumerate(loop):
                self.timer.reset()
                iteration_count = self.epoch * len(train_loader) + batch_idx  # batch count
                self.net.train()
                self.optimizer.zero_grad()
                # forward
                batch = EasyDict(batch)
                batch.to(self.device)
                prediction = self.net(batch, iteration_count)
                self.timer.check("forward")
                # backward
                loss_dict = self.net.compute_loss(batch, prediction)
                self.timer.check("compute_loss")
                loss_dict.loss_all.backward()
                self.optimizer.step()
                self.timer.check("backward")
                # update progress bar
                loop.set_postfix({'loss': f'{loss_dict.loss_all.item():.2f}'})
                # log during training
                if not self.debug:
                    # log values
                    if batch_idx % self.cfg.log_interval == 0:
                        for k, v in loss_dict.items():
                            self.training_writer.log('Loss/' + k, v, iteration_count)
                        if 's_val' in prediction:
                            self.training_writer.log('Statistics/s_val', prediction.s_val, iteration_count)
                            self.training_writer.log('Statistics/cos_anneal_ratio', prediction.cos_anneal_ratio, iteration_count)
                            self.training_writer.log('Statistics/sample_depth_near_input_range', prediction.sample_depth_near_input_range, iteration_count)
                        self.timer.check("log values")
                    # save network
                    if batch_idx % self.cfg.save_interval == 0:
                        self.save_network(self.net, self.logdir, self.epoch, name=f'_{batch_idx:05d}')
                        self.timer.check("save network")
                    # extra loop for 4 full image logging
                    if batch_idx % self.cfg.full_image_interval == 0 and \
                        'rendered_depth' in prediction and \
                            self.cfg.batch_size >= 4:
                        self.validate_depth_image(batch, iteration_count)
                        self.timer.check("validate depth image")
                    # extra loop for mesh logging
                    if batch_idx % self.cfg.mesh_interval == 0 and \
                        'rendered_depth' in prediction:
                        self.validate_mesh(batch, iteration_count)
                        self.timer.check("validate mesh")

                # print time
                if self.timer.enabled:
                    tsum = 0.
                    tstr = "Timings: "
                    for tname, tval in self.timer.timings.items():
                        tstr += f"{tname}={tval:.1f}ms  "
                        tsum += tval
                    tstr += f"tot={tsum:.1f}ms"
                    print(tstr)

                # fractional part
                if self.epoch == self.cfg.epochs - 1 and batch_idx > self.epochs_fractional * len(train_loader):
                    break

            loop.close()

            # validation, when an epoch is finished
            self.net.eval()
            # with torch.no_grad():
            loop = tqdm(val_loader, total=len(val_loader))
            loop.set_description("Validation")
            for batch_idx, batch in enumerate(loop):
                batch = EasyDict(batch)
                batch.to(self.device)
                prediction = self.net(batch, iteration_count)
                loss_dict = self.net.compute_loss(batch, prediction)
                if not self.debug:
                    self.validation_writer.update(batch.detach(), prediction.detach(), loss_dict.detach())

            # log during validation, contains loss, accuracy, precision, recall
            loop.close()
            if not self.debug:
                self.validation_writer.log(iteration_count)

            # end of epoch
            if not self.debug:
                self.save_network(self.net, self.logdir, self.epoch, name='_end')
            self.epoch += 1

        print('\033[1;34m' + 'Training finished!' + '\033[0m')

    def validate_depth_image(self, batch, iteration_count, new_resolution=[256, 188]):
        print('\033[1;33m' + 'Validating images...' + '\033[0m')
        W, H = new_resolution
        render_args = {'is_selected_rays': False, 'new_resolution': [W, H], 'render_normal': True}
        for k, v in batch.items():  # select 4 scenes
            batch[k] = v[:4]
        prediction = self.net.render_full_image(batch, iteration_count, grasp_branch=False, **render_args)

        def concat_4_images(imgs):
            '''拼接图片
            4 * H * W -> 2H * 2W
            '''
            return torch.cat([torch.cat([imgs[0], imgs[1]], dim=1),
                              torch.cat([imgs[2], imgs[3]], dim=1)], dim=0)

        img_gt = batch.depth_img[:, 0, :]
        img_pred = prediction.rendered_depth.reshape(-1, H, W)
        img_normal = prediction.normal_imgs.reshape(-1, H, W, 3)

        # 根据 valid_ray_mask 处理 img_pred (转彩色图并标记 valid ray)
        img_pred = img_pred.unsqueeze(-1).repeat(1, 1, 1, 3)
        mask = prediction.valid_ray_mask.reshape(-1, H, W)
        # img_pred[..., 2] += (mask * 1.0)  # 只增加 B 通道颜色
        img_pred += (~mask).unsqueeze(-1).repeat(1, 1, 1, 3).int() * 2.0  # 全白
        img_pred = torch.clamp(img_pred, -1.0, 1.0)

        # 根据 valid_ray_mask 处理 img_normal
        img_normal += (~mask).unsqueeze(-1).repeat(1, 1, 1, 3).int() * 2.0  # 全白
        img_normal = torch.clamp(img_normal, -1.0, 1.0)
        img_normal = img_normal * 127 + 128  # [-1, 1] -> [0, 255]
        img_normal = img_normal.int()

        self.training_writer.log_images("Images/img_gt", concat_4_images(img_gt), iteration_count, dataformats='HW')
        self.training_writer.log_images("Images/img_pred", concat_4_images(img_pred), iteration_count, dataformats='HWC')
        self.training_writer.log_images("Images/img_normal", concat_4_images(img_normal), iteration_count, dataformats='HWC')
        print(f'Images saved to tensorboard. (iteration_count: {iteration_count})')

    def validate_mesh(self, batch, iteration_count, resolution=512, threshold=0.0):
        '''导出整个场景(立方体)的 mesh
        code from NeuS/exp_runner.py
        单卡显存大约支持 128**3 分辨率, 这里分 batch 操作
        注意在最终导出时的分辨率尽可能大, ref: https://github.com/Totoro97/NeuS/issues/5
        '''
        print('\033[1;33m' + 'Validating mesh...' + '\033[0m')
        meshes_path = self.logdir / "meshes"
        meshes_path.mkdir(parents=True, exist_ok=True)
        for k, v in batch.items():  # select 1 scenes
            batch[k] = v[:1]  # 保留维度

        prediction = self.net.extract_geometry(batch, iteration_count, resolution)
        volume = prediction.sdf.reshape([resolution, resolution, resolution]).detach().cpu().numpy()
        vertices, triangles = mcubes.marching_cubes(volume, threshold)
        print(f'Got {len(vertices)} vertices and {len(triangles)} triangles. (iteration_count: {iteration_count})')

        # vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]  # 缩放到 [-1.01, 1.01]

        mesh = trimesh.Trimesh(vertices, triangles)  # 生成 mesh
        mesh.export(meshes_path / f'{iteration_count:0>8d}.ply')
        print(f'Mesh saved to file. (iteration_count: {iteration_count})')

    def predict_grasp(self, batch):
        '''预测抓取点
        tsdf: [1, resolution**3]
        pos_list: [n, 1, 3]
        extrinsics: [1, 7]
        intrinsics: [1, 6]
        目前只支持一个 tsdf 预测一个抓取点, 还是分 batch 操作
        '''
        batch.to(self.device)
        return self.net.predict_grasp(batch)
