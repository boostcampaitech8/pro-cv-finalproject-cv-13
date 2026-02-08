# append by LDH
from __future__ import annotations

import numpy as np
import torch
from torch import autocast
from torch import distributed as dist

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.gsl_gpu import GeneralizedSurfaceLoss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.utilities.collate_outputs import collate_outputs


class nnUNetTrainer_gsl(nnUNetTrainer):
    """
    Minimal custom trainer that adds GSL on top of the base loss.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.gsl_loss = None  # -> initialize
        self.gsl_enabled = True
        self.gsl_lambda_max = 0.1
        self.gsl_warmup_epochs = 50
        self.gsl_include_background = False
        self.gsl_eps = 1e-6
        self.gsl_class_indices = [1, 2, 3, 12, 13]
        # Speed up early feedback; reduce per-epoch iterations
        self.num_iterations_per_epoch = 250

    def initialize(self):
        super().initialize()
        if self.gsl_loss is None:
            self.gsl_loss = GeneralizedSurfaceLoss(
                include_background=self.gsl_include_background,
                ignore_label=self.label_manager.ignore_label,
                eps=self.gsl_eps,
                class_indices=self.gsl_class_indices,
            )

    def _get_gsl_lambda(self) -> float:
        if not self.gsl_enabled:
            return 0.0
        if self.gsl_warmup_epochs <= 0:
            return float(self.gsl_lambda_max)
        warmup = min(1.0, self.current_epoch / float(self.gsl_warmup_epochs))
        return float(self.gsl_lambda_max) * warmup

    def _compute_gsl(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.gsl_enabled or self.gsl_loss is None:
            return output.new_zeros(())
        if self.label_manager.has_regions:
            probs = torch.sigmoid(output)
        else:
            probs = torch.softmax(output, dim=1)
        return self.gsl_loss(probs, target)

    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            base_l = self.loss(output, target)

            if self.enable_deep_supervision:
                output_gsl = output[0]
                target_gsl = target[0] if isinstance(target, list) else target
            else:
                output_gsl = output
                target_gsl = target

            gsl_l = self._compute_gsl(output_gsl, target_gsl)
            lambda_gsl = self._get_gsl_lambda()
            l = base_l + (lambda_gsl * gsl_l)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()

        return {
            'loss': l.detach().cpu().numpy(),
            'base_loss': base_l.detach().cpu().numpy(),
            'gsl_loss': gsl_l.detach().cpu().numpy(),
            'lambda_gsl': lambda_gsl,
        }

    def on_train_epoch_end(self, train_outputs):
        outputs = collate_outputs(train_outputs)

        if self.is_ddp:
            losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(losses_tr, outputs['loss'])
            loss_here = np.vstack(losses_tr).mean()

            base_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(base_losses_tr, outputs['base_loss'])
            base_loss_here = np.vstack(base_losses_tr).mean()

            gsl_losses_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gsl_losses_tr, outputs['gsl_loss'])
            gsl_loss_here = np.vstack(gsl_losses_tr).mean()

            lambda_gsl_tr = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(lambda_gsl_tr, outputs['lambda_gsl'])
            lambda_gsl_here = np.mean(lambda_gsl_tr)
        else:
            loss_here = np.mean(outputs['loss'])
            base_loss_here = np.mean(outputs['base_loss'])
            gsl_loss_here = np.mean(outputs['gsl_loss'])
            lambda_gsl_here = np.mean(outputs['lambda_gsl'])

        self.logger.log('train_losses', loss_here, self.current_epoch)
        self.logger.log('train_base_loss', base_loss_here, self.current_epoch)
        self.logger.log('train_gsl_loss', gsl_loss_here, self.current_epoch)
        self.logger.log('train_lambda_gsl', lambda_gsl_here, self.current_epoch)

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            base_l = self.loss(output, target)

            if self.enable_deep_supervision:
                output_gsl = output[0]
                target_gsl = target[0] if isinstance(target, list) else target
            else:
                output_gsl = output
                target_gsl = target

            gsl_l = self._compute_gsl(output_gsl, target_gsl)
            lambda_gsl = self._get_gsl_lambda()
            l = base_l + (lambda_gsl * gsl_l)

        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {
            'loss': l.detach().cpu().numpy(),
            'base_loss': base_l.detach().cpu().numpy(),
            'gsl_loss': gsl_l.detach().cpu().numpy(),
            'lambda_gsl': lambda_gsl,
            'tp_hard': tp_hard,
            'fp_hard': fp_hard,
            'fn_hard': fn_hard,
        }

    def on_validation_epoch_end(self, val_outputs):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        if self.is_ddp:
            world_size = dist.get_world_size()

            tps = [None for _ in range(world_size)]
            dist.all_gather_object(tps, tp)
            tp = np.vstack([i[None] for i in tps]).sum(0)

            fps = [None for _ in range(world_size)]
            dist.all_gather_object(fps, fp)
            fp = np.vstack([i[None] for i in fps]).sum(0)

            fns = [None for _ in range(world_size)]
            dist.all_gather_object(fns, fn)
            fn = np.vstack([i[None] for i in fns]).sum(0)

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated['loss'])
            loss_here = np.vstack(losses_val).mean()

            base_losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(base_losses_val, outputs_collated['base_loss'])
            base_loss_here = np.vstack(base_losses_val).mean()

            gsl_losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(gsl_losses_val, outputs_collated['gsl_loss'])
            gsl_loss_here = np.vstack(gsl_losses_val).mean()

            lambda_gsl_val = [None for _ in range(world_size)]
            dist.all_gather_object(lambda_gsl_val, outputs_collated['lambda_gsl'])
            lambda_gsl_here = np.mean(lambda_gsl_val)
        else:
            loss_here = np.mean(outputs_collated['loss'])
            base_loss_here = np.mean(outputs_collated['base_loss'])
            gsl_loss_here = np.mean(outputs_collated['gsl_loss'])
            lambda_gsl_here = np.mean(outputs_collated['lambda_gsl'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)
        self.logger.log('val_base_loss', base_loss_here, self.current_epoch)
        self.logger.log('val_gsl_loss', gsl_loss_here, self.current_epoch)
        self.logger.log('val_lambda_gsl', lambda_gsl_here, self.current_epoch)

    def on_epoch_end(self):
        super().on_epoch_end()
        self.print_to_log_file('train_base_loss',
                               np.round(self.logger.my_fantastic_logging['train_base_loss'][-1], decimals=4))
        self.print_to_log_file('train_gsl_loss',
                               np.round(self.logger.my_fantastic_logging['train_gsl_loss'][-1], decimals=4))
        self.print_to_log_file('train_lambda_gsl',
                               np.round(self.logger.my_fantastic_logging['train_lambda_gsl'][-1], decimals=4))
        self.print_to_log_file('val_base_loss',
                               np.round(self.logger.my_fantastic_logging['val_base_loss'][-1], decimals=4))
        self.print_to_log_file('val_gsl_loss',
                               np.round(self.logger.my_fantastic_logging['val_gsl_loss'][-1], decimals=4))
        self.print_to_log_file('val_lambda_gsl',
                               np.round(self.logger.my_fantastic_logging['val_lambda_gsl'][-1], decimals=4))
