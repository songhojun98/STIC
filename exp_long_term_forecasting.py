from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _slice_outputs(self, outputs):
        f_dim = -1 if self.args.features == 'MS' else 0
        if isinstance(outputs, dict):
            sliced = {}
            debug_feature_keys = {
                'gate_input',
                'gate_stats',
                'gate_feat_h',
                'gate_feat_c',
            }
            for key, value in outputs.items():
                if not torch.is_tensor(value):
                    sliced[key] = value
                    continue
                value = value[:, -self.args.pred_len:, ...]
                if key in debug_feature_keys:
                    sliced[key] = value
                    continue
                if value.dim() == 3 and value.shape[-1] > 1:
                    sliced[key] = value[:, :, f_dim:]
                else:
                    sliced[key] = value
            return sliced
        return outputs[:, -self.args.pred_len:, f_dim:]

    def _slice_target(self, batch_y):
        f_dim = -1 if self.args.features == 'MS' else 0
        return batch_y[:, -self.args.pred_len:, f_dim:]

    def _resolve_target_index(self):
        if self.args.features == 'S':
            return 0
        configured_target_index = getattr(self.args, 'stic_target_index', -1)
        if configured_target_index >= 0:
            return min(int(configured_target_index), max(int(self.args.enc_in) - 1, 0))
        return max(int(self.args.enc_in) - 1, 0)

    def _split_target_context(self, batch_x):
        target_index = self._resolve_target_index()
        target = batch_x[..., target_index:target_index + 1]
        if batch_x.size(-1) == 1:
            context = batch_x[..., :0]
        else:
            context = torch.cat(
                (batch_x[..., :target_index], batch_x[..., target_index + 1:]),
                dim=-1,
            )
        return target, context

    def _merge_target_context(self, target, context):
        target_index = self._resolve_target_index()
        if context.size(-1) == 0:
            return target
        context_before = context[..., :target_index]
        context_after = context[..., target_index:]
        return torch.cat((context_before, target, context_after), dim=-1)

    def _apply_context_corruption(self, batch_x):
        corruption_mode = getattr(self.args, 'stic_context_corruption_mode', 'none')
        corruption_prob = getattr(self.args, 'stic_context_corruption_prob', 0.0)
        if (
            self.args.model != 'STIC'
            or corruption_mode == 'none'
            or corruption_prob <= 0
            or batch_x.size(-1) <= 1
        ):
            return batch_x, False, 'none'

        if torch.rand(1, device=batch_x.device).item() > corruption_prob:
            return batch_x, False, 'none'

        target, context = self._split_target_context(batch_x)
        if context.size(-1) == 0:
            return batch_x, False, 'none'

        applied_mode = corruption_mode
        if corruption_mode == 'mixed':
            candidates = ['shuffle', 'swap', 'dropout']
            applied_mode = candidates[
                int(torch.randint(len(candidates), (1,), device=batch_x.device).item())
            ]

        if applied_mode == 'shuffle':
            channel_perm = torch.randperm(context.size(-1), device=context.device)
            corrupted_context = context[..., channel_perm]
        elif applied_mode == 'swap':
            if context.size(0) == 1:
                return batch_x, False, 'none'
            batch_perm = torch.randperm(context.size(0), device=context.device)
            identity_perm = torch.arange(context.size(0), device=context.device)
            if torch.equal(batch_perm, identity_perm):
                batch_perm = torch.roll(batch_perm, shifts=1)
            corrupted_context = context[batch_perm]
        elif applied_mode == 'dropout':
            keep_prob = 1.0 - getattr(self.args, 'stic_context_dropout_p', 0.3)
            keep_prob = min(max(keep_prob, 0.0), 1.0)
            if keep_prob == 0:
                corrupted_context = torch.zeros_like(context)
            else:
                keep_mask = (torch.rand_like(context) < keep_prob).to(context.dtype)
                corrupted_context = context * keep_mask
        else:
            return batch_x, False, 'none'

        return self._merge_target_context(target, corrupted_context), True, applied_mode

    @staticmethod
    def _init_gate_stats():
        prefixes = ('clean', 'corrupt')
        stats = {
            'corruption_mode_counts': {},
            'paired_batches': 0,
            'paired_gate_gap_sum': 0.0,
            'paired_gate_win_rate_sum': 0.0,
        }
        for prefix in prefixes:
            stats[f'{prefix}_batches'] = 0
            stats[f'{prefix}_positions'] = 0
            stats[f'{prefix}_gate_mean_sum'] = 0.0
            stats[f'{prefix}_gate_std_sum'] = 0.0
            stats[f'{prefix}_horizon_gate_sum'] = None
            stats[f'{prefix}_utility_mean_sum'] = 0.0
            stats[f'{prefix}_gate_utility_corr_sum'] = 0.0
            stats[f'{prefix}_gate_utility_corr_count'] = 0
            stats[f'{prefix}_context_better_gate_sum'] = 0.0
            stats[f'{prefix}_context_better_count'] = 0
            stats[f'{prefix}_history_better_gate_sum'] = 0.0
            stats[f'{prefix}_history_better_count'] = 0
        return stats

    def _update_gate_stats(self, stats, outputs, batch_y, corrupted_batch, corruption_mode='none'):
        if not isinstance(outputs, dict) or 'gate' not in outputs:
            return

        prefix = 'corrupt' if corrupted_batch else 'clean'
        gate = outputs['gate'].detach()
        stats[f'{prefix}_batches'] += 1
        stats[f'{prefix}_positions'] += gate.numel()
        stats[f'{prefix}_gate_mean_sum'] += gate.mean().item()
        stats[f'{prefix}_gate_std_sum'] += gate.std(unbiased=False).item()
        horizon_gate = gate.mean(dim=0).squeeze(-1).cpu()
        if stats[f'{prefix}_horizon_gate_sum'] is None:
            stats[f'{prefix}_horizon_gate_sum'] = torch.zeros_like(horizon_gate)
        stats[f'{prefix}_horizon_gate_sum'] += horizon_gate
        if corrupted_batch and corruption_mode != 'none':
            mode_counts = stats['corruption_mode_counts']
            mode_counts[corruption_mode] = mode_counts.get(corruption_mode, 0) + 1

        if 'pred_h' not in outputs or 'pred_c' not in outputs:
            return

        err_h = (outputs['pred_h'].detach() - batch_y.detach()).pow(2)
        err_c = (outputs['pred_c'].detach() - batch_y.detach()).pow(2)
        utility = err_h - err_c
        stats[f'{prefix}_utility_mean_sum'] += utility.mean().item()
        gate_flat = gate.reshape(-1)
        utility_flat = utility.reshape(-1)
        gate_std = gate_flat.std(unbiased=False)
        utility_std = utility_flat.std(unbiased=False)
        if gate_flat.numel() > 1 and gate_std.item() > 0 and utility_std.item() > 0:
            centered_gate = gate_flat - gate_flat.mean()
            centered_utility = utility_flat - utility_flat.mean()
            corr = (centered_gate * centered_utility).mean() / (
                gate_std * utility_std + 1e-8
            )
            stats[f'{prefix}_gate_utility_corr_sum'] += corr.item()
            stats[f'{prefix}_gate_utility_corr_count'] += 1
        context_better = err_c < err_h
        history_better = err_h <= err_c

        if context_better.any():
            stats[f'{prefix}_context_better_gate_sum'] += gate[context_better].sum().item()
            stats[f'{prefix}_context_better_count'] += int(context_better.sum().item())

        if history_better.any():
            stats[f'{prefix}_history_better_gate_sum'] += gate[history_better].sum().item()
            stats[f'{prefix}_history_better_count'] += int(history_better.sum().item())

    def _update_paired_gate_stats(self, stats, clean_outputs, corrupt_outputs):
        if (
            not isinstance(clean_outputs, dict)
            or not isinstance(corrupt_outputs, dict)
            or 'gate' not in clean_outputs
            or 'gate' not in corrupt_outputs
        ):
            return

        clean_gate = clean_outputs['gate'].detach()
        corrupt_gate = corrupt_outputs['gate'].detach()
        stats['paired_batches'] += 1
        stats['paired_gate_gap_sum'] += (clean_gate.mean() - corrupt_gate.mean()).item()
        stats['paired_gate_win_rate_sum'] += (clean_gate > corrupt_gate).float().mean().item()

    @staticmethod
    def _safe_average(total, count):
        return total / count if count else None

    @staticmethod
    def _format_horizon_profile(horizon_gate_sum, batch_count):
        if horizon_gate_sum is None or batch_count == 0:
            return None
        horizon_mean = horizon_gate_sum / batch_count
        segment_count = min(4, horizon_mean.numel())
        segment_size = max((horizon_mean.numel() + segment_count - 1) // segment_count, 1)
        segment_means = []
        for start in range(0, horizon_mean.numel(), segment_size):
            segment = horizon_mean[start:start + segment_size]
            if segment.numel() == 0:
                continue
            segment_means.append(segment.mean().item())
        return ','.join(f'{value:.3f}' for value in segment_means)

    def _format_gate_stats(self, stats):
        parts = []
        for prefix in ('clean', 'corrupt'):
            batch_count = stats[f'{prefix}_batches']
            if batch_count == 0:
                continue
            gate_mean = self._safe_average(stats[f'{prefix}_gate_mean_sum'], batch_count)
            gate_std = self._safe_average(stats[f'{prefix}_gate_std_sum'], batch_count)
            utility_mean = self._safe_average(stats[f'{prefix}_utility_mean_sum'], batch_count)
            gate_utility_corr = self._safe_average(
                stats[f'{prefix}_gate_utility_corr_sum'],
                stats[f'{prefix}_gate_utility_corr_count'],
            )
            horizon_profile = self._format_horizon_profile(
                stats[f'{prefix}_horizon_gate_sum'],
                batch_count,
            )
            parts.append(f'{prefix}_gate_mean={gate_mean:.4f}')
            parts.append(f'{prefix}_gate_std={gate_std:.4f}')
            if horizon_profile is not None:
                parts.append(f'{prefix}_gate_hq={horizon_profile}')
            if utility_mean is not None:
                parts.append(f'{prefix}_utility_mean={utility_mean:.4f}')
            if gate_utility_corr is not None:
                parts.append(f'{prefix}_gate_utility_corr={gate_utility_corr:.4f}')

            context_better_gate = self._safe_average(
                stats[f'{prefix}_context_better_gate_sum'],
                stats[f'{prefix}_context_better_count'],
            )
            history_better_gate = self._safe_average(
                stats[f'{prefix}_history_better_gate_sum'],
                stats[f'{prefix}_history_better_count'],
            )
            context_better_rate = self._safe_average(
                stats[f'{prefix}_context_better_count'],
                stats[f'{prefix}_positions'],
            )
            if context_better_rate is not None:
                parts.append(f'{prefix}_ctx_better_rate={context_better_rate:.4f}')
            if context_better_gate is not None:
                parts.append(f'{prefix}_gate|ctx_better={context_better_gate:.4f}')
            if history_better_gate is not None:
                parts.append(f'{prefix}_gate|hist_better={history_better_gate:.4f}')
            if context_better_gate is not None and history_better_gate is not None:
                parts.append(
                    f'{prefix}_gate_alignment={context_better_gate - history_better_gate:.4f}'
                )
        paired_batches = stats['paired_batches']
        if paired_batches:
            pair_gap = self._safe_average(stats['paired_gate_gap_sum'], paired_batches)
            pair_win_rate = self._safe_average(stats['paired_gate_win_rate_sum'], paired_batches)
            parts.append(f'paired_gate_gap={pair_gap:.4f}')
            parts.append(f'paired_gate_win_rate={pair_win_rate:.4f}')
        if stats['corruption_mode_counts']:
            mode_summary = ','.join(
                f'{name}:{count}'
                for name, count in sorted(stats['corruption_mode_counts'].items())
            )
            parts.append(f'corrupt_modes={mode_summary}')
        return ' | '.join(parts)

    def _compute_gate_target(self, utility):
        gate_target_mode = getattr(self.args, 'stic_gate_target_mode', 'soft')
        utility = utility.detach()
        if gate_target_mode == 'soft':
            tau = max(getattr(self.args, 'stic_gate_soft_tau', 0.02), 1e-6)
            return torch.sigmoid(utility / tau)
        return (utility > 0).float()

    def _compute_loss(self, outputs, batch_y, criterion, corrupted_batch=False):
        if not isinstance(outputs, dict):
            pred_loss = criterion(outputs, batch_y)
            return pred_loss, {"pred_loss": pred_loss}

        pred = outputs['pred']
        pred_loss = criterion(pred, batch_y)
        total_loss = pred_loss
        loss_terms = {"pred_loss": pred_loss}

        aux_weight = getattr(self.args, 'stic_aux_weight', 0.1)
        gate_weight = getattr(self.args, 'stic_gate_weight', 0.1)
        corruption_gate_weight = getattr(
            self.args, 'stic_context_corruption_gate_weight', 0.1
        )
        aux_enabled = outputs.get('aux_loss_enabled', True)
        gate_trainable = outputs.get('gate_trainable', True)

        if 'pred_h' in outputs and 'pred_c' in outputs and aux_enabled and aux_weight > 0:
            history_aux_loss = criterion(outputs['pred_h'], batch_y)
            context_aux_loss = criterion(outputs['pred_c'], batch_y)
            if corrupted_batch:
                corrupt_context_aux_weight = max(
                    0.0, getattr(self.args, 'stic_corrupt_context_aux_weight', 0.0)
                )
                aux_normalizer = 1.0 + corrupt_context_aux_weight
                aux_loss = (
                    history_aux_loss + corrupt_context_aux_weight * context_aux_loss
                ) / aux_normalizer
            else:
                aux_loss = 0.5 * (history_aux_loss + context_aux_loss)
            total_loss = total_loss + aux_weight * aux_loss
            loss_terms['aux_loss'] = aux_loss
            loss_terms['history_aux_loss'] = history_aux_loss
            loss_terms['context_aux_loss'] = context_aux_loss

        if (
            'gate' in outputs
            and 'pred_h' in outputs
            and 'pred_c' in outputs
            and gate_trainable
            and gate_weight > 0
        ):
            gate = outputs['gate'].clamp(min=1e-6, max=1 - 1e-6)
            err_h = (outputs['pred_h'] - batch_y).pow(2)
            err_c = (outputs['pred_c'] - batch_y).pow(2)
            utility = err_h - err_c
            gate_target = self._compute_gate_target(utility)
            if getattr(self.args, 'stic_gate_target_mode', 'soft') == 'soft':
                gate_loss = F.mse_loss(gate, gate_target)
            else:
                gate_loss = F.binary_cross_entropy(gate, gate_target)
            outputs['gate_targets'] = gate_target
            total_loss = total_loss + gate_weight * gate_loss
            loss_terms['gate_loss'] = gate_loss

        if 'gate' in outputs and gate_trainable and corrupted_batch and corruption_gate_weight > 0:
            corrupt_gate_target = torch.zeros_like(outputs['gate'])
            corrupt_gate_loss = F.binary_cross_entropy(
                outputs['gate'].clamp(min=1e-6, max=1 - 1e-6),
                corrupt_gate_target,
            )
            total_loss = total_loss + corruption_gate_weight * corrupt_gate_loss
            loss_terms['corrupt_gate_loss'] = corrupt_gate_loss

        return total_loss, loss_terms

    def _compute_pair_rank_loss(self, clean_outputs, corrupt_outputs):
        if (
            not isinstance(clean_outputs, dict)
            or not isinstance(corrupt_outputs, dict)
            or 'gate' not in clean_outputs
            or 'gate' not in corrupt_outputs
            or not clean_outputs.get('gate_trainable', True)
            or not corrupt_outputs.get('gate_trainable', True)
        ):
            return None
        margin = getattr(self.args, 'stic_pair_rank_margin', 0.05)
        return F.relu(margin - clean_outputs['gate'] + corrupt_outputs['gate']).mean()

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = self._slice_outputs(outputs)
                batch_y = self._slice_target(batch_y).to(self.device)
                pred = outputs['pred'] if isinstance(outputs, dict) else outputs

                pred = pred.detach()
                true = batch_y.detach()

                loss, _ = self._compute_loss(outputs, batch_y, criterion)

                total_loss.append(loss.item())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            gate_stats = self._init_gate_stats()

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                clean_batch_x = batch_x
                corrupted_batch_x, corrupted_batch, corruption_mode = self._apply_context_corruption(
                    clean_batch_x
                )
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        target_y = self._slice_target(batch_y).to(self.device)
                        if corrupted_batch and self.args.model == 'STIC':
                            clean_outputs = self.model(
                                clean_batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                            clean_outputs = self._slice_outputs(clean_outputs)
                            outputs = self.model(
                                corrupted_batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                            outputs = self._slice_outputs(outputs)
                            loss, loss_terms = self._compute_loss(
                                outputs, target_y, criterion, corrupted_batch=True
                            )
                            pair_rank_weight = getattr(self.args, 'stic_pair_rank_weight', 0.0)
                            if pair_rank_weight > 0:
                                pair_rank_loss = self._compute_pair_rank_loss(clean_outputs, outputs)
                                if pair_rank_loss is not None:
                                    loss = loss + pair_rank_weight * pair_rank_loss
                                    loss_terms['pair_rank_loss'] = pair_rank_loss
                        else:
                            clean_outputs = None
                            outputs = self.model(clean_batch_x, batch_x_mark, dec_inp, batch_y_mark)
                            outputs = self._slice_outputs(outputs)
                            loss, loss_terms = self._compute_loss(
                                outputs, target_y, criterion, corrupted_batch=False
                            )
                        train_loss.append(loss.item())
                else:
                    target_y = self._slice_target(batch_y).to(self.device)
                    if corrupted_batch and self.args.model == 'STIC':
                        clean_outputs = self.model(clean_batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        clean_outputs = self._slice_outputs(clean_outputs)
                        outputs = self.model(
                            corrupted_batch_x, batch_x_mark, dec_inp, batch_y_mark
                        )
                        outputs = self._slice_outputs(outputs)
                        loss, loss_terms = self._compute_loss(
                            outputs, target_y, criterion, corrupted_batch=True
                        )
                        pair_rank_weight = getattr(self.args, 'stic_pair_rank_weight', 0.0)
                        if pair_rank_weight > 0:
                            pair_rank_loss = self._compute_pair_rank_loss(clean_outputs, outputs)
                            if pair_rank_loss is not None:
                                loss = loss + pair_rank_weight * pair_rank_loss
                                loss_terms['pair_rank_loss'] = pair_rank_loss
                    else:
                        clean_outputs = None
                        outputs = self.model(clean_batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        outputs = self._slice_outputs(outputs)
                        loss, loss_terms = self._compute_loss(
                            outputs, target_y, criterion, corrupted_batch=False
                        )
                    train_loss.append(loss.item())

                if clean_outputs is not None:
                    self._update_gate_stats(gate_stats, clean_outputs, target_y, False, 'none')
                    self._update_gate_stats(
                        gate_stats, outputs, target_y, True, corruption_mode
                    )
                    self._update_paired_gate_stats(gate_stats, clean_outputs, outputs)
                else:
                    self._update_gate_stats(gate_stats, outputs, target_y, False, 'none')

                if (i + 1) % 100 == 0:
                    extra_terms = ', '.join(
                        f'{name}: {term.item():.7f}'
                        for name, term in loss_terms.items()
                        if name != 'pred_loss'
                    )
                    if extra_terms:
                        extra_terms = ' | ' + extra_terms
                    corruption_suffix = (
                        f' | corruption: {corruption_mode}' if corrupted_batch else ''
                    )
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | pred_loss: {3:.7f}{4}{5}".format(
                            i + 1,
                            epoch + 1,
                            loss.item(),
                            loss_terms['pred_loss'].item(),
                            extra_terms,
                            corruption_suffix,
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            gate_summary = self._format_gate_stats(gate_stats)
            if gate_summary:
                print("Gate stats | {}".format(gate_summary))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                outputs = self._slice_outputs(outputs)
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                pred = outputs['pred'] if isinstance(outputs, dict) else outputs
                pred = pred.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if pred.shape[-1] != batch_y.shape[-1]:
                        pred = np.tile(pred, [1, 1, int(batch_y.shape[-1] / pred.shape[-1])])
                    pred = test_data.inverse_transform(pred.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                f_dim = -1 if self.args.features == 'MS' else 0
                pred = pred[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
