import contextlib
import random
import os
import time
import datetime
import numpy as np
from math import ceil
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.data.data_manager import DataManager, build_data_loader
from dassl.engine import TRAINER_REGISTRY, TrainerXU, SimpleNet
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform
from dassl.utils import count_num_param

from .utils.pl_mask import FixMatchMask, FlexMatchMask, FreeMatchMask
from .utils.freematch import self_adaptative_fairness


class NormalClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x, stochastic=True):
        return self.linear(x)


@TRAINER_REGISTRY.register()
class ME(TrainerXU):
    """
    Marginal Entropy Trainer
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.conf_thre = cfg.TRAINER.ME.CONF_THRE
        self.weight_h = cfg.TRAINER.ME.WEIGHT_H
        self.me = cfg.TRAINER.ME.ME
        self.baseline = cfg.TRAINER.ME.BASELINE
        self.len_x = len(self.dm.dataset.train_x)
        self.len_u = len(self.dm.dataset.train_u)
        self.len_tot = self.len_x + self.len_u

        if self.baseline == 'fixmatch':
            self.pl_mask = FixMatchMask(self.conf_thre)
        elif self.baseline == 'flexmatch':
            self.pl_mask = FlexMatchMask(self.num_classes, self.len_tot, self.conf_thre)
        elif self.baseline == 'freematch':
            self.pl_mask = FreeMatchMask(self.num_classes)
            self.weight_f = 0.001


    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.ME.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X
        assert cfg.TRAINER.ME.BASELINE in ['fixmatch', 'freematch', 'flexmatch']

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.ME.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        self.dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = self.dm.train_loader_x
        self.train_loader_u = self.dm.train_loader_u
        self.val_loader = self.dm.val_loader
        self.test_loader = self.dm.test_loader
        self.num_classes = self.dm.num_classes
        self.lab2cname = self.dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = NormalClassifier(self.G.fdim, self.num_classes)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.ME.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.ME.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

    def assess_y_pred_quality(self, y_pred, y_true, mask):
        n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
        acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
        acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
        keep_rate = mask.sum() / mask.numel()
        output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
        return output
    
    def forward_backward(self, batch_x, batch_u):
        parsed_data = self.parse_batch_train(batch_x, batch_u)
        input_x, input_x2, label_x, input_u, input_u2, label_u, index_x, index_u = parsed_data
        input_u = torch.cat([input_x, input_u], 0)
        input_u2 = torch.cat([input_x2, input_u2], 0)
        index_u = torch.cat([index_x, index_u])
        n_x = input_x.size(0)

        ####################
        # Generate pseudo labels
        ####################
        with torch.no_grad():
            prob_u = F.softmax(self.C(self.G(input_u), stochastic=False), 1)
            max_probs, pseudo_labels = prob_u.max(1)
            mask_u = self.pl_mask.compute_mask(max_probs, pseudo_labels, index_u)

            # Evaluate pseudo labels' accuracy
            y_u_pred_stats = self.assess_y_pred_quality(
                pseudo_labels[n_x:], label_u, mask_u[n_x:]
            )

        ####################
        # Supervised loss
        ####################
        output_x = self.C(self.G(input_x), stochastic=False)
        loss_x = F.cross_entropy(output_x, label_x)

        ####################
        # Unsupervised loss
        ####################
        output_u = self.C(self.G(input_u2), stochastic=False)
        loss_u = F.cross_entropy(output_u, pseudo_labels, reduction="none")
        loss_u = (loss_u * mask_u).mean()

        if self.baseline == 'freematch':
            loss_saf, _ = self_adaptative_fairness(mask_u, output_u, self.pl_mask.p_model, self.pl_mask.label_hist)

        ####################
        # Marginal Entropy loss
        ####################
        if self.me:
            prob_u_marginal = F.softmax(output_u, 1).mean(0)

            if self.me == 'shannon':
                loss_marginal_entropy = self.weight_h * (prob_u_marginal * torch.log(prob_u_marginal + 1e-12)).sum()
            elif self.me == 'alpha':
                if self.weight_h == 1:
                    loss_marginal_entropy = (prob_u_marginal * torch.log(prob_u_marginal + 1e-12)).sum()
                else:
                    loss_marginal_entropy = -(1/(self.weight_h-1)) * (1 - (prob_u_marginal ** self.weight_h).sum())
            else:
                raise ValueError(f"Unknown marginal entropy type: {self.me}")
            
        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()

        loss_all += loss_u
        loss_summary["loss_u_aug"] = loss_u.item()

        if self.baseline == 'freematch':
            loss_all += self.weight_f * loss_saf
            loss_summary["loss_sat"] = self.weight_f * loss_saf.item()

        if self.me:
            loss_all += loss_marginal_entropy
            loss_summary["loss_marginal_entropy"] = loss_marginal_entropy.item()

        # if loss_all contains NaN
        if (loss_all != loss_all).data.any():
            print("NaN detected in loss.")
        
        self.model_backward_and_update(loss_all)

        loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
        loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
        loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]

        if self.baseline in ['flexmatch', 'freematch']:
            loss_summary["mean_threshold"] = self.pl_mask.classwise_threshold.mean()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    
    def parse_batch_train(self, batch_x, batch_u):
        input_x = batch_x["img"]
        input_x2 = batch_x["img2"]
        label_x = batch_x["label"]
        input_u = batch_u["img"]
        input_u2 = batch_u["img2"]
        # label_u is used only for evaluating pseudo labels' accuracy
        label_u = batch_u["label"]
        index_x = batch_x["index"]
        index_u = batch_u["index"] + self.len_x

        input_x = input_x.to(self.device)
        input_x2 = input_x2.to(self.device)
        label_x = label_x.to(self.device)
        input_u = input_u.to(self.device)
        input_u2 = input_u2.to(self.device)
        label_u = label_u.to(self.device)
        index_x = index_x.to(self.device)
        index_u = index_u.to(self.device)

        return input_x, input_x2, label_x, input_u, input_u2, label_u, index_x, index_u

    def model_inference(self, input):
        
        features = self.G(input)
        prediction = self.C(features, stochastic=False)

        return prediction

    def after_train(self):
        print("Finish training")

        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()

        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()

    @torch.no_grad()
    def test_train(self):
        """A generic testing pipeline."""
        y_pred = []

        self.set_model_mode("eval")
        self.evaluator.reset()

        print(f"Evaluate on the *train_x* set")
        iter_train_x = iter(self.train_loader_x)
        n_batches_x = len(self.train_loader_x)
        print(f"n_batches_x: {n_batches_x}")
        for i in tqdm(range(n_batches_x)):
            batch = next(iter_train_x)
            input, label = self.parse_batch_test(batch)
            output = self.model_inference(input)
            pred = output.max(1)[1]
            y_pred.extend(pred.cpu().numpy().tolist())
            self.evaluator.process(output, label)
        
        self.evaluator.evaluate()

        # print(f"Evaluate on the *train_u* set")
        # iter_train_u = iter(self.train_loader_u)
        # n_batches_u = len(self.train_loader_u)
        # for i in tqdm(range(n_batches_u)):
        #     batch = next(iter_train_u)
        #     input, label = self.parse_batch_test(batch)
        #     output = self.model_inference(input)
        #     pred = output.max(1)[1]
        #     y_pred.extend(pred.cpu().numpy().tolist())

        # # Predictions on train_x and train_u
        # # print(y_pred)

        # # histogram of y_pred
        # hist, _ = np.histogram(y_pred, bins=self.num_classes)
        # print("Histogram of y_pred:")
        # print(hist)