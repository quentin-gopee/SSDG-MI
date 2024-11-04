import torch
from torch.nn import functional as F
from collections import Counter


class PLMask():
    """
    Pseudo-Label Masking
    """
    def __init__(self):
        pass

    def compute_mask(self, max_probs=None, pseudo_labels=None, idx_ulb=None):
        pass


class FixMatchMask(PLMask):
    """
    FixMatch Masking
    """
    def __init__(self, conf_thre=0.95):
        super().__init__()
        self.conf_thre = conf_thre
    
    @torch.no_grad()
    def compute_mask(self, max_probs, pseudo_labels=None, idx_ulb=None):
        mask = (max_probs >= self.conf_thre).float()
        return mask
    

class FlexMatchMask(PLMask):
    """
    FlexMatch Masking (adapted from TorchSSL)
    """
    def __init__(self, num_classes, len_unlabeled, conf_thre=0.95):
        super().__init__()
        self.num_classes = num_classes
        self.len_unlabeled = len_unlabeled
        self.conf_thre = conf_thre
        self.selected_label = torch.ones((self.len_unlabeled,), dtype=torch.long, ) * -1
        self.classwise_acc = torch.zeros((self.num_classes,))
        self.classwise_threshold = torch.zeros((self.num_classes,))

    @torch.no_grad()
    def update(self):
        pseudo_counter = Counter(self.selected_label.tolist())
        if max(pseudo_counter.values()) < self.len_unlabeled:  # not all(5w) -1
            for i in range(self.num_classes):
                self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

    @torch.no_grad()
    def compute_mask(self, max_probs, pseudo_labels, idx_ulb):
        if not self.selected_label.is_cuda:
            self.selected_label = self.selected_label.to(max_probs.device)
        if not self.classwise_acc.is_cuda:
            self.classwise_acc = self.classwise_acc.to(max_probs.device)

        self.classwise_threshold = self.conf_thre * (self.classwise_acc[pseudo_labels] / (2. - self.classwise_acc[pseudo_labels])) # convex
        mask = max_probs.ge(self.classwise_threshold)
        select = max_probs.ge(self.conf_thre)
        mask = mask.to(max_probs.dtype)

        # update
        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1]] = pseudo_labels[select == 1]
        self.update()

        return mask


class FreeMatchMask(PLMask):
    """
    FreeMatch Masking (adapted from TorchSSL)
    """
    def __init__(self, num_classes, momentum=0.999):
        super().__init__()
        self.num_classes = num_classes
        self.m = momentum
        
        self.p_model = torch.ones((self.num_classes)) / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) / self.num_classes
        self.time_p = self.p_model.mean()
        
        self.classwise_threshold = torch.zeros((self.num_classes,))
    
    @torch.no_grad()
    def update(self, max_probs, pseudo_labels):
        self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        self.p_model = self.p_model * self.m + (1 - self.m) * max_probs.mean(dim=0)
        hist = torch.bincount(pseudo_labels.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())

    @torch.no_grad()
    def compute_mask(self, max_probs, pseudo_labels, idx_ulb=None):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(max_probs.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(max_probs.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(max_probs.device)

        self.update(max_probs, pseudo_labels)

        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        self.classwise_threshold = self.time_p * mod[pseudo_labels]
        mask = max_probs.ge(self.classwise_threshold).to(max_probs.dtype)

        return mask