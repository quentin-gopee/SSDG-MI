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

    def update(self):
        pseudo_counter = Counter(self.selected_label.tolist())
        if max(pseudo_counter.values()) < self.len_unlabeled:  # not all(5w) -1
            for i in range(self.num_classes):
                self.classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())

    def compute_mask(self, max_probs, pseudo_labels, idx_ulb):
        if not self.selected_label.is_cuda:
            self.selected_label = self.selected_label.to(max_probs.device)
        if not self.classwise_acc.is_cuda:
            self.classwise_acc = self.classwise_acc.to(max_probs.device)
            
        classwise_treshold = self.conf_thre * (self.classwise_acc[pseudo_labels] / (2. - self.classwise_acc[pseudo_labels])) # convex
        mask = max_probs.ge(classwise_treshold)
        select = max_probs.ge(self.conf_thre)
        mask = mask.to(max_probs.dtype)

        # update
        if idx_ulb[select == 1].nelement() != 0:
            self.selected_label[idx_ulb[select == 1]] = pseudo_labels[select == 1]
        self.update()

        output = {
            "mask": mask,
            "selected_label": self.selected_label,
            "classwise_threshold": classwise_treshold
        }

        return output