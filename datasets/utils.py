from collections import defaultdict
import numpy as np
import random
from dassl.utils import read_json, write_json
import os.path as osp
from dassl.data.datasets import Datum


def random_numbers(n_sum, n_numbers):
    '''
    Generates a list of n_numbers random numbers between 1 and num_sum-n_numbers that sum to n_num
    '''
    rand_num = np.sort(random.sample(range(1, n_sum), n_numbers-1))
    num_labeled_per_class = [rand_num[0]] + [rand_num[i] - rand_num[i-1] for i in range(1, len(rand_num))] + [n_sum-rand_num[-1]]
    return num_labeled_per_class

def exp_imbalance_l(N, C, gamma):
    '''
    N: total number of samples
    C: number of classes
    gamma: imbalance ratio

    return: list of number of samples per class with exponential imbalance
    '''
    n1 = N*(1-gamma**(-1/(C-1)))/(1-gamma**(-C/(C-1)))
    n_samples = []

    for i in range(C):
        n_samples.append(int(n1*gamma**(-i/(C-1))))

    # add remaining samples to the first classes
    for i in range(N-sum(n_samples)):
        n_samples[i] += 1

    return n_samples

def exp_imbalance_u(m1, C, gamma):
    '''
    m1: number of samples in the majority class
    C: number of classes
    gamma: imbalance ratio

    return: list of number of samples per class with exponential imbalance
    '''
    n_samples = []

    for i in range(C):
        n_samples.append(int(m1*gamma**(-i/(C-1))))

    return n_samples

def count_classes(items):
    class_count = defaultdict(int)
    for item in items:
        class_count[item.label] += 1
    return class_count

def write_json_train(filepath, src_domains, image_dir, train_x, train_u):
        def _convert_to_list(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                domain = item.domain
                dname = src_domains[domain]
                impath = impath.replace(image_dir, "")
                if impath.startswith("/") or impath.startswith("\\"):
                    impath = impath[1:]
                out.append((impath, label, dname))
            return out

        train_x = _convert_to_list(train_x)
        train_u = _convert_to_list(train_u)
        output = {"train_x": train_x, "train_u": train_u}

        write_json(output, filepath)
        print(f'Saved the split to "{filepath}"')

def read_json_train(filepath, src_domains, image_dir):
        """
        The latest office_home_dg dataset's class folders have
        been changed to only contain the class names, e.g.,
        000_Alarm_Clock/ is changed to Alarm_Clock/.
        """

        def _convert_to_datums(items):
            out = []
            for impath, label, dname in items:
                if dname not in src_domains:
                    continue
                domain = src_domains.index(dname)
                impath2 = osp.join(image_dir, impath)
                if not osp.exists(impath2):
                    impath = impath.split("/")
                    if impath[-2].startswith("0"):
                        impath[-2] = impath[-2][4:]
                    impath = "/".join(impath)
                    impath2 = osp.join(image_dir, impath)
                item = Datum(impath=impath2, label=int(label), domain=domain)
                out.append(item)
            return out

        print(f'Reading split from "{filepath}"')
        split = read_json(filepath)
        train_x = _convert_to_datums(split["train_x"])
        train_u = _convert_to_datums(split["train_u"])

        return train_x, train_u