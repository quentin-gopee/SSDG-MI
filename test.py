import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# new imports
from yacs.config import CfgNode as CN
import copy

# datasets
print('creating ssdg_pacs dataset...')
import datasets.ssdg_pacs

# trainers
print('creating FBCSA trainer...')
import trainers.FBCSA