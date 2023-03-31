# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import re
import torch
from . import training_stats

#----------------------------------------------------------------------------

def init():
    if 'MASTER_ADDR' not in os.environ:
        slurm_nodelist = os.environ.get("SLURM_NODELIST")
        if slurm_nodelist:
            root_node = slurm_nodelist.split(" ")[0].split(",")[0]
        else:
            root_node = "127.0.0.1"
        root_node = resolve_root_node_address(root_node)
        os.environ["MASTER_ADDR"] = root_node
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '29500'
    if 'RANK' not in os.environ:
        os.environ['RANK'] = os.environ.get('SLURM_PROCID','0')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = os.environ.get('SLURM_LOCALID', '0')
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = os.environ.get('SLURM_NTASKS', '1')
j
    backend = 'gloo' if os.name == 'nt' else 'nccl'
    torch.distributed.init_process_group(backend=backend, init_method='env://')
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    sync_device = torch.device('cuda') if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)

def resolve_root_node_address(root_node: str) -> str:
    if "[" in root_node:
        name, numbers = root_node.split("[", maxsplit=1)
        number = numbers.split(",", maxsplit=1)[0]
        if "-" in number:
            number = number.split("-")[0]

        number = re.sub("[^0-9]", "", number)
        root_node = name + number

    return root_node

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

#----------------------------------------------------------------------------

def should_stop():
    return False

#----------------------------------------------------------------------------

def update_progress(cur, total):
    _ = cur, total

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------
