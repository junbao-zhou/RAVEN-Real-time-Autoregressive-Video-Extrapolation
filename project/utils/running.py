"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
from enum import Enum, auto
from typing import Optional, Union, Dict

import torch
from torch import Tensor

from project.utils import comm

"""
we may want to log running stats inside models, so we use a global variable to store them.
But note these stats are not logged to stdout and logfile, they are only logged to the experiment tracker.
"""
RUNNING_AVERAGE_METER = None
RUNNING_ACCUMULATOR = None

"""
MoE related modules behave differently in different training phases.
We use a global variable to store the current training phase.
"""
TRAINING_PHASE = None


class TrainingPhase(Enum):
    IN_FORWARD = auto()
    IN_BACKWARD = auto()
    IN_OPTIMIZATION = auto()
    IN_EVAL = auto()


def get_training_phase():
    global TRAINING_PHASE
    return TRAINING_PHASE


def set_training_phase(phase):
    global TRAINING_PHASE
    TRAINING_PHASE = phase


class AverageMeter:
    def __init__(self):
        self.reset()

    def update(
        self,
        log_dict: Dict[str, Union[float, Tensor]],
        cnt: Optional[Union[Dict[str, Union[int, Tensor]], Tensor]] = None
    ):
        if cnt is None:
            cnt_dict = dict()
        elif torch.is_tensor(cnt):  # could be seqlens or just a number, anyway sum it up
            cnt_dict = {k: torch.sum(cnt).item() for k in log_dict.keys()}
        else:
            assert isinstance(cnt, dict), f"cnt should be None, tensor or dict, but got {type(cnt)}"
            cnt_dict = cnt

        for k, v in log_dict.items():
            self.put_scalar(k, v, cnt_dict.get(k, 1))

    def put_scalar(self, key, value, cnt=1):
        if value is None: return
        if torch.is_tensor(value):
            value = value.item()
        if torch.is_tensor(cnt):
            cnt = torch.sum(cnt).item()
        if key not in self._keys:
            self._keys.append(key)
            self._avg[key] = 0.
            self._cnt[key] = 0
        self._avg[key] = (self._avg[key] * self._cnt[key] + value * cnt) / (self._cnt[key] + cnt)
        self._cnt[key] += cnt

    def sync(self):
        gather_avgs = comm.all_gather_object(self._avg)
        gather_cnts = comm.all_gather_object(self._cnt)
        keys = set()
        for avg in gather_avgs:
            keys.update(avg.keys())
        self._keys = list(keys)
        for k in keys:
            avgs = [v[k] for v in gather_avgs if k in v]
            cnts = [v[k] for v in gather_cnts if k in v]
            self._cnt[k] = sum(cnts)
            self._avg[k] = sum([a * c / self._cnt[k] for a, c in zip(avgs, cnts)])

    def reset(self):
        self._keys = []
        self._avg = {}
        self._cnt = {}

    @property
    def keys(self):
        return sorted(self._keys)

    @property
    def avg(self):
        return {k: self._avg[k] for k in self.keys}

    @property
    def cnt(self):
        return {k: self._cnt[k] for k in self.keys}

    def items(self):
        for k in self.keys:
            yield k, self._avg[k]


class Accumulator:
    def __init__(self):
        self.reset()
        self.sum = {}

    def state_dict(self) -> dict:
        return self.sum

    def load_state_dict(self, state_dict, **kwargs):
        self.sum = state_dict

    def update(self, log_dict):
        for k, v in log_dict.items():
            if k not in self.keys:
                self.keys.append(k)
                self.keys = sorted(self.keys)
                self._sum[k] = torch.tensor(0, dtype=torch.long)
            if v is None: continue
            if torch.is_tensor(v): v = v.item()
            self._sum[k] += int(v)

    def put_scalar(self, key, value):
        if key not in self.keys:
            self.keys.append(key)
            self.keys = sorted(self.keys)
            self._sum[key] = torch.tensor(0, dtype=torch.long)
        if value is None: return
        if torch.is_tensor(value): value = value.item()
        self._sum[key] += int(value)

    def put_scalars(self, prefix="", **kwargs):
        for k, v in kwargs.items():
            self.put_scalar(f"{prefix}{k}", v)

    def sync(self):
        gather_meters = comm.all_gather_object(self._sum)
        keys = set()
        for meter in gather_meters:
            keys.update(meter.keys())
        for k in keys:
            meters = [v[k] for v in gather_meters if k in v]
            if k not in self.sum:
                self.sum[k] = torch.tensor(0, dtype=torch.long)
            self.sum[k] += sum(meters)

    def reset(self):
        self.keys = []
        self._sum = {}

    def items(self):
        for k in self.keys:
            yield k, self.sum[k]


def get_running_average_meter():
    global RUNNING_AVERAGE_METER
    if RUNNING_AVERAGE_METER is None:
        RUNNING_AVERAGE_METER = AverageMeter()
    return RUNNING_AVERAGE_METER


def get_running_accumulator():
    global RUNNING_ACCUMULATOR
    if RUNNING_ACCUMULATOR is None:
        RUNNING_ACCUMULATOR = Accumulator()
    return RUNNING_ACCUMULATOR
