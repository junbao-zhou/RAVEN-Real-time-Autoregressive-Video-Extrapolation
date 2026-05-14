"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import json
import logging
import os
from abc import ABC
from dataclasses import dataclass, field
from typing import Optional

from torch.utils.data import DataLoader

from project.data import DATASET_REGISTRY
from project.data.utils import merge_dicts
from project.utils import comm
from project.utils.config import CfgNode
from project.utils.dataclass import Dataclass
from project.utils.file_io import maybe_download, maybe_upload

logger = logging.getLogger()


@dataclass
class BaseDataloaderConfig(Dataclass):
    batch_size: Optional[int] = field(default=None)  # only for validation
    num_workers: int = field(default=4)
    prefetch_factor: int = field(default=4)
    pin_memory: bool = field(default=True)
    collate_fn: str = field(default="default")


class BaseDataloader(ABC):
    """
    Base class for dataloader in support of resume and load checkpoint.
    """
    def __init__(
        self,
        meta_cfg: CfgNode,
        dataset_cfg: CfgNode,
        ckpt_path: str = None,
        **kwargs
    ):
        self.config = BaseDataloaderConfig(**kwargs)
        self.state_dict, self.num_workers = self.load(ckpt_path)

        dataset_cls = DATASET_REGISTRY.get(dataset_cfg["_class_name"])
        self.dataset = dataset_cls(
            collate_fn=self.config.collate_fn,  # collate within dataset during training
            state_dict=self.state_dict,
            **meta_cfg,
            **dataset_cfg["_config"]
        )

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=None,
            num_workers=self.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory
        )
        self.loader_iter = iter(self.dataloader)

    def load(self, ckpt_path: str):
        rank = comm.get_rank()
        world_size = comm.get_world_size()
        num_workers = self.config.num_workers

        if ckpt_path is None:
            return dict(), num_workers

        # download entire ckpt dir (no distributed=True to avoid race)
        local_ckpt_path = maybe_download(ckpt_path)

        # discover all saved rank jsons
        saved_files = sorted(
            f for f in os.listdir(local_ckpt_path)
            if f.startswith("rank_") and f.endswith(".json")
        )
        saved_world_size = len(saved_files)
        if saved_world_size == 0:
            logger.warning(f"No rank json found in {ckpt_path}, starting fresh.")
            return dict(), num_workers

        # read one file to figure out saved num_workers
        with open(os.path.join(local_ckpt_path, saved_files[0]), "r", encoding="utf8") as f:
            sample = json.load(f)
        saved_num_workers = sample.get("num_workers", num_workers)

        saved_total = saved_world_size * saved_num_workers
        cur_total = world_size * num_workers

        # fast path: topology unchanged
        if saved_world_size == world_size and saved_num_workers == num_workers:
            path = os.path.join(local_ckpt_path, f"rank_{rank}.json")
            with open(path, "r", encoding="utf8") as f:
                state_dict = json.load(f)
            logger.info(f"Resume rank {rank} from {path}")
            return state_dict, num_workers

        # redistribution path: saved_total must be divisible by world_size
        if saved_total % world_size != 0:
            raise RuntimeError(
                f"Cannot redistribute: saved_world_size({saved_world_size}) x "
                f"saved_num_workers({saved_num_workers}) = {saved_total} "
                f"is not divisible by current world_size({world_size})."
            )

        new_num_workers = saved_total // world_size
        logger.info(
            f"Redistributing: {saved_world_size}x{saved_num_workers} -> "
            f"{world_size}x{new_num_workers}, rank {rank}"
        )

        # figure out which (old_rank, old_worker) slots this rank needs
        # global slot ordering: rank0_worker0, rank0_worker1, ..., rank1_worker0, ...
        global_start = rank * new_num_workers
        global_end = global_start + new_num_workers

        state_dict = {}
        needed_ranks = set()
        for g in range(global_start, global_end):
            needed_ranks.add(g // saved_num_workers)

        # load needed rank jsons
        rank_data = {}
        for old_rank in needed_ranks:
            path = os.path.join(local_ckpt_path, f"rank_{old_rank}.json")
            with open(path, "r", encoding="utf8") as f:
                rank_data[old_rank] = json.load(f)

        # remap slots to new worker ids
        for new_worker in range(new_num_workers):
            global_slot = global_start + new_worker
            old_rank = global_slot // saved_num_workers
            old_worker = global_slot % saved_num_workers
            old_state = rank_data[old_rank]
            worker_state = old_state.get(str(old_worker), {})
            state_dict[str(new_worker)] = worker_state

        # clear last_worker_id since topology changed, let dataset start from worker 0
        state_dict["last_worker_id"] = -1
        state_dict["num_workers"] = new_num_workers

        logger.info(f"Rank {rank} redistributed state: workers={new_num_workers}")
        return state_dict, new_num_workers

    def save(self, save_dir: str):
        """
        Save the dataloader to the checkpoint in the save_dir.

        Args:
            save_dir (str): directory to save the dataloader.
        """
        self.state_dict["num_workers"] = self.num_workers
        maybe_upload(self.state_dict, f"rank_{comm.get_rank()}.json", save_dir)
        comm.barrier()

    def __iter__(self):
        """
        Initialize the dataloader iterator.
        """
        return self

    def __next__(self):
        """
        Get the next batch of data.
        """
        try:
            batch = next(self.loader_iter)
            self._update_state(batch)
            return batch
        except StopIteration:
            logger.info("DataLoader reached the end of the dataset.")
            raise StopIteration

    def _update_state(self, batch: dict):
        """
        Update the state dict of the dataloader.

        Args:
            batch (dict): current batch of data.
        """
        if batch.get("worker_id") is None or batch.get("offsets") is None:  # not in training
            return
        worker_id = str(batch["worker_id"])
        if worker_id not in self.state_dict:
            self.state_dict[worker_id] = dict()
        self.state_dict[worker_id] = merge_dicts(self.state_dict[worker_id], batch["offsets"])
        self.state_dict["last_worker_id"] = batch["worker_id"]
        self.state_dict = dict(sorted(self.state_dict.items(), key=lambda x: str(x[0])))
