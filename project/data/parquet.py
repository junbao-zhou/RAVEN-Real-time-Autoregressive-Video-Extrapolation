"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import itertools
import logging
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from pyarrow.fs import HadoopFileSystem, LocalFileSystem
from pyarrow.parquet import ParquetFile
from torch.utils.data import IterableDataset

from project.data import utils
from project.utils import comm, fs
from project.utils.random import combine_seed, yield_seed

logger = logging.getLogger()


def get_filesystem(path: str) -> Union[LocalFileSystem, HadoopFileSystem]:
    if path.startswith("hdfs://"):
        return HadoopFileSystem.from_uri(path)
    else:
        return LocalFileSystem()


class ParquetFileReader:
    def __init__(self, file_path, columns=None):
        self.file_path = file_path
        self.columns = columns

    def __call__(self, state):
        if isinstance(state, int):
            current_seed = state
            start_group_idx = 0
            start_row_idx = 0
        else:
            current_seed = state['seed']
            start_group_idx = state['group_idx']
            start_row_idx = state['row_idx']

        fs_instance = get_filesystem(self.file_path)
        parquet_file = ParquetFile(self.file_path, filesystem=fs_instance)

        try:
            num_row_groups = parquet_file.num_row_groups
            row_offsets = []
            total_rows = 0
            for i in range(num_row_groups):
                row_offsets.append(total_rows)
                total_rows += parquet_file.metadata.row_group(i).num_rows

            while True:
                rng_groups = random.Random(current_seed)
                group_indices = list(range(num_row_groups))
                rng_groups.shuffle(group_indices)

                for g_idx in range(start_group_idx, num_row_groups):
                    real_group = group_indices[g_idx]
                    table = parquet_file.read_row_group(real_group, columns=self.columns)
                    rows = table.to_pylist()  # batch convert, much faster than per-row slice
                    num_rows = len(rows)
                    row_offset = row_offsets[real_group]

                    rng_rows = random.Random(combine_seed(current_seed, real_group))
                    row_indices = list(range(num_rows))
                    rng_rows.shuffle(row_indices)

                    r_start = start_row_idx if g_idx == start_group_idx else 0
                    for r_idx in range(r_start, num_rows):
                        real_row = row_indices[r_idx]
                        row_data = dict(rows[real_row])
                        row_data["source_path"] = self.file_path
                        row_data["source_row_index"] = row_offset + real_row

                        # compute next state; wrap around on epoch boundary
                        if r_idx + 1 < num_rows:
                            next_state = {'seed': current_seed, 'group_idx': g_idx, 'row_idx': r_idx + 1}
                        elif g_idx + 1 < num_row_groups:
                            next_state = {'seed': current_seed, 'group_idx': g_idx + 1, 'row_idx': 0}
                        else:
                            next_state = {'seed': yield_seed(current_seed), 'group_idx': 0, 'row_idx': 0}

                        row_data["offset"] = next_state
                        yield row_data

                # next epoch
                current_seed = yield_seed(current_seed)
                start_group_idx = 0
                start_row_idx = 0

        finally:
            parquet_file.close()


class ParquetDataset(IterableDataset, ABC):
    def __init__(
        self,
        paths: List[str],
        seed: int,
        verbose: bool,
        columns: Optional[List[str]],
        total_seqlen: Optional[int],
        max_seqlen: Optional[int],
        max_seqlen_per_sample: int,
        max_retries: int,
        weights: Optional[List[float]],
        collate_fn: str,
        state_dict: Optional[dict],
    ):
        self.seed = seed
        self.verbose = verbose
        self.collate_fn = utils.get_collate_fn(collate_fn)
        self.max_seqlen = max_seqlen if total_seqlen is None else total_seqlen // comm.get_world_size()
        self.max_seqlen_per_sample = max_seqlen_per_sample
        self.max_retries = max_retries

        if weights is not None:
            assert len(paths) == len(weights), "Length of paths and weights must be the same."
            self.paths = []
            self.weights = []
            for path, weight in zip(paths, weights):
                files = fs.listdir(path, recursive=True)
                parquet_files = sorted(f for f in files if f.endswith(".parquet"))
                self.paths.extend(parquet_files)
                self.weights.extend([weight / len(parquet_files)] * len(parquet_files))
            # normalize weights
            total_w = sum(self.weights)
            self.weights = [w / total_w for w in self.weights]
        else:
            self.paths = sorted(
                f for f in itertools.chain(*(fs.listdir(p, recursive=True) for p in paths))
                if f.endswith(".parquet")
            )
            assert len(self.paths) > 0, "No parquet files found."
            self.weights = None

        self.readers = [(fp, ParquetFileReader(fp, columns=columns)) for fp in self.paths]

        self.offset = self.seed
        self.parquet_offsets = {}
        self.state_dict = state_dict if state_dict is not None else {}
        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()

    def load(self):
        real_worker_id = utils.get_worker_id()
        num_workers = utils.get_num_workers()

        last_worker_id = self.state_dict.get("last_worker_id", -1)
        worker_id = (real_worker_id + last_worker_id + 1) % num_workers

        offset = self.offset + self.rank * num_workers + worker_id
        state_dict_offsets = self.state_dict.get(str(worker_id), {})
        self.offset = state_dict_offsets.get("offset", offset)
        self.parquet_offsets = state_dict_offsets.get("parquet_offsets", {})

        logger.info(
            f"Dataset {self.__class__.__name__} Rank {self.rank} Worker {real_worker_id}"
            f" shift to resume worker {worker_id}, offset {self.offset}"
        )
        return worker_id

    @abstractmethod
    def process_data(self, data: dict, rng: random.Random) -> dict:
        raise NotImplementedError

    def _should_skip(self, data, seqlen, cur_seqlen):
        """Check if a sample should be skipped. Returns (skip: bool, reason: str)."""
        if data is None:
            return True, "process_data returned None"
        if self.max_seqlen_per_sample is not None and seqlen > self.max_seqlen_per_sample:
            return True, f"seqlen {seqlen} exceeds max_seqlen_per_sample {self.max_seqlen_per_sample}"
        if cur_seqlen + seqlen > self.max_seqlen:
            return True, f"seqlen {seqlen} would exceed max_seqlen budget"
        return False, ""

    def __iter__(self):
        num_workers = utils.get_num_workers()
        worker_id = self.load()

        if self.weights is None:
            readers = utils.get_portion_for_rank_and_worker(
                self.readers, self.rank, self.world_size, worker_id, num_workers, self.seed
            )
            weights = None
        else:
            readers = self.readers
            weights = self.weights

        base_seed = combine_seed(self.seed, self.rank, worker_id)
        iterators = [
            (fp, reader(self.parquet_offsets.get(fp, combine_seed(base_seed, i))))
            for i, (fp, reader) in enumerate(readers)
        ]

        avg_seqlen = 0.0
        cnt = 0

        while True:
            rng = random.Random(self.offset)
            ret = []
            cur_seqlen = 0
            num_retries = 0
            parquet_offsets = {}

            while len(ret) == 0 or cur_seqlen + avg_seqlen <= self.max_seqlen:
                try:
                    file_path, iterator = rng.choices(iterators, weights=weights, k=1)[0]
                    data = next(iterator)
                    parquet_offsets[file_path] = data["offset"]
                    data = self.process_data(data, rng)

                    seqlen = data["seqlens"] if data is not None else 0
                    skip, reason = self._should_skip(data, seqlen, cur_seqlen)

                    if skip:
                        # only count retries for "over budget" skips
                        if data is not None and cur_seqlen + seqlen > self.max_seqlen:
                            num_retries += 1
                            if num_retries >= self.max_retries:
                                if self.verbose:
                                    logger.warning(f"break after {num_retries} retries, avg_seqlen={avg_seqlen:.1f}")
                                break
                        if self.verbose:
                            logger.warning(f"{self.__class__.__name__}: {reason}")
                        continue

                    # accept sample
                    avg_seqlen = avg_seqlen * cnt / (cnt + 1) + seqlen / (cnt + 1)
                    cnt += 1
                    cur_seqlen += seqlen
                    num_retries = 0
                    ret.append(data)

                except Exception as ex:
                    logger.warning(f"Skipping bad sample: {ex}", exc_info=True)
                    continue

            self.offset = yield_seed(self.offset)
            result = self.collate_fn(ret)
            result["worker_id"] = worker_id
            result["offsets"] = dict(offset=self.offset, parquet_offsets=parquet_offsets)
            yield result
