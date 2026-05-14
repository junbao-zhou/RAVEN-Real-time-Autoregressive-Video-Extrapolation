"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import argparse
import json
import os
import subprocess

import torch.distributed as dist

from project.engines import ENGINE_REGISTRY
from project.utils.config import CfgNode
from project.utils.config import gcfg


def get_git_info():
    """Get git commit hash; clean means no staged, unstaged, or untracked changes."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
        # porcelain output is empty iff working tree is fully clean
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        is_clean = len(status) == 0
        return commit, is_clean
    except Exception:
        return None, None


def remap_config_rootdirs(cfg, rootdir_mapping, current_cluster, key_path=()):
    current_rootdir = os.path.normpath(rootdir_mapping[current_cluster])
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    source_rootdirs = {
        cluster_name: os.path.normpath(rootdir)
        for cluster_name, rootdir in rootdir_mapping.items()
        if cluster_name != current_cluster
    }

    def replace_path(value, path):
        normalized_value = os.path.normpath(value)
        for source_rootdir in source_rootdirs.values():
            if normalized_value == source_rootdir or normalized_value.startswith(source_rootdir + os.sep):
                suffix = normalized_value[len(source_rootdir):].lstrip(os.sep)
                remapped_value = os.path.join(current_rootdir, suffix) if suffix else current_rootdir
                if local_rank == 0:
                    print(
                        f"Remapped config path at {'.'.join(path) if path else '<root>'}: "
                        f"{value} -> {remapped_value}"
                    )
                return remapped_value
        return value

    if isinstance(cfg, CfgNode):
        for key, value in cfg.items():
            if key_path == ("persistence",) and key == "rootdir_mapping":
                continue
            cfg[key] = remap_config_rootdirs(value, rootdir_mapping, current_cluster, key_path + (key,))
        return cfg

    if isinstance(cfg, dict):
        remapped_items = []
        for key, value in cfg.items():
            remapped_key = remap_config_rootdirs(key, rootdir_mapping, current_cluster, key_path + ("<key>",))
            remapped_value = remap_config_rootdirs(value, rootdir_mapping, current_cluster, key_path + (str(remapped_key),))
            remapped_items.append((remapped_key, remapped_value))
        cfg.clear()
        cfg.update(remapped_items)
        return cfg

    if isinstance(cfg, list):
        for index, value in enumerate(cfg):
            cfg[index] = remap_config_rootdirs(value, rootdir_mapping, current_cluster, key_path + (str(index),))
        return cfg

    if isinstance(cfg, tuple):
        return tuple(
            remap_config_rootdirs(value, rootdir_mapping, current_cluster, key_path + (str(index),))
            for index, value in enumerate(cfg)
        )

    if isinstance(cfg, str):
        return replace_path(cfg, key_path)

    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("opts", help="Modify config options using the command-line 'KEY VALUE' pairs",
                        default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    gcfg.init()
    gcfg.merge_from_file(args.config_file)
    gcfg.merge_from_list(args.opts)
    if gcfg.persistence.get("exp_name", None) is None:
        gcfg.persistence.exp_name = os.path.splitext(os.path.basename(args.config_file))[0][:128]
    rootdir_mapping = os.getenv("PROJECT_ROOTDIR_MAPPING", None)
    if rootdir_mapping is not None:
        rootdir_mapping = json.loads(rootdir_mapping)
        cluster = os.getenv("PROJECT_CLUSTER", None)
        if cluster is not None and cluster in rootdir_mapping:
            remap_config_rootdirs(gcfg, rootdir_mapping, cluster)

    # Only check git info on rank 0 to avoid redundant calls in distributed env
    git_commit, git_clean = get_git_info()
    if git_commit:
        gcfg.persistence.git_commit = git_commit
        gcfg.persistence.git_clean = git_clean

    try:
        engine_cls = ENGINE_REGISTRY.get(gcfg._class_name)
        engine = engine_cls(gcfg)
        engine.run()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
