"""
@author: Yanzuo Lu
@email:  oliveryanzuolu@gmail.com
"""
import logging
import os
import pickle
import shutil
import subprocess
import time
from typing import List

import torch.distributed as dist

from project.utils import comm

MNT_PATH_PREFIXES = os.environ.get("PROJECT_MNT_PATH_PREFIXES", "/mnt/,/nfs/").split(",")

logger = logging.getLogger()


def is_mnt_path(path: str) -> bool:
    """
    Detects whether a path is a mounted path.
    A mounted path must startswith one of the prefixes in MNT_PATH_PREFIXES.
    """
    return any(os.path.abspath(path).startswith(prefix) for prefix in MNT_PATH_PREFIXES)


def is_hdfs_path(path: str) -> bool:
    """
    Detects whether a path is an hdfs path.
    A hdfs path must startswith "hdfs://" protocol prefix.
    """
    return path.lower().startswith("hdfs://")


def mkdir(path: str, distributed: bool = False, sync: bool = True, group: dist.ProcessGroup = None):
    """
    Create directory. Support either hdfs or local path.
    Create all parent directory if not present. No-op if directory already present.
    """
    if sync:
        comm.barrier(group)

    if is_hdfs_path(path):
        if comm.get_rank() == 0 or distributed:
            subprocess.run(["hdfs", "dfs", "-mkdir", "-p", path])
    elif is_mnt_path(path):
        if comm.get_rank() == 0 or distributed:
            os.makedirs(path, exist_ok=True)
        while not exists(path):
            time.sleep(1)
    else:
        if comm.get_local_rank() == 0 or distributed:
            os.makedirs(path, exist_ok=True)

    if sync:
        comm.barrier(group)


def isdir(path: str) -> bool:
    """
    Check whether a path is a directory. Support either hdfs or local path
    Return True if the path is a directory.
    """
    if is_hdfs_path(path):
        process = subprocess.run(["hdfs", "dfs", "-test", "-d", path], capture_output=True)
        return process.returncode == 0
    return os.path.isdir(path)


def copy(src: str, tgt: str):
    """
    Copy file. Source and destination supports either hdfs or local path.
    """
    src_hdfs = is_hdfs_path(src)
    tgt_hdfs = is_hdfs_path(tgt)

    chunk_size = os.environ.get("PROJECT_HDFS_PARALLEL_CHUNK_SIZE", None)
    chunk_number = os.environ.get("PROJECT_HDFS_PARALLEL_NUM_CHUNKS", None)
    file_number = os.environ.get("PROJECT_HDFS_PARALLEL_NUM_FILES", None)

    args = []
    if chunk_size is not None:
        args.append(f"-c{chunk_size}")
    if chunk_number is not None:
        args.append(f"--ct={chunk_number}")
    if file_number is not None:
        args.append(f"-t{file_number}")

    try:
        if src_hdfs and tgt_hdfs:
            subprocess.run(["hdfs", "dfs", "-cp", "-f", src, tgt])
        elif src_hdfs and not tgt_hdfs:
            subprocess.run(["hdfs", "dfs", "-copyToLocal", "-f", *args, src, tgt])
        elif not src_hdfs and tgt_hdfs:
            subprocess.run(["hdfs", "dfs", "-copyFromLocal", "-f", *args, src, tgt])
        else:
            if os.path.isdir(src):
                shutil.copytree(src, tgt, dirs_exist_ok=True)
            else:
                shutil.copy(src, tgt)

    except Exception as e:
        logger.info("Failed to copy file from {} to {}. Remember to save it on your own!!!".format(src, tgt))
        logger.info("Error details: {}".format(e))


def exists(path: str) -> bool:
    """
    Check whether a path exists. Support either hdfs or local path
    Return True if the path exists.
    """
    if is_hdfs_path(path):
        process = subprocess.run(["hdfs", "dfs", "-test", "-e", path], capture_output=True)
        return process.returncode == 0
    return os.path.exists(path)


def listdir(path: str, recursive: bool = False, use_metafile: bool = True) -> List[str]:
    """
    List directory. Supports either hdfs or local path. Returns full path.

    Examples:
        - listdir("hdfs://dir") -> ["hdfs://dir/file1", "hdfs://dir/file2"]
        - listdir("/dir") -> ["/dir/file1", "/dir/file2"]
    """
    files = []
    dirs = []

    if is_hdfs_path(path):
        metafile = os.path.join(path, "metafile.pkl")  # A metafile contains a list of file paths
        if exists(metafile) and use_metafile:
            from project.utils.file_io import maybe_download
            with open(maybe_download(metafile, override=True), "rb") as f:
                paths = pickle.loads(f.read())
                normpath = path.rstrip('/') + '/'
                paths = [os.path.join(normpath, p.replace(normpath, "")) for p in paths]
                return paths

        pipe = subprocess.Popen(
            args=["hdfs", "dfs", "-ls", path],
            shell=False,
            stdout=subprocess.PIPE)

        for line in pipe.stdout:
            parts = line.strip().split()

            # drwxr-xr-x   - user group  4 file
            if len(parts) < 5:
                continue

            files.append(parts[-1].decode("utf8"))
            if parts[0].decode("utf8").startswith("d"):
                dirs.append(parts[-1].decode("utf8"))

        pipe.stdout.close()
        pipe.wait()

    else:
        files = [os.path.join(path, file) for file in os.listdir(path)]
        dirs = [os.path.join(path, dir) for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]

    if recursive and len(dirs) > 0:
        for path in dirs:
            files.extend(listdir(path, recursive=True))

    return files


def remove(path: str, distributed: bool = False, sync: bool = True, group: dist.ProcessGroup = None):
    """
    Remove a file or directory.
    """
    group_rank = None
    if group is not None:
        group_rank = dist.get_rank(group)
        if group_rank < 0:
            return

    if sync:
        comm.barrier(group)

    if is_hdfs_path(path):
        should_remove = distributed or (group_rank == 0 if group is not None else comm.get_rank() == 0)
        if should_remove and exists(path):
            if isdir(path):
                subprocess.run(["hdfs", "dfs", "-rm", "-r", path])
            else:
                subprocess.run(["hdfs", "dfs", "-rm", path])
    elif is_mnt_path(path):
        should_remove = distributed or (group_rank == 0 if group is not None else comm.get_rank() == 0)
        if should_remove and exists(path):
            if isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
        if not distributed and sync:
            while exists(path):
                time.sleep(1)
    else:
        should_remove = distributed or comm.get_local_rank() == 0
        if should_remove and exists(path):
            if isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

    if sync:
        comm.barrier(group)
