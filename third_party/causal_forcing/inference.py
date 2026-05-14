import argparse
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from einops import rearrange
from omegaconf import OmegaConf
from project.utils.misc import save_video
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from third_party.causal_forcing.modeling.causal_inference import CausalInferencePipeline
from third_party.causal_forcing.modeling.memory import DynamicSwapInstaller, get_cuda_free_memory_gb


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None, vbench=None):
        self.vbench = vbench
        if vbench:
            with open("assets/vbench_all_dimension.txt", encoding="utf-8") as f:
                vbench_shorts = [line.strip() for line in f]

            with open(prompt_path, encoding="utf-8") as f:
                prompt_list = [line.strip() for line in f]
                self.prompt_list = []
                self.filename_list = []
            for index, (short, prompt) in enumerate(zip(vbench_shorts, prompt_list)):
                r = 25 if index < 75 else 5
                for i in range(r):
                    self.filename_list.append(f"{short}-{i}.mp4")
                    self.prompt_list.append(prompt)

            self.extended_prompt_list = None
        else:
            with open(prompt_path, encoding="utf-8") as f:
                self.prompt_list = [line.rstrip() for line in f]

            if extended_prompt_path is not None:
                with open(extended_prompt_path, encoding="utf-8") as f:
                    self.extended_prompt_list = [line.rstrip() for line in f]
                assert len(self.extended_prompt_list) == len(self.prompt_list)
            else:
                self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.vbench:
            batch["filename"] = self.filename_list[idx]
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--vbench", action="store_true", help="Use VBench prompt list")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

if "LOCAL_RANK" in os.environ:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    config.distributed = True
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    config.distributed = False
    print(f"Single GPU mode on device {device}")

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)

print(f"Free VRAM {get_cuda_free_memory_gb(device)} GB")
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True

torch.set_grad_enabled(False)

pipeline = CausalInferencePipeline(config, device=device)

if hasattr(config, "checkpoint_path") and config.checkpoint_path:
    state_dict = torch.load(config.checkpoint_path, map_location="cpu")
    key = "generator_ema" if config.use_ema else "generator"
    if isinstance(state_dict, dict) and key in state_dict:
        state_dict = state_dict[key]
    state_dict = {
        k.replace("model._fsdp_wrapped_module.", "model.", 1).replace("_fsdp_wrapped_module.", "", 1): v
        for k, v in state_dict.items()
    }
    missing, unexpected = pipeline.generator.load_state_dict(state_dict, strict=False)
    if local_rank == 0:
        if len(missing) > 0:
            print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
        if len(unexpected) > 0:
            print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")

pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

dataset = TextDataset(
    prompt_path=config.data_path,
    extended_prompt_path=config.extended_prompt_path,
    vbench=args.vbench,
)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()

    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]

    prompt = batch["prompts"][0]
    extended_prompt = batch["extended_prompts"][0] if "extended_prompts" in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    rng = torch.Generator(device=device).manual_seed(config.seed + idx)
    sampled_noise = torch.empty(
        [config.num_samples, config.num_output_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16,
    ).normal_(generator=rng)

    print("sampled_noise.device", sampled_noise.device)
    print("prompts", prompts)

    video, _ = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        return_latents=True,
        initial_latent=None,
        low_memory=low_memory,
        rng=rng,
    )
    video = rearrange(video, "b t c h w -> b c t h w").cpu()

    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    if idx < num_prompts:
        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                if args.vbench:
                    output_path = os.path.join(config.output_folder, batch["filename"][0])
                else:
                    output_folder = os.path.join(config.output_folder, f"idx_{seed_idx:04d}")
                    os.makedirs(output_folder, exist_ok=True)
                    output_path = os.path.join(output_folder, f"{idx:04d}.mp4")
            else:
                output_path = os.path.join(config.output_folder, f"rank{rank}-{prompt[:100]}-{seed_idx}.mp4")
            save_video(
                video[seed_idx:seed_idx + 1],
                output_path,
                fps=16,
                normalize=False,
                value_range=(0, 1),
            )

    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group()
