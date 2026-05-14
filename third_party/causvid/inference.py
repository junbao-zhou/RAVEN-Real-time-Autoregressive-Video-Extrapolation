from third_party.causvid.modeling.causal_inference import InferencePipeline
from diffusers.utils import export_to_video
# from causvid.data import TextDataset
from torch.utils.data import Dataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import os
import random
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from einops import rearrange
from project.utils.misc import save_video


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
                r = 25 if index < 75 else 5  # temporal_flickering repeat 25
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


def set_seed(seed: int, deterministic: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)


parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--vbench", action='store_true')
# parser.add_argument("--checkpoint_folder", type=str)
# parser.add_argument("--output_folder", type=str)
# parser.add_argument("--prompt_file_path", type=str)

args = parser.parse_args()

torch.set_grad_enabled(False)

config = OmegaConf.load(args.config_path)

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    # os.environ["NCCL_CROSS_NIC"] = "1"
    # os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    # os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    print(rank, os.environ)

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        print(f"[Rank {rank}] Initializing distributed processing...")
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed)  # Keep the same seed across all devices for inference
    config.distributed = True  # Mark as distributed for pipeline
    # if rank == 0:
    print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed
    print(f"Single GPU mode on device {device}")

pipeline = InferencePipeline(config, device=device)
pipeline.to(device=device, dtype=torch.bfloat16)

state_dict = torch.load(os.path.join(config.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']

pipeline.generator.load_state_dict(
    state_dict, strict=True
)

dataset = TextDataset(config.prompt_file_path, vbench=args.vbench)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# sampled_noise = torch.randn(
#     [1, 21, 16, 60, 104], device="cuda", dtype=torch.bfloat16
# )

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# for prompt_index in tqdm(range(len(dataset))):
#     prompts = [dataset[prompt_index]]

#     video = pipeline.inference(
#         noise=sampled_noise,
#         text_prompts=prompts
#     )[0].permute(0, 2, 3, 1).cpu().numpy()

#     export_to_video(
#         video, os.path.join(config.output_folder, f"output_{prompt_index:03d}.mp4"), fps=16)

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item()

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    all_video = []
    # num_generated_frames = 0  # Number of generated (latent) frames

    prompt = batch['prompts'][0]
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    num_samples = int(config.get("num_samples", 1))
    if extended_prompt is not None:
        prompts = [extended_prompt] * num_samples
    else:
        prompts = [prompt] * num_samples
    initial_latent = None
    rng = torch.Generator(device=device).manual_seed(config.seed + idx)

    sampled_noise = torch.empty(
        [num_samples, int(config.get("num_output_frames", 21)), 16, 60, 104], device=device, dtype=torch.bfloat16
    ).normal_(generator=rng)

    print("sampled_noise.device", sampled_noise.device)
    # print("initial_latent.device", initial_latent.device)
    print("prompts", prompts)
    # Generate 81 frames

    video = pipeline.inference(
        noise=sampled_noise,
        text_prompts=prompts,
        rng=rng
    )

    current_video = rearrange(video, 'b t c h w -> b c t h w').cpu()
    all_video.append(current_video)

    # Final output video
    video = torch.cat(all_video, dim=2)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts:
        for seed_idx in range(num_samples):
            # All processes save their videos
            if args.vbench:
                output_path = os.path.join(config.output_folder, batch['filename'][0])
            elif config.get("save_with_index", False):
                output_folder = os.path.join(config.output_folder, f"idx_{seed_idx:04d}")
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(output_folder, f"{idx:04d}.mp4")
            else:
                output_path = os.path.join(config.output_folder, f'{idx}-{seed_idx}.mp4')
            save_video(
                video[seed_idx:seed_idx + 1],
                output_path,
                fps=16,
                normalize=False,
                value_range=(0, 1),
            )

if dist.is_initialized():
    dist.destroy_process_group()
