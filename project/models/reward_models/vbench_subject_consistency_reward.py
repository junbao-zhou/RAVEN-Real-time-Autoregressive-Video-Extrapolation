from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torchvision import transforms

from .base_reward_model import BaseRewardModel
from .vbench_dino.vision_transformer import vit_base


class VBenchSubjectConsistencyRewardModel(BaseRewardModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        frame_batch_size: int = 32,
        use_norm: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__()
        self.validate_frame_sampling(fps, num_frames, "VBenchSubjectConsistencyRewardModel", allow_none=True)
        if frame_batch_size <= 0:
            raise ValueError(f"frame_batch_size must be positive, got {frame_batch_size}")
        if use_norm and (std is None or std == 0):
            raise ValueError("std must be provided and non-zero when use_norm=True")

        self.model = vit_base(patch_size=16, num_classes=0)
        if model_path is not None:
            ckpt = torch.load(model_path, map_location="cpu")
            self.load_state_dict(ckpt)
        self.model.eval()

        self.fps = fps
        self.num_frames = num_frames
        self.frame_batch_size = frame_batch_size
        self.use_norm = use_norm
        self.mean = mean
        self.std = std
        self.image_transform = transforms.Compose([
            transforms.Resize(224, antialias=False),
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(model_path=pretrained_model_name_or_path, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        remapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            if not new_key.startswith("model."):
                new_key = f"model.{new_key}"
            remapped_state_dict[new_key] = value
        return super().load_state_dict(remapped_state_dict, strict=strict, assign=assign)

    def prepare_batch_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
    ):
        self.validate_video_prompt_batch(video_tensors, prompts, "VBenchSubjectConsistencyRewardModel")
        source_fps = self.normalize_source_fps(source_fps, len(video_tensors))
        processed_videos = []
        for tensor, sample_source_fps in zip(video_tensors, source_fps):
            video = self.to_rgb_uint8_video(tensor, "VBenchSubjectConsistencyRewardModel")
            video = self.sample_video_frames(
                video,
                source_fps=sample_source_fps,
                fps=self.fps,
                num_frames=self.num_frames,
                model_name="VBenchSubjectConsistencyRewardModel",
            )
            processed_videos.append(video)
        return processed_videos

    def reward_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
        use_norm: Optional[bool] = None,
    ):
        if use_norm is None:
            use_norm = self.use_norm
        videos = self.prepare_batch_from_frames(video_tensors, prompts, source_fps=source_fps)
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        rewards = []
        for video in videos:
            if video.shape[0] < 2:
                rewards.append(torch.zeros((), device=model_device, dtype=torch.float32))
                continue

            features = []
            for start in range(0, video.shape[0], self.frame_batch_size):
                frames = self.image_transform(video[start:start + self.frame_batch_size]).to(device=model_device, dtype=model_dtype)
                with torch.no_grad():
                    feat = self.model(frames).to(torch.float32)
                    features.append(F.normalize(feat, dim=-1, p=2))
            features = torch.cat(features, dim=0)
            sim_pre = F.cosine_similarity(features[:-1], features[1:], dim=-1).clamp_min(0.0)
            sim_fir = F.cosine_similarity(features[:1].expand_as(features[1:]), features[1:], dim=-1).clamp_min(0.0)
            rewards.append(((sim_pre + sim_fir) * 0.5).mean())

        reward = torch.stack(rewards).float()
        if use_norm:
            reward = (reward - float(self.mean)) / float(self.std)
        return {"SC": reward}

    def forward(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
        use_norm: Optional[bool] = None,
        **kwargs,
    ):
        del kwargs
        return self.reward_from_frames(
            video_tensors=video_tensors,
            prompts=prompts,
            source_fps=source_fps,
            use_norm=use_norm,
        )
