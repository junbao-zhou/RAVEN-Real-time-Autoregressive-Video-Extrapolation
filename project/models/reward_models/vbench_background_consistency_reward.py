from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .base_reward_model import BaseRewardModel
from .vbench_aesthetic_reward import (
    FunctionalMultiheadAttentionForward,
    MultiheadAttentionForward,
    _custom_openai_attention_pool_forward,
    _custom_openai_residual_attention,
)


class VBenchBackgroundConsistencyRewardModel(BaseRewardModel):
    def __init__(
        self,
        clip_model_name_or_path: Optional[str] = None,
        fps: Optional[float] = None,
        num_frames: Optional[int] = None,
        frame_batch_size: int = 32,
        use_norm: bool = False,
        mean: Optional[float] = None,
        std: Optional[float] = None,
    ):
        super().__init__()
        self.validate_frame_sampling(fps, num_frames, "VBenchBackgroundConsistencyRewardModel", allow_none=True)
        if frame_batch_size <= 0:
            raise ValueError(f"frame_batch_size must be positive, got {frame_batch_size}")
        if use_norm and (std is None or std == 0):
            raise ValueError("std must be provided and non-zero when use_norm=True")
        if clip_model_name_or_path is None:
            raise ValueError("clip_model_name_or_path or pretrained_model_name_or_path must be provided for VBenchBackgroundConsistencyRewardModel")

        try:
            import clip
            from clip.model import AttentionPool2d, ResidualAttentionBlock
        except ImportError as e:
            raise ImportError("clip is required for VBenchBackgroundConsistencyRewardModel") from e

        self.clip_model_name_or_path = clip_model_name_or_path
        self.fps = fps
        self.num_frames = num_frames
        self.frame_batch_size = frame_batch_size
        self.use_norm = use_norm
        self.mean = mean
        self.std = std
        clip_model_name_or_path = os.path.expanduser(self.clip_model_name_or_path)
        self.clip_model, _ = clip.load(clip_model_name_or_path, device="cpu")
        for module in self.clip_model.modules():
            if isinstance(module, ResidualAttentionBlock):
                module.mfu_attention_forward = MultiheadAttentionForward()
                module.attention = _custom_openai_residual_attention.__get__(module, type(module))
            elif isinstance(module, AttentionPool2d):
                module.mfu_attention_forward = FunctionalMultiheadAttentionForward()
                module.forward = _custom_openai_attention_pool_forward.__get__(module, type(module))
        self.image_transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=False),
            transforms.CenterCrop(224),
            transforms.Lambda(lambda x: x.float().div(255.0)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self._align_runtime_dtypes()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(clip_model_name_or_path=pretrained_model_name_or_path, **kwargs)

    def _align_runtime_dtypes(self):
        for module in self.clip_model.modules():
            if isinstance(module, nn.LayerNorm):
                module.float()

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self._align_runtime_dtypes()
        return module

    def prepare_batch_from_frames(
        self,
        video_tensors: List[torch.Tensor],
        prompts: List[str],
        source_fps: Union[float, List[Optional[float]]],
    ):
        self.validate_video_prompt_batch(video_tensors, prompts, "VBenchBackgroundConsistencyRewardModel")
        source_fps = self.normalize_source_fps(source_fps, len(video_tensors))
        processed_videos = []
        for tensor, sample_source_fps in zip(video_tensors, source_fps):
            video = self.to_rgb_uint8_video(tensor, "VBenchBackgroundConsistencyRewardModel")
            video = self.sample_video_frames(
                video,
                source_fps=sample_source_fps,
                fps=self.fps,
                num_frames=self.num_frames,
                model_name="VBenchBackgroundConsistencyRewardModel",
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
        self._align_runtime_dtypes()
        videos = self.prepare_batch_from_frames(video_tensors, prompts, source_fps=source_fps)
        model_device = next(self.clip_model.parameters()).device
        rewards = []
        for video in videos:
            if video.shape[0] < 2:
                rewards.append(torch.zeros((), device=model_device, dtype=torch.float32))
                continue

            features = []
            for start in range(0, video.shape[0], self.frame_batch_size):
                frames = self.image_transform(video[start:start + self.frame_batch_size]).to(device=model_device)
                with torch.no_grad():
                    feat = self.clip_model.encode_image(frames).to(torch.float32)
                    features.append(F.normalize(feat, dim=-1, p=2))
            features = torch.cat(features, dim=0)
            sim_pre = F.cosine_similarity(features[:-1], features[1:], dim=-1).clamp_min(0.0)
            sim_fir = F.cosine_similarity(features[:1].expand_as(features[1:]), features[1:], dim=-1).clamp_min(0.0)
            rewards.append(((sim_pre + sim_fir) * 0.5).mean())

        reward = torch.stack(rewards).float()
        if use_norm:
            reward = (reward - float(self.mean)) / float(self.std)
        return {"BC": reward}

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
