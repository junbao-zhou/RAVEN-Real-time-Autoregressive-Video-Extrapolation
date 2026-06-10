# RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO

[Yanzuo Lu](https://yanzuo.lu/) · [Ronglai Zuo](https://2000zrl.github.io/) · [Jiankang Deng](https://jiankangdeng.github.io/) — Imperial College London

Project page: <https://yanzuo.lu/raven>

## TL;DR

https://github.com/user-attachments/assets/c1aa3b08-4a6e-431f-8b63-d7266774de3b

Causal autoregressive video diffusion models support real-time streaming generation by extrapolating future chunks from previously generated content. Distilling such generators from high-fidelity bidirectional teachers yields competitive few-step models, yet a persistent gap between the history distributions encountered during training and those arising at inference constrains generation quality over long horizons. We introduce the **Real-time Autoregressive Video Extrapolation Network (RAVEN)**, a training-time test framework that repacks each self rollout into an interleaved sequence of clean historical endpoints and noisy denoising states. This formulation aligns training attention with inference-time extrapolation and allows downstream chunk losses to supervise the history representations on which future predictions depend. We further propose **Consistency-model Group Relative Policy Optimization (CM-GRPO)**, which reformulates a consistency sampling step as a conditional Gaussian transition and applies online Reinforcement Learning (RL) directly to this kernel, avoiding the Euler–Maruyama auxiliary process adopted in prior flow-model RL formulations. Experiments demonstrate that RAVEN surpasses recent causal video distillation baselines across quality, semantic, and dynamic degree evaluations, and that CM-GRPO provides further gains when combined with RAVEN.

## Setup

The base environment (Python 3.10, CUDA 12.8 toolkit, `uv`, system libs) is provisioned via conda from `tools/environment.yaml`. The project itself then lives in a `uv`-managed venv with Python dependencies pinned in `tools/requirements.lock` (torch 2.11+cu128, transformers 4.57, diffusers 0.37, tensorboard), plus locally-built flash-attention 2/3 and magi-attention wheels.

```sh
conda env create -f tools/environment.yaml       # creates the `raven` conda env
conda activate raven
bash tools/prepare_venv.sh                        # builds ./venv, syncs requirements.lock,
                                                  # then builds + installs flash_attn{,_3}
                                                  # and magi_attention wheels into ./assets
source venv/bin/activate
```

Targets Hopper (SM 9.0) by default — adjust `TORCH_CUDA_ARCH_LIST` / `FLASH_ATTN_CUDA_ARCHS` in `tools/prepare_venv.sh` for other GPUs. Override `CONDA_ENV`, `CUDA_HOME`, or `MAX_JOBS` via env vars if your layout differs.

Download the corresponding model checkpoints (Wan2.1-T2V-1.3B base, our released RAVEN and CM-GRPO weights, and any upstream baseline weights referenced by the configs you intend to run) yourself and point the `weight` fields in each config at the local paths.

CM-GRPO ships in three interchangeable flavors on [`mvp-lab/RAVEN`](https://huggingface.co/mvp-lab/RAVEN); pick the one that matches your config:

- **LoRA adapter** (`cmgrpo_raven_lora.safetensors`) — adapter only. CM-GRPO was trained on top of RAVEN, so the backbone loads `raven_model.pt` as the base and the adapter on top:

  ```jsonc
  "backbone": {
      "weight": "/path/to/raven_model.pt",
      "lora": {
          "enabled": true,
          "weight": "/path/to/cmgrpo_raven_lora.safetensors"
      }
  }
  ```

- **Base + LoRA bundle** (`cmgrpo_raven_full.pt`) — RAVEN base and the LoRA adapter packed into a single PEFT-wrapped state dict (the raw output of our DCP→torch checkpoint conversion). Skip the separate base weight and load the bundle through `lora.weight`:

  ```jsonc
  "backbone": {
      "lora": {
          "enabled": true,
          "weight": "/path/to/cmgrpo_raven_full.pt"
      }
  }
  ```

- **Merged** (`cmgrpo_raven_merge.pt`) — full backbone with the adapter already baked into RAVEN. Drop the `lora` block entirely and load it as the base weight:

  ```jsonc
  "backbone": {
      "weight": "/path/to/cmgrpo_raven_merge.pt"
  }
  ```

  This flavor is also compatible with the `third_party/<baseline>/` inference entrypoints as well as the original upstream baseline implementations.

RAVEN itself (`raven_model.pt`) is a single full backbone and follows the merged pattern.

## Running

Every command dispatches through `tools/multi_run.sh <jsonc>`, which wraps `torchrun` over `main.py`. Override `N` (procs per node), `NNODES`, `MASTER_ADDR`, `MASTER_PORT` via env vars; defaults autodetect from SLURM or local GPUs. Set `D=<n>` to launch under `debugpy` with `n` procs.

Train RAVEN:

```sh
bash tools/multi_run.sh configs/trials/distribution_matching_distillation/causal_wan2.1_1.3B_t2v/raven.jsonc
```

CM-GRPO on top of RAVEN:

```sh
bash tools/multi_run.sh configs/trials/group_relative_policy_optimization/causal_wan2.1_1.3B_t2v/cmgrpo_raven_raft0.35ta2aq1iq1ms0.75.jsonc
```

Sample the VBench prompt suite (videos only; scoring is in the next section):

```sh
bash tools/multi_run.sh configs/trials/vbench_t2v/causal_wan2.1_1.3B_t2v/raven.jsonc
bash tools/multi_run.sh configs/trials/vbench_t2v/causal_wan2.1_1.3B_t2v/cmgrpo.jsonc
```

Qualitative samples on the 100-prompt baseline set:

```sh
bash tools/multi_run.sh configs/trials/generate_t2v/causal_wan2.1_1.3B_t2v/raven_baseline_prompts.jsonc
bash tools/multi_run.sh configs/trials/generate_t2v/causal_wan2.1_1.3B_t2v/cmgrpo_baseline_prompts.jsonc
```

## Baseline sampling

Baseline methods compared in the paper (CausVid, Self Forcing, Reward Forcing, Causal Forcing, LongLive, Rolling Forcing) are vendored under `third_party/<baseline>/` with a consistent `inference.py / inference.sh` interface and identical sampling settings, so each baseline's output directory can be fed straight into the VBench scoring pipeline below. Each `inference.sh` wraps `torchrun -m third_party.<baseline>.inference --config_path <yaml>`:

```sh
bash third_party/causal_forcing/inference.sh   third_party/causal_forcing/configs/causal_forcing_dmd_chunkwise_vbench.yaml
bash third_party/causvid/inference.sh          third_party/causvid/configs/wan_causal_dmd_vbench.yaml
bash third_party/longlive/inference.sh         third_party/longlive/configs/longlive_vbench.yaml
bash third_party/reward_forcing/inference.sh   third_party/reward_forcing/configs/reward_forcing_vbench.yaml
bash third_party/rolling_forcing/inference.sh  third_party/rolling_forcing/configs/rolling_forcing_dmd_vbench.yaml
bash third_party/self_forcing/inference.sh     third_party/self_forcing/configs/self_forcing_dmd_vbench.yaml
```

Each `*_vbench.yaml` references the upstream-released model checkpoint and writes mp4s into `runs/<baseline>_vbench_extended/videos/`. The prompt list comes from `assets/vbench_self_forcing_extended.txt` (945 prompts, shipped).

## VBench evaluation

Scoring uses the official VBench harness, which lives in its own venv under `third_party/vbench/` to avoid clashing with the project venv.

Install once:

```sh
bash third_party/vbench/prepare_venv.sh
```

Creates `third_party/vbench/venv/`, syncs `third_party/vbench/requirements.lock`, builds `detectron2` from source, and pre-downloads every VBench dimension submodule (DINO, RAFT, AMT, CLIP, etc.) into `$VBENCH_CACHE_DIR` (default `~/.cache/vbench`).

Score any video directory (RAVEN/CM-GRPO outputs from the Running section, or any baseline output from above):

```sh
bash third_party/vbench/eval.sh runs/<run_name>/videos
```

Internally this:
1. **Static filters** the first 75 motion-dimension prompts via `vbench static_filter` to pick the highest-motion sample per prompt;
2. Runs `vbench` evaluation across all 16 dimensions via `torchrun -m vbench.launch.evaluate`;
3. Aggregates with `python -m third_party.vbench.cal_final_score` to produce the Total / Quality / Semantic scores reported in the paper.

Outputs land alongside the input dir as `<videos>_filtered/evaluation_results/`. Override `VBENCH_SAMPLES_PER_PROMPT` (default 5), `STATIC_FILTER_SAMPLES_PER_PROMPT` (default 25), `N`, `NNODES`, etc. via env vars. The eval reads `assets/vbench_all_dimension.txt` (the canonical VBench prompt list, shipped).

## Citation

If you find this work useful, please cite RAVEN. A BibTeX entry will be added when available.

```bibtex
@article{lu2026raven,
  title = {RAVEN: Real-time Autoregressive Video Extrapolation with Consistency-model GRPO},
  author = {Lu, Yanzuo and Zuo, Ronglai and Deng, Jiankang},
  year = 2026,
  journal = {arXiv preprint arXiv:2605.15190}
}
```
