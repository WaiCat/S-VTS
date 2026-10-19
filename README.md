# Silence-Guided Video Topic Segmentation

Code for *Leveraging Silence as a Transition-Aware Cue for Topically Coherent Topic Segmentation in Lecture Videos*.

The model builds on MMVTS. Text and visual clip features go through Multimodal Fusion Layers, the same as in MMVTS. A silence-conditioned gate is then applied at the predictor stage, just before boundary classification. Only the main setting reported in the paper is included. Ablation variants are not part of this release.

## Layout

```
S-VTS/
├── run.sh                 # train + predict + evaluate, 3 seeds
└── src/
    ├── main_multimodal.py # training / prediction entry point
    ├── evaluate.py        # F1, BS@30, F1@30, mIoU, Avg
    ├── silence_utils.py   # silence features (Eq. 5-7)
    ├── datasets/          # AVLecture and VT-SSUM dataset builders
    └── models/            # encoders, MFL (MA-MoE), predictor, losses
```

## Equations and code

| Paper | Code |
|---|---|
| Eq. 1-2: projections W_t, W_v (2D, 3D and OCR features concatenated before W_v) | `src/models/projector/linear_projector.py`, `src/models/vis_encoder/vis_encoder.py` |
| Eq. 3: MFL, i.e. self-attention over the concatenated text-visual sequence followed by an MoE FFN | `src/models/cross_encoder/ma_moe_encoder.py`, `src/models/cross_encoder/moe.py` |
| Eq. 4: m_i = h^v_i ⊕ h^t_i | `BasePredictor.fuse_features` in `src/models/predictor/base_predictor.py` |
| Eq. 5-7: s_i = [f^flag; f^dur; f^dist] | `compute_silence_features` in `src/silence_utils.py` |
| Eq. 8-9: g_i = σ(W_s s_i + W_m m_i + b_g), m̃_i = m_i ⊙ g_i | `LinearPredictor.apply_silence_fusion` in `src/models/predictor/linear_predictor.py` |
| Eq. 10-11: z_i = W_c m̃_i + b_c, softmax / argmax | `LinearPredictor.classifier` |
| Eq. 12: L = L_vts + λL_balance + θL_mcssl + γL_cma | `src/models/modules/loss_layer.py`, `src/models/cross_encoder/moe.py` |

The auxiliary losses use m_i before the gate. The classifier uses the gated m̃_i.

## Setup

Python 3.10, PyTorch 1.12.1 and transformers 4.26.0. Install everything with:

```bash
pip install -r src/requirements.txt
```

## Data

`run.sh` reads everything from `DATA_ROOT`, which defaults to `./data`.

```
DATA_ROOT/
├── avlecture/
│   ├── {train,dev,test}.jsonl
│   ├── labeled/{feats,feats_3d,feats_ocr}/<split>/<lecture>/clip-XXXX.npy
│   └── silence_db30_results/db30_min1p0s/**/<lecture>.silence.json
├── vt_ssum_30min_clean2_stagerealigned_relabel_only/processed_jsonl/{train,dev,test}.jsonl
└── vt_ssum/
    ├── {feats,feats_3d,feats_ocr}/<split>/<video>/clip-XXXXX.npy
    └── silence_db30/<video>.silence.json
```

- **jsonl:** one video per line, with these fields:
  - `example_id`: `<prefix>@@<lecture>`
  - `text`: one transcript string per clip
  - `labels`: 1 marks the last clip of a topic
  - `stet`: `[start, end]` in seconds for each clip
  - `topic_end_seconds`: ground-truth boundary times, used by `evaluate.py`
- **Visual features:** 2D is CN-CLIP ViT-B/16 (512-d). 3D is 2048-d on AVLecture and 512-d on VT-SSUM. OCR is 768-d.
- **Silence intervals:** detected from the audio with a −30 dB threshold and stored as `[{"start": s, "end": e}, ...]`. The detection step itself is not included here.
- **VT-SSUM:** only videos of at least 30 minutes with valid transcript-timestamp alignment are used (realigned split, 249 videos).

## Training and evaluation

```bash
PRETRAIN_CKPT=/path/to/mmvts_pretrain_ckpt DATASET=avlecture bash run.sh
DATASET=vtssum    bash run.sh
```

Each command trains with seeds 42, 59 and 88, predicts on the test split, and runs `evaluate.py`. Results are written to `output/<dataset>/seed<seed>-<time>/`:

- `predict_<dataset>_max_seq2048.txt` contains the predictions.
- `example_level_predict_<dataset>_max_seq2048_results_str_metric.txt` contains the scores. The `threshold=0.5` line lists F1 / BS@30 / F1@30 / mIoU / Avg.

**AVLecture initialisation:** AVLecture is fine-tuned from the MMVTS pseudo-pretrain checkpoint. Set its path with `PRETRAIN_CKPT`. The checkpoint was trained with a 768-d fused representation, so the classifier is re-initialised for the 1536-d concatenated m_i, and every other weight is loaded. VT-SSUM starts from `allenai/longformer-base-4096`.

**Environment variables:**

| Variable | Default | Meaning |
|---|---|---|
| `DATASET` | (required) | `avlecture` or `vtssum` |
| `DATA_ROOT` | `./data` | data directory |
| `PRETRAIN_CKPT` | (required for AVLecture) | MMVTS pseudo-pretrain checkpoint |
| `PYTHON` | `python3` | interpreter |
| `NUM_GPU` | 2 | GPUs; gradient accumulation is set to `8 / NUM_GPU` |
| `SEEDS` | `42 59 88` | seeds to run |
| `OUT_ROOT` | `output/<dataset>` | output directory |
| `EXTRA_ARGS` | (empty) | extra arguments for `main_multimodal.py`, e.g. `--max_steps 4` for a quick check |

## Hyperparameters

These are the paper settings (Appendix A.5), all fixed in `run.sh`.

| Setting | Value |
|---|---|
| Optimizer | AdamW, lr 5e-5, weight decay 0.01 |
| Schedule | cosine, warmup ratio 0.1, 5 epochs |
| Batch | effective batch 8 (2 GPUs × per-device 1 × accumulation 4) |
| Model | max sequence length 2048, dropout 0.15, FP32, gradient clipping 1.0 |
| MoE | 4 experts, top-2 routing, hidden size 3072 |
| Loss weights | λ = 1.0 (MoE balance), θ = 0.5 (L_mcssl), γ = 0.5 × 0.8 = 0.40 (L_cma), class-weighted cross-entropy with weight 0.65 on the boundary class |
| Silence | τ_min = 1.0 s, δ = 5.0 s, d_max = 5.0 s |

The model checkpoint used for prediction is the one with the best validation `overall_f1`.

## License

The code builds on the MMVTS codebase (SpokenNLP, Apache License 2.0).
