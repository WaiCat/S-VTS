#!/usr/bin/env bash
set -euo pipefail

# Paper main setting: text + 2D/3D/OCR visual features, MA-MoE, and
# predictor-stage silence gate.  No audio, entity/wiki feature, or ablation path.
# Usage: DATASET=avlecture bash run.sh
#        DATASET=vtssum    bash run.sh
ROOT=$(cd "$(dirname "$0")" && pwd)
SRC="$ROOT/src"
PYTHON=${PYTHON:-python3}
NUM_GPU=${NUM_GPU:-2}
PORT=${PORT:-29500}

DATASET=${DATASET:?Set DATASET=avlecture or DATASET=vtssum}
DATA_ROOT=$(cd "${DATA_ROOT:-"$ROOT/data"}" && pwd)
export DATA_ROOT   # read by the dataset builders under src/datasets
case "$DATASET" in
  avlecture)
    BUILDER=avlecture
    FEAT_ROOT="$DATA_ROOT/avlecture/labeled"
    VIS3D_DIM=2048
    DEFAULT_SILENCE_ROOT="$DATA_ROOT/avlecture/silence_db30_results/db30_min1p0s"
    TEST_FILE="$DATA_ROOT/avlecture/test.jsonl"
    # AVLecture fine-tunes from the MMVTS pseudo-pretrain checkpoint. It was trained with
    # gate fusion, so the 768-d classifier is re-initialised for the 1536-d concat m_i.
    PRETRAIN_CKPT=${PRETRAIN_CKPT:?Set PRETRAIN_CKPT to the MMVTS pseudo-pretrain checkpoint for AVLecture}
    INIT_ARGS=(--init_model False --model_name_or_path "$PRETRAIN_CKPT" --ignore_mismatched_sizes True)
    ;;
  vtssum)
    BUILDER=vt_ssum_30min_clean2_stagerealigned_relabel_only
    FEAT_ROOT="$DATA_ROOT/vt_ssum"
    VIS3D_DIM=512
    DEFAULT_SILENCE_ROOT="$DATA_ROOT/vt_ssum/silence_db30"
    TEST_FILE="$DATA_ROOT/$BUILDER/processed_jsonl/test.jsonl"
    INIT_ARGS=(--init_model True)
    ;;
  *) echo "[ERROR] DATASET must be avlecture or vtssum" >&2; exit 1 ;;
esac
SILENCE_ROOT=${SILENCE_ROOT:-$DEFAULT_SILENCE_ROOT}   # -30 dB silence intervals
SILENCE_CACHE_DIR=${SILENCE_CACHE_DIR:-"$DATA_ROOT/paper_silence_cache/$DATASET-fdef2-min1-near5"}
DATASET_CACHE_DIR=${DATASET_CACHE_DIR:-"$ROOT/cache/$DATASET-near5-v3d$VIS3D_DIM"}
OUT_ROOT=${OUT_ROOT:-"$ROOT/output/$DATASET"}
SEEDS=${SEEDS:-"42 59 88"}
LR=${LR:-5e-5}
WEIGHT_LABEL_ZERO=${WEIGHT_LABEL_ZERO:-0.65}
EPOCHS=${EPOCHS:-5}
GRAD_ACCUM=${GRAD_ACCUM:-$(( 8 / NUM_GPU ))}
if [ "$GRAD_ACCUM" -lt 1 ]; then GRAD_ACCUM=1; fi

for SEED in $SEEDS; do
OUT_DIR="$OUT_ROOT/seed$SEED-$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
"$PYTHON" -m torch.distributed.run --nproc_per_node="$NUM_GPU" --master_port="$PORT" \
  "$SRC/main_multimodal.py" \
  --dataset_name "$SRC/datasets/$BUILDER" \
  --dataset_root_dir "$DATA_ROOT" \
  --dataset_cache_dir "$DATASET_CACHE_DIR" \
  --metric_name "$SRC/metrics/seqeval.py" \
  "${INIT_ARGS[@]}" --do_train True --do_eval True --do_predict True --seed "$SEED" \
  --text_encoder_name_or_path allenai/longformer-base-4096 \
  --vis2d_encoder_name cn_clip_vit_b_16 \
  --use_raw_shot False --max_vis_seq_length 2048 \
  --use_vis2d True --use_vis3d True --use_vis_ocr True --use_audio False \
  --vis2d_feature_cache_dir "$FEAT_ROOT/feats" \
  --vis3d_feature_cache_dir "$FEAT_ROOT/feats_3d" \
  --vis_ocr_feature_cache_dir "$FEAT_ROOT/feats_ocr" \
  --hidden_size_vis2d 512 --hidden_size_vis3d "$VIS3D_DIM" --hidden_size_vis_ocr 768 \
  --freeze_text_encoder False --freeze_vis2d_encoder True \
  --fuse_type cat_t_v --cross_encoder_type ma_moe --cross_encoder_lr 5e-5 \
  --num_cross_encoder_layers 1 --num_cross_encoder_heads 12 --hidden_dropout_prob 0.15 \
  --weight_label_zero "$WEIGHT_LABEL_ZERO" \
  --do_modality_cl True --modality_cl_lw 0.5 \
  --do_align_av False --do_align_at False --do_align_tv True --align_tv_weight 0.8 \
  --do_topic_mm_cl True --topic_mm_cl_lw 0.5 \
  --topic_cl_type list --topic_cl_choice near --topic_cl_fct simcse \
  --topic_mm_cl_pos_k 1 --topic_mm_cl_neg_k 3 \
  --use_silence True --silence_root "$SILENCE_ROOT" \
  --silence_feature flag_dur_dist --silence_min_duration_sec 1.0 \
  --silence_near_sec 5.0 --silence_window_expand_sec 0.0 --silence_max_dist 5.0 \
  --silence_dim 3 --silence_feature_cache_dir "$SILENCE_CACHE_DIR" \
  --silence_fusion gate --silence_inject head_only \
  --max_seq_length 2048 --num_gpu "$NUM_GPU" \
  --learning_rate "$LR" --weight_decay 0.01 --warmup_ratio 0.1 --lr_scheduler_type cosine \
  --per_device_train_batch_size 1 --gradient_accumulation_steps "$GRAD_ACCUM" \
  --per_device_eval_batch_size 4 --num_train_epochs "$EPOCHS" \
  --evaluation_strategy steps --eval_cnt "$EPOCHS" --eval_accumulation_steps 1000 \
  --load_best_model_at_end --save_total_limit 2 --metric_for_best_model overall_f1 \
  --overwrite_cache False --preprocessing_num_workers 1 --return_detailed_metrics True \
  --test_data_name "$BUILDER" --output_dir "$OUT_DIR" --overwrite_output_dir ${EXTRA_ARGS:-} 2>&1 | tee "$OUT_DIR/run.log"

PRED_FILE="$OUT_DIR/predict_${BUILDER}_max_seq2048.txt"
"$PYTHON" "$SRC/evaluate.py" -d "$TEST_FILE" -p "$PRED_FILE" -l en \
  | tee "$OUT_DIR/example_level_predict_${BUILDER}_max_seq2048_results_str_metric.txt"
done
