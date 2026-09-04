#!/usr/bin/env bash
# Mac 上跑通 pickcup IL pipeline：批量录制 -> ACT 训练 -> 仿真闭环评估
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PICKTASK="$ROOT/picktask"
SCRIPTS="$PICKTASK/scripts"
export PYTHONPATH="$ROOT/lerobot/src:${PYTHONPATH:-}"

DATASET_NAME="${DATASET_NAME:-pickcup_train_mac}"
EPISODES="${EPISODES:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-2000}"
EVAL_EPISODES="${EVAL_EPISODES:-3}"
BATCH_SIZE="${BATCH_SIZE:-4}"

DATASET_ROOT="$PICKTASK/pickcupdata/train/$DATASET_NAME"
OUTPUT_DIR="$ROOT/outputs/train/act_${DATASET_NAME}"
POLICY_DIR="$OUTPUT_DIR"

echo "=== 1/3 批量 scripted 录制 (${EPISODES} episodes) ==="
python "$SCRIPTS/pickcup_batch_record.py" --episodes "$EPISODES" --output "$DATASET_NAME" --seed 42

echo ""
echo "=== 2/3 ACT 训练 (${TRAIN_STEPS} steps, device=auto/MPS) ==="
python "$SCRIPTS/pickcup_train_act.py" \
  --dataset "$DATASET_ROOT" \
  --repo-id "local/${DATASET_NAME}" \
  --output "$OUTPUT_DIR" \
  --steps "$TRAIN_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --device auto

echo ""
echo "=== 3/3 仿真闭环评估 (${EVAL_EPISODES} episodes) ==="
python "$SCRIPTS/pickcup_sim_eval.py" \
  --policy "$POLICY_DIR" \
  --dataset "$DATASET_ROOT" \
  --episodes "$EVAL_EPISODES" \
  --device auto \
  --seed 7

echo ""
echo "Pipeline 完成。"
echo "  dataset: $DATASET_ROOT"
echo "  policy:  $POLICY_DIR"
