#!/bin/bash
# Run multi-turn SFT on a pretrained checkpoint (plain shell, no SLURM).
#
# Usage:
#   bash run_sft.sh
#
# Override paths/knobs via environment variables, e.g.:
#   PRETRAIN_ROOT=/data/checkpoints/C_6p5e19 \
#   TRAIN_DATA_DIR=/data/sft/cot_data \
#   NUM_GPUS=2 WANDB_ENTITY=my-team \
#   bash run_sft.sh
#
# This is the non-SLURM port of the original train_sft_multi_turn.sh: instead of
# submitting an sbatch job per (spec, cot_type) it runs `accelerate launch`
# directly in the foreground.

set -euo pipefail

# Repo root = directory of this script (contains config/, scripts/, training/, ...).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Configurable inputs (override via env) ---------------------------------
# Root holding the pretrain checkpoints. Each model lives at
#   ${PRETRAIN_ROOT}/{total_compute}_{modelsize}_alpha{alpha}/final/
PRETRAIN_ROOT="${PRETRAIN_ROOT:-${REPO_ROOT}/checkpoints/C_6p5e19}"
# Directory of generated CoT SFT data (output of the CoT generation step).
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${REPO_ROOT}/data/sft/cot_data_puzzle_seq}"
DATA_NAME="${DATA_NAME:-generated_cot}"
# Weights & Biases team/entity. Leave empty to use your default wandb account.
WANDB_ENTITY="${WANDB_ENTITY:-}"
NUM_GPUS="${NUM_GPUS:-2}"

CONFIG="${REPO_ROOT}/config/configs/qwen_multiturn_sft/sft_config_multi_path_2048.yaml"
MAP_CONFIG_FILE="${REPO_ROOT}/sft_max_train_files_map.json"

lr="3e-4"

# Each spec entry: total_compute|modelsize|alpha|beta
pretrain_specs=(
  "6p5e19|680m|0.750|0.030"
)

# CoT field / type / block size (kept index-aligned).
cot_fields=("cot_by_method.trajectory_sep.cot_format_no_labels")
cot_types=("trajectory_sep_no_labels")
block_sizes=("3072")

model_id_from_spec() {   # SFT run-name id (includes modelsize)
  printf "C%s_%s_alpha%s_beta%s" "$1" "$2" "$3" "$4"
}
pretrain_dir_from_spec() {  # pretrain checkpoint dir name
  printf "%s_%s_alpha%s" "$1" "$2" "$3"
}

max_train_files_for_spec() {
  python - "$MAP_CONFIG_FILE" "$1" "$2" "$3" "$4" <<'PY'
import json, pathlib, sys
cfg_path = pathlib.Path(sys.argv[1])
total_compute, modelsize, alpha, beta = map(str, sys.argv[2:6])
cfg = json.loads(cfg_path.read_text())
default_value = cfg.get("default_max_train_files")
matched = None
for item in cfg.get("entries_by_spec", []):
    if (str(item.get("total_compute")) == total_compute
            and str(item.get("modelsize")) == modelsize
            and str(item.get("alpha")) == alpha
            and str(item.get("beta")) == beta):
        matched = item.get("max_train_files"); break
if matched is not None:
    print(f"{int(matched)}\tentry")
elif default_value is not None:
    print(f"{int(default_value)}\tdefault")
else:
    raise SystemExit(f"No max_train_files mapping for ({total_compute},{modelsize},{alpha},{beta})")
PY
}

for spec in "${pretrain_specs[@]}"; do
  IFS="|" read -r total_compute modelsize alpha beta <<<"${spec}"
  model_id="$(model_id_from_spec "${total_compute}" "${modelsize}" "${alpha}" "${beta}")"

  pretrain_ckpt_name="$(pretrain_dir_from_spec "${total_compute}" "${modelsize}" "${alpha}")"
  pretrain_dir="${PRETRAIN_ROOT}/${pretrain_ckpt_name}/final"

  if [[ -f "${pretrain_dir}/config.json" ]]; then
    pretrained_model="${pretrain_dir}"
  elif [[ -f "${pretrain_dir}/hf_model/config.json" ]]; then
    pretrained_model="${pretrain_dir}/hf_model"
  else
    echo "[WARN] HF model files not found at ${pretrain_dir} — trainer will resolve from spec"
    pretrained_model="${pretrain_dir}"
  fi

  pretrained_weights=""
  for w in \
    "${pretrain_dir}/model.safetensors" "${pretrain_dir}/pytorch_model.bin" \
    "${pretrained_model}/model.safetensors" "${pretrained_model}/pytorch_model.bin"; do
    [[ -f "${w}" ]] && { pretrained_weights="${w}"; break; }
  done

  IFS=$'\t' read -r max_train_files max_source < <(max_train_files_for_spec "${total_compute}" "${modelsize}" "${alpha}" "${beta}")
  [[ "${max_source}" != "entry" ]] && echo "[WARN] No exact spec mapping for ${model_id}; using default_max_train_files=${max_train_files}"

  for j in "${!cot_fields[@]}"; do
    cot_field="${cot_fields[$j]}"
    cot_type="${cot_types[$j]}"
    block_size="${block_sizes[$j]}"

    extra_args=" --pretrained-model ${pretrained_model}"
    [[ -n "${pretrained_weights}" ]] && extra_args+=" --pretrained-weights ${pretrained_weights}"
    extra_args+=" --naming-scheme pretrain_spec"
    extra_args+=" --total-compute ${total_compute} --modelsize ${modelsize} --alpha ${alpha} --beta ${beta}"
    extra_args+=" --cot-field ${cot_field} --cot-type ${cot_type}"
    [[ -n "${block_size}" ]] && extra_args+=" --block-size ${block_size}"
    [[ -n "${TRAIN_DATA_DIR}" ]] && extra_args+=" --train-files ${TRAIN_DATA_DIR}"
    [[ -n "${DATA_NAME}" ]] && extra_args+=" --data-name ${DATA_NAME}"
    [[ -n "${WANDB_ENTITY}" ]] && extra_args+=" --wandb-entity ${WANDB_ENTITY}"
    extra_args+=" --max-train-files ${max_train_files}"

    echo "=========================================="
    echo "[SFT] model_id=${model_id}"
    echo "[SFT] pretrain_dir=${pretrain_dir}"
    echo "[SFT] max_train_files=${max_train_files} cot_type=${cot_type}"
    echo "=========================================="

    export PYTHONWARNINGS="ignore"
    export NCCL_TIMEOUT=1800000
    export TORCH_NCCL_BLOCKING_WAIT=1

    accelerate launch --num_processes "${NUM_GPUS}" \
      "${REPO_ROOT}/scripts/train/run_sft.py" \
      --config "${CONFIG}" --lr "${lr}"${extra_args}
  done
done
