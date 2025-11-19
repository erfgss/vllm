#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Setup Verl + vLLM environment, run GSM8K Qwen0.5B PEFT SFT example, then test with vLLM

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERL_REPO="https://github.com/volcengine/verl.git"
VERL_BRANCH="main"
VERL_DIR="${REPO_ROOT}/verl"
TARGET_DIR="${VERL_DIR}/examples/data_preprocess"
SFT_OUTPUT="${VERL_DIR}/gsm8k"
TRAIN_SCRIPT="${VERL_DIR}/examples/sft/gsm8k/run_qwen_05_peft.sh"
TRAIN_path="${VERL_DIR}/examples/sft/gsm8k"
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
MODEL_DIR="${VERL_DIR}/models/Qwen2.5-0.5B-Instruct"
train_epochs=2
data_dir="${REPO_ROOT}/data/gsm8k"
GPU_node=8

echo "VERL_REPO=${VERL_REPO}"
echo "VERL_BRANCH=${VERL_BRANCH}"
echo "VERL_DIR=${VERL_DIR}"
echo "TARGET_DIR=${TARGET_DIR}"
echo "SFT_OUTPUT=${SFT_OUTPUT}"
echo "TRAIN_SCRIPT=${TRAIN_SCRIPT}"
echo "TRAIN_path=${TRAIN_path}"
echo "MODEL_ID=${MODEL_ID}"
echo "MODEL_DIR=${MODEL_DIR}"
echo "train_epochs=${train_epochs}"
echo "data_dir=${data_dir}"
echo "GPU_node=${GPU_node}"

echo "===== Setting up Verl environment ====="

if [ -d "${VERL_DIR}" ]; then
    echo "Verl exists, skip clone"
else
    git clone --branch "${VERL_BRANCH}" --single-branch "${VERL_REPO}" "${VERL_DIR}"
fi
echo "Entering ${VERL_DIR} ..."
cd "${VERL_DIR}"
uv pip install --no-deps -e .
uv pip install -e .[vllm]

echo "Entering ${TARGET_DIR} ..."
cd "${TARGET_DIR}"
echo "Running gsm8k.py "
python3 gsm8k.py --local_save_dir "${data_dir}";

echo "===== gsm8k.py preprocessing completed! ====="

echo "===== Downloading model: ${MODEL_ID} ====="
echo "Target directory: ${MODEL_DIR}"
huggingface-cli download "${MODEL_ID}" --resume-download --local-dir "${MODEL_DIR}"

echo "===== Downloading model: ${MODEL_ID} completed! ====="
echo "Entering ${TRAIN_path} ..."
cd "${TRAIN_path}"
echo "===== Starting SFT Training ====="
bash "${TRAIN_SCRIPT}" "${GPU_node}" "${SFT_OUTPUT}" \
    data.train_files="${data_dir}/train.parquet" \
    data.val_files="${data_dir}/test.parquet" \
    model.partial_pretrain="${MODEL_DIR}" \
    trainer.total_epochs="${train_epochs}" \
    trainer.logger=tensorboard
echo "===== End SFT Training ====="

echo "===== Model Restoration ====="
step=$((29 * train_epochs))

LOCAL_DIR="${SFT_OUTPUT}/global_step_${step}"
TARGET_DIR_model="${SFT_OUTPUT}/global_step_${step}_m"

python "${VERL_DIR}/scripts/legacy_model_merger.py" merge \
  --backend fsdp \
  --local_dir "${LOCAL_DIR}" \
  --target_dir "${TARGET_DIR_model}"

echo "Merging FSDP weights from: ${LOCAL_DIR}"
echo "===== test model ====="
echo "===== latency test ====="
vllm bench latency --model "${TARGET_DIR_model}" --input-len 512 --output-len 128
echo "===== throughput test ====="
vllm bench throughput --model "${TARGET_DIR_model}" --input-len 512  --output-len 128
echo "Run_qwen_05_peft tests completed!"