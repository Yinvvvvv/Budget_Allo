#!/bin/bash

# ==== 从 ARNOLD 环境变量获取配置 ====
NODE_RANK="${ARNOLD_ID:-0}"
NNODES="${ARNOLD_WORKER_NUM:-1}"
GPUS_PER_NODE="${ARNOLD_WORKER_GPU:-4}"
HEAD_IP_RESOLVED="${ARNOLD_WORKER_0_HOST:-127.0.0.1}"

# ==== 解析端口列表 ====
ARNOLD_WORKER_0_PORT_ARRAY=(${ARNOLD_WORKER_0_PORT//,/ })
DP_PORT="${DP_PORT:-${ARNOLD_WORKER_0_PORT_ARRAY[0]:-13345}}"
echo "[INFO] Using DP_PORT=${DP_PORT} for vLLM data-parallel RPC"

# ==== 模型和服务配置 ====
MODEL_DIR="/home/$USER/models/Qwen2.5-7B-Instruct"
API_PORT="${API_PORT:-8000}"

# ==== DP 并行度配置 ====
DP_LOCAL="${GPUS_PER_NODE}"
DP_TOTAL="$(( NNODES * DP_LOCAL ))"
START_RANK="$(( NODE_RANK * DP_LOCAL ))"
echo "[INFO] NODE_RANK=${NODE_RANK} NNODES=${NNODES} GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "[INFO] HEAD_IP=${HEAD_IP_RESOLVED} DP_PORT=${DP_PORT}"
echo "[INFO] DP_TOTAL=${DP_TOTAL} DP_LOCAL=${DP_LOCAL} START_RANK=${START_RANK}"

if [[ "${NODE_RANK}" -eq 0 ]]; then
  # ===== Rank 0（head / 提供 API）=====
  vllm serve "${MODEL_DIR}" \
    --data-parallel-size "${DP_TOTAL}" \
    --tensor-parallel-size 1 \
    --data-parallel-size-local "${DP_LOCAL}" \
    --api-server-count "${DP_TOTAL}" \
    --host 0.0.0.0 --port "${API_PORT}" \
    --data-parallel-address "${HEAD_IP_RESOLVED}" \
    --data-parallel-rpc-port "${DP_PORT}"
else
  # ===== 其他 Rank（worker / headless）=====
  vllm serve "${MODEL_DIR}" \
    --data-parallel-size "${DP_TOTAL}" \
    --tensor-parallel-size 1 \
    --data-parallel-size-local "${DP_LOCAL}" \
    --headless \
    --host 0.0.0.0 --port "${API_PORT}" \
    --data-parallel-start-rank "${START_RANK}" \
    --data-parallel-address "${HEAD_IP_RESOLVED}" \
    --data-parallel-rpc-port "${DP_PORT}"
fi