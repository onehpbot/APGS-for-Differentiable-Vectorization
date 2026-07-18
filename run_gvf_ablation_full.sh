#!/bin/bash
# ==============================================================================
# GVF 全量消融（w/o vs w/ GVF），跨 data_dir 下所有子文件夹的所有图片。
# 目的：同时拿 GVF 在 (1) 最终质量 (2) 收敛速度 上的证据。
#
# 两个条件除 GVF 外完全一致（同 seed、同 loss、同拓扑），保证 paired 对照：
#   A. w/o GVF : --snake_w 0      --gvf_guided_densify 0   (GVF 损失关 + densify 不引导)
#   B. w/ GVF  : --snake_w $SNAKE_W --gvf_guided_densify 1  (GVF 损失开 + densify 沿边缘定向)
#
# 跑完后用： python scripts/aggregate_gvf_ablation.py \
#                 --wo_dir output/GVF_wo --w_dir output/GVF_w
#
# 服务器环境：conda activate LIG （torch 2.5.1 / cu121 / py3.10）。本机(Windows)勿用。
# 不用 set -e：单图失败已在 main.py 内捕获跳过；进程级崩溃也尽量让两个条件都跑到。
# ==============================================================================

# ---- 可调参数 ----
DATA_DIR=${DATA_DIR:-/data/workspace_PH/dataset/AP}   # 覆盖：DATA_DIR=xxx bash run_gvf_ablation_full.sh
NUM_EPOCHS=${NUM_EPOCHS:-5000}
TARGET_CURVES=${TARGET_CURVES:-2048}
LOG_INTERVAL=${LOG_INTERVAL:-100}     # 收敛曲线采样：5000/100=51 点
SEED=${SEED:-42}
SNAKE_W=${SNAKE_W:-50}                # 已验证的 operative 值（见 memory）
L1_W=${L1_W:-1.0}; L2_W=${L2_W:-1.0}; SSIM_W=${SSIM_W:-1.0}; WIDTH_W=${WIDTH_W:-10.0}
# -----------------

COMMON_ARGS="--data_dir $DATA_DIR --num_epochs $NUM_EPOCHS --target_curves $TARGET_CURVES \
--log_interval $LOG_INTERVAL --seed $SEED \
--l1_weight $L1_W --l2_weight $L2_W --ssim_weight $SSIM_W --lambda_w $WIDTH_W"

echo "=============================================================="
echo " GVF 全量消融 | data_dir=$DATA_DIR | epochs=$NUM_EPOCHS | seed=$SEED"
echo " snake_w(operative)=$SNAKE_W | log_interval=$LOG_INTERVAL"
echo "=============================================================="

echo -e "\n>>> [1/2] w/o GVF  (snake_w=0, densify 不引导)"
python scripts/main.py --exp_name GVF_wo $COMMON_ARGS --snake_w 0 --gvf_guided_densify 0

echo -e "\n>>> [2/2] w/ GVF  (snake_w=$SNAKE_W, densify 沿边缘定向)"
python scripts/main.py --exp_name GVF_w $COMMON_ARGS --snake_w $SNAKE_W --gvf_guided_densify 1

echo -e "\n=============================================================="
echo " ✅ 两个条件跑完。现在聚合分析："
echo "    python scripts/aggregate_gvf_ablation.py --wo_dir output/GVF_wo --w_dir output/GVF_w"
echo "=============================================================="
