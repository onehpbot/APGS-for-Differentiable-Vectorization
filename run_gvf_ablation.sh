#!/bin/bash
set -e

CYAN='\033[1;36m'
GREEN='\033[1;32m'
NC='\033[0m' 

echo -e "${CYAN}🚀 开始执行 GVF 先验消融实验...${NC}"

# 设置通用的基础参数 (确保除了 GVF 权重外，其他变量完全一致)
TARGET_CURVES=1024
L1_W=1.0
L2_W=1.0
SSIM_W=1.0
WIDTH_W=10.0 # 保持宽度正则化开启以稳定形状

# ==========================================================
# 实验 A: 无 GVF 先验 (Baseline)
# ==========================================================
echo -e "\n${GREEN}>>> [1/2] 运行 w/o GVF (snake_w = 0.0)${NC}"
python scripts/main.py \
    --exp_name "GVF_Ablation_wo_GVF" \
    --snake_w 0.0 \
    --lambda_w $WIDTH_W \
    --l1_weight $L1_W \
    --l2_weight $L2_W \
    --ssim_weight $SSIM_W \
    --target_curves $TARGET_CURVES \
    --log_interval 50 # 每 50 步记录一次数据用于画收敛曲线

# ==========================================================
# 实验 B: 有 GVF 先验 (APGS 完整版)
# ==========================================================
echo -e "\n${GREEN}>>> [2/2] 运行 w/ GVF (snake_w = 0.5)${NC}"
python scripts/main.py \
    --exp_name "GVF_Ablation_w_GVF" \
    --snake_w 0.5 \
    --lambda_w $WIDTH_W \
    --l1_weight $L1_W \
    --l2_weight $L2_W \
    --ssim_weight $SSIM_W \
    --target_curves $TARGET_CURVES \
    --log_interval 50

echo -e "\n${CYAN}🎉 GVF 消融实验执行完毕！请前往 output/ 目录查看结果。${NC}"