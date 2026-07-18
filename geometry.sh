#!/bin/bash
set -e

CYAN='\033[1;36m'
GREEN='\033[1;32m'
NC='\033[0m' 

echo -e "${CYAN}🚀 开始执行高级消融实验与参数分析...${NC}"

# ==========================================================
# Group A: 几何先验消融 (保持基础 Loss L1+L2+SSIM 全开)
# ==========================================================
echo -e "\n${CYAN}--- Group A: 几何先验 (GVF & Width) ---${NC}"

# A1: 都没有 (Baseline)
echo -e "${GREEN}>>> 运行 A1: No Priors${NC}"
python scripts/main.py --exp_name "Ablation_Prior_None" --snake_w 0.0 --lambda_w 0.0

# A2: 仅宽度正则化
echo -e "${GREEN}>>> 运行 A2: Only Width Loss${NC}"
python scripts/main.py --exp_name "Ablation_Prior_Width" --snake_w 0.0 --lambda_w 10.0

# A3: 仅 GVF 损失
echo -e "${GREEN}>>> 运行 A3: Only GVF Loss${NC}"
python scripts/main.py --exp_name "Ablation_Prior_GVF" --snake_w 0.5 --lambda_w 0.0

# A4: 全开 (已有默认参数就是全开，也可显式指定)
echo -e "${GREEN}>>> 运行 A4: Full Priors (GVF + Width)${NC}"
python scripts/main.py --exp_name "Ablation_Prior_Full" --snake_w 0.5 --lambda_w 10.0