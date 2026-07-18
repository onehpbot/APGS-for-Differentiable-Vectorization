#!/bin/bash

# ==============================================================================
# APGS Core Loss Ablation Study
# Automatically processes all images in the directory and generates CSV reports
# Usage: bash run_ablations.sh
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e 

# Define console output colors
# CYAN='\033[1;36m'
# GREEN='\033[1;32m'
# NC='\033[0m' # No Color

echo -e "${CYAN}🚀 Starting the 3-group Loss function ablation study...${NC}"

# Experiment 1: L1
EXP1_NAME="Ablation_Loss_PureL1"
echo -e "\n${GREEN}>>> Running [1/7]: ${EXP1_NAME}${NC}"
python scripts/main.py \
    --exp_name $EXP1_NAME \
    --l1_weight 1.0 \
    --l2_weight 0.0 \
    --ssim_weight 0.0

# Experiment 2: L2 
EXP2_NAME="Ablation_Loss_PureL2"
echo -e "\n${GREEN}>>> Running [2/7]: ${EXP2_NAME}${NC}"
python scripts/main.py \
    --exp_name $EXP2_NAME \
    --l1_weight 0.0 \
    --l2_weight 1.0 \
    --ssim_weight 0.0

# Experiment 3: SSIM
EXP3_NAME="Ablation_Loss_PureSSIM"
echo -e "\n${GREEN}>>> Running [3/7]: ${EXP3_NAME}${NC}"
python scripts/main.py \
    --exp_name $EXP3_NAME \
    --l1_weight 0.0 \
    --l2_weight 0.0 \
    --ssim_weight 1.0

# Experiment 4: L1 + L2
EXP4_NAME="Ablation_Loss_L1_L2"
echo -e "\n${GREEN}>>> Running [4/7]: ${EXP4_NAME}${NC}"
python scripts/main.py \
    --exp_name $EXP4_NAME \
    --l1_weight 1.0 \
    --l2_weight 1.0 \
    --ssim_weight 0.0

# Experiment 5: L1 + SSIM
EXP5_NAME="Ablation_Loss_L1_SSIM"
echo -e "\n${GREEN}>>> Running [5/7]: ${EXP5_NAME}${NC}"
python scripts/main.py \
    --exp_name $EXP5_NAME \
    --l1_weight 1.0 \
    --l2_weight 0.0 \
    --ssim_weight 1.0

# Experiment 6: L2 + SSIM
EXP6_NAME="Ablation_Loss_L2_SSIM"
echo -e "\n${GREEN}>>> Running [6/7]: ${EXP6_NAME}${NC}"
python scripts/main.py \
    --exp_name $EXP6_NAME \
    --l1_weight 0.0 \
    --l2_weight 1.0 \
    --ssim_weight 1.0

# Experiment 7: L1 + L2 + SSIM (Full version)
EXP7_NAME="Ablation_Loss_L1_L2_SSIM"
echo -e "\n${GREEN}>>> Running [7/7]: ${EXP7_NAME}${NC}"
python scripts/main.py \
    --exp_name $EXP7_NAME \
    --l1_weight 1.0 \
    --l2_weight 1.0 \
    --ssim_weight 1.0

echo -e "\n${CYAN}🎉 All ablation experiments completed! Please check the metrics_report.csv file in the output directory.${NC}"