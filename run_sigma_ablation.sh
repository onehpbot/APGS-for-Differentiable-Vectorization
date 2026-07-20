#!/bin/bash
# ==============================================================================
# σ_edge 退火消融：验证当前 0.002→0.0001 是否最优。
# 5 组组合把当前值包在中间，并含一组"不退火"(s_const)隔离退火效应本身。
# 为纯净隔离 σ（渲染参数），本脚本关掉 GVF（snake_w=0 + densify 不引导）。
#   想看 σ×GVF 交互，把下面 SNAKE_W/GVF_GUIDED 改成 50/1 即可。
#
# 组合在下面 COMBOS 数组里，随便增删。每组跨 data_dir 所有子文件夹所有图。
# 跑完聚合（多条件版）：
#   python scripts/aggregate_ablation.py \
#     --cond s_soft=output/sigma_s_soft --cond s_mid=output/sigma_s_mid \
#     --cond s_default=output/sigma_s_default --cond s_hard=output/sigma_s_hard \
#     --cond s_const=output/sigma_s_const --ref s_default
#
# 服务器：conda activate LIG。不用 set -e（单图失败已 try/except 跳过）。
# ==============================================================================

# ---- 可调参数 ----
DATA_DIR=${DATA_DIR:-/D:/Document/StudyAndWorkFile/workspace/test_dataset/BP}
NUM_EPOCHS=${NUM_EPOCHS:-5000}
TARGET_CURVES=${TARGET_CURVES:-2048}
LOG_INTERVAL=${LOG_INTERVAL:-100}
SEED=${SEED:-42}
L1_W=${L1_W:-1.0}; L2_W=${L2_W:-1.0}; SSIM_W=${SSIM_W:-1.0}; WIDTH_W=${WIDTH_W:-10.0}
SNAKE_W=${SNAKE_W:-0}          # σ 消融默认关 GVF 以隔离
GVF_GUIDED=${GVF_GUIDED:-0}    # 同上
# σ 组合（name start end, σ_px=σ·W）：两轴设计，名字全用值式(0p01)。
#   start 轴（论点1: 起始不影响终值 PSNR，只影响收敛/耗时）：固定 end=0.0001，扫 start
#         σ_px：0.1→~58px  0.01→~6px  0.005→~3px  0.001→~0.6px
#   end 轴（论点2: 终值要够小）：固定退火幅度 ratio=start/end=100（跨2个数量级），扫绝对 end
#         0.1→0.001, 0.05→0.0005, 0.01→0.0001 —— 只变绝对终值，退火形状一致
#   注：end=0.0001 的 ratio-100 点 = 0.01→0.0001 = start_0p01（已含），不重复跑。
COMBOS=(
  "start_0p1   0.1    0.0001"
  "start_0p01  0.01   0.0001"   # 同时是 end 轴 ratio-100 的 end=0.0001 点
  "start_0p005 0.005  0.0001"   # = default (0.005→0.0001)
  "start_0p001 0.001  0.0001"
  "end_0p001   0.1    0.001"    # ratio 100
  "end_0p0005  0.05   0.0005"   # ratio 100
)
# -----------------

COMMON_ARGS="--data_dir $DATA_DIR --num_epochs $NUM_EPOCHS --target_curves $TARGET_CURVES \
--log_interval $LOG_INTERVAL --seed $SEED \
--l1_weight $L1_W --l2_weight $L2_W --ssim_weight $SSIM_W --lambda_w $WIDTH_W \
--snake_w $SNAKE_W --gvf_guided_densify $GVF_GUIDED"

echo "=============================================================="
echo " σ_edge 退火消融 | data_dir=$DATA_DIR | epochs=$NUM_EPOCHS | seed=$SEED"
echo " GVF 关闭(隔离 σ) | 共 ${#COMBOS[@]} 组"
echo "=============================================================="

i=0
for combo in "${COMBOS[@]}"; do
  i=$((i+1))
  set -- $combo; NAME=$1; S0=$2; S1=$3
  echo -e "\n>>> [$i/${#COMBOS[@]}] $NAME  (sigma $S0 → $S1)"
  python scripts/main.py --exp_name "sigma_$NAME" $COMMON_ARGS --sigma_start $S0 --sigma_end $S1
done

echo -e "\n=============================================================="
echo " ✅ σ 消融跑完。聚合："
echo "    python scripts/aggregate_ablation.py \\"
for combo in "${COMBOS[@]}"; do set -- $combo; echo "      --cond $1=output/sigma_$1 \\"; done
echo "      --ref s_default"
echo "=============================================================="
