import torch

class Config:
    # 基础与设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # IO 与 数据集配置
    DATA_DIR = rf"/data/workspace_PH/dataset/AP"  
    OUT_DIR = rf"output/"                
    MAX_RES = 800
    # 🌟 图像下采样配置 (例如 0.5 表示宽高减半；1.0 表示使用原图)
    DOWNSAMPLE_SCALE = 0.5             
    
    # 曲线容量与分层释放配置
    TARGET_CURVES = 2048
    INITIAL_CURVES = 2048
    RELEASE_INTERVAL = 99999999
    CURVES_PER_RELEASE = 256
    
    # 优化器配置
    NUM_EPOCHS = 5000
    LR = 0.01
    LR_MIN = 1e-5
    # GVF/Snake 权重线性预热：在前 WARMUP_FRAC·NUM_EPOCHS 步内从 0 线性增到 SNAKE_W。
    # 设 0.0 即禁用预热（snake_w 全程恒定）。注意不要设太大（如旧的 0.9）——
    # cosine LR 在后期已衰减到 ~1e-5，warmup 拖到后期会让 GVF 同时"满权重×近零学习率"，等于失效。
    WARMUP_FRAC = 0.1
    
    # 损失函数权重
    W_MAX = 0.01
    LAMBDA_W = 10.0
    LAMBDA_OPL1 = 0.005
    SNAKE_W = 100
    L1_WEIGHT = 1.0
    L2_WEIGHT = 1.0
    SSIM_WEIGHT = 1.0
    
    # 动态拓扑控制 (Prune & Densify)
    DENSIFY = True
    DENSIFY_INTERVAL = 500
    # densify 在 epoch < NUM_EPOCHS - STOP_BEFORE 期间触发。原值 5000==NUM_EPOCHS → 条件恒假，
    # densify 从不运行（所有消融其实跑的是静态曲线）。改为 500，让 densify 在 500..4500 期间生效，
    # 给 GVF 一个结构通道（新曲线可沿边缘切向定向、落点）。
    STOP_BEFORE = 500
    SIGMA_START = 0.002
    SIGMA_END = 0.0001
    TAU_OP_START = 0.02
    TAU_OP_END = 0.002