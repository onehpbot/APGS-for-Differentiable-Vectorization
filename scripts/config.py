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
    STOP_BEFORE = 5000
    SIGMA_START = 0.002
    SIGMA_END = 0.0001
    TAU_OP_START = 0.02
    TAU_OP_END = 0.002