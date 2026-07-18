import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np
from scipy import ndimage

class MultiCurveFitter(nn.Module):
    def __init__(self, num_curves=10, w_max=0.01):
        super().__init__()
        self.num_curves = num_curves
        self.w_max = w_max
        
        _start = torch.rand(num_curves, 2) * 0.85 + 0.05
        _theta = torch.full((num_curves,), math.pi / 4.0) 
        _length = torch.full((num_curves,), 0.03)
        _dir = torch.stack([torch.cos(_theta), torch.sin(_theta)], dim=1)
        _end = _start + _length.unsqueeze(1) * _dir
        _p0 = _start
        _p3 = _end
        _p1 = _start + (_end - _start) * (1.0 / 3.0)
        _p2 = _start + (_end - _start) * (2.0 / 3.0)
        
        self.cp = nn.Parameter(torch.stack([_p0, _p1, _p2, _p3], dim=1))
        self.raw_thickness = nn.Parameter(torch.ones(num_curves) * math.log(math.exp(0.005) - 1))
        self.raw_color = nn.Parameter(torch.randn(num_curves, 3) * 2.0)
        self.raw_opacity = nn.Parameter(torch.full((num_curves,), math.log(0.9 / 0.1)))

    def get_params(self):
        return self.cp, F.softplus(self.raw_thickness), torch.sigmoid(self.raw_color), torch.sigmoid(self.raw_opacity)

    def width_reg(self):
        return (F.relu(F.softplus(self.raw_thickness) - self.w_max) ** 2).mean()

def prune_by_opacity_threshold(fitter, tau_opacity):
    with torch.no_grad():
        opacity = torch.sigmoid(fitter.raw_opacity.detach())
        N = opacity.shape[0]
        keep = opacity >= tau_opacity
        if keep.sum() < 8:
            _, idx = torch.topk(opacity, 8, largest=True)
            keep = torch.zeros(N, dtype=torch.bool, device=opacity.device)
            keep[idx] = True
        return keep, N - keep.sum().item()

def sample_new_curves(err, target, n_new, device, thin_min=0.002, thin_max=0.008):
    H, W = err.shape
    err_np = err.detach().cpu().numpy()
    tgt_np = target.detach().cpu().numpy()
    thr = max(float(np.quantile(err_np, 0.90)), float(err_np.mean()) * 1.5)
    mask = err_np > thr
    labeled, n_cc = ndimage.label(mask)
    if n_cc > 0:
        sizes = ndimage.sum(err_np, labeled, range(1, n_cc + 1))
        order = np.argsort(sizes)[::-1]
        
    cp_list, thick_list, col_list = [], [], []
    for i in range(n_new):
        if n_cc > 0:
            rid = order[i % n_cc] + 1
            ys, xs = np.where(labeled == rid)
            cy, cx = float(ys.mean()), float(xs.mean())
        else:
            cy, cx = float(np.random.rand() * H), float(np.random.rand() * W)
            
        r = 6
        y0, y1 = max(0, int(cy - r)), min(H, int(cy + r))
        x0, x1 = max(0, int(cx - r)), min(W, int(cx + r))
        nx, ny = cx / W, cy / H
        
        theta = math.pi / 4.0
        length = 0.03
        dx, dy = math.cos(theta), math.sin(theta)
        sx, sy = nx, ny
        ex, ey = sx + length * dx, sy + length * dy
        
        cp_list.append([[sx, sy], [sx + (ex-sx)/3, sy + (ey-sy)/3], [sx + 2*(ex-sx)/3, sy + 2*(ey-sy)/3], [ex, ey]])
        thick_list.append(float(np.random.uniform(thin_min, thin_max)))
        col_list.append(tgt_np[y0:y1, x0:x1].reshape(-1, 3).mean(0))
        
    new_cp = torch.from_numpy(np.array(cp_list, dtype=np.float32)).to(device)
    new_raw_thick = torch.log(torch.expm1(torch.tensor(thick_list, dtype=torch.float32, device=device).clamp(min=1e-4)))
    new_raw_col = torch.log(torch.from_numpy(np.stack(col_list).astype(np.float32)).to(device).clamp(0.02, 0.98) / (1 - torch.from_numpy(np.stack(col_list).astype(np.float32)).to(device).clamp(0.02, 0.98)))
    new_raw_op = torch.full((n_new,), math.log(0.7 / 0.3), device=device)
    return new_cp, new_raw_thick, new_raw_col, new_raw_op

def apply_resize(fitter, keep_mask, new_cp, new_raw_thick, new_raw_col, new_raw_op):
    with torch.no_grad():
        cp, rt, rc, ro = fitter.cp.data[keep_mask], fitter.raw_thickness.data[keep_mask], fitter.raw_color.data[keep_mask], fitter.raw_opacity.data[keep_mask]
        fitter.cp = nn.Parameter(torch.cat([cp, new_cp], 0))
        fitter.raw_thickness = nn.Parameter(torch.cat([rt, new_raw_thick], 0))
        fitter.raw_color = nn.Parameter(torch.cat([rc, new_raw_col], 0))
        fitter.raw_opacity = nn.Parameter(torch.cat([ro, new_raw_op], 0))
        fitter.num_curves = fitter.cp.shape[0]

def make_optimizer(fitter, lr):
    return optim.Adam([
        {'params': fitter.cp, 'lr': lr}, {'params': fitter.raw_thickness, 'lr': lr},
        {'params': fitter.raw_color, 'lr': lr}, {'params': fitter.raw_opacity, 'lr': lr}
    ])