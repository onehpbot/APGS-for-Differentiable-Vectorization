import torch
import torch.nn as nn
try:
    from gsplat.project_gaussians_2d import project_strokes_2d
    from gsplat.rasterize_strokes import rasterize_strokes
except ImportError:
    print("❌ 无法导入 gsplat！")
    raise

class CUDABezierRenderer(nn.Module):
    def __init__(self, H, W, device, curve_beta=1.0, fine_M=200):
        super().__init__()
        self.H, self.W = H, W
        self.device = device
        self.curve_beta = curve_beta
        self.fine_M = fine_M
        self.steps = 32
        self.res_tensor = torch.tensor([W, H], dtype=torch.float32, device=device)

    def _adaptive_t(self, cp_batch):
        # (保留原本的自适应 t 计算逻辑)
        N = cp_batch.shape[0]
        M = self.fine_M
        beta = self.curve_beta
        with torch.no_grad():
            t_fine = torch.linspace(0.0, 1.0, M, device=self.device)
            tt = t_fine.view(1, -1, 1)
            p0 = cp_batch[:, 0:1, :]; p1 = cp_batch[:, 1:2, :]
            p2 = cp_batch[:, 2:3, :]; p3 = cp_batch[:, 3:4, :]
            pts = ((1-tt)**3*p0 + 3*(1-tt)**2*tt*p1 + 3*(1-tt)*tt**2*p2 + tt**3*p3)
            seg = pts[:, 1:, :] - pts[:, :-1, :]
            seg_arc = torch.sqrt(torch.sum(seg**2, dim=-1) + 1e-8)
            a, b = seg[:, :-1, :], seg[:, 1:, :]
            dot = (a * b).sum(-1)
            cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
            turn = torch.atan2(cross.abs(), dot)
            turn_v = torch.cat([turn[:, :1], turn, turn[:, -1:]], dim=1)
            turn_seg = 0.5 * (turn_v[:, :-1] + turn_v[:, 1:])
            w = seg_arc * (1.0 + beta * turn_seg)
            cumw = torch.cumsum(w, dim=1)
            total = cumw[:, -1:]
            cost = torch.cat([torch.zeros(N, 1, device=self.device), cumw], dim=1)
            cost = cost / (total + 1e-8)
            q = torch.linspace(0.0, 1.0, self.steps + 1, device=self.device)
            qe = q.view(1, -1).expand(N, -1).contiguous()
            idx = torch.searchsorted(cost, qe, right=True).clamp(1, M - 1)
            i0 = idx - 1
            c0 = cost.gather(1, i0); c1 = cost.gather(1, idx)
            t0 = t_fine[i0];              t1 = t_fine[idx]
            frac = (qe - c0) / (c1 - c0 + 1e-8)
            t_new = (t0 + frac * (t1 - t0)).clamp(0.0, 1.0)
            t_new[:, 0] = 0.0
            t_new[:, -1] = 1.0
        return t_new.view(N, self.steps + 1, 1)

    def forward(self, cp_batch, thickness_batch, color_batch, opacity_batch, tau, sigma_edge):
        # (保留原本前向传播的所有代码，包含对 project_strokes_2d 和 rasterize_strokes 的调用)
        num_curves = cp_batch.shape[0]
        t = self._adaptive_t(cp_batch)
        p0 = cp_batch[:, 0:1, :]
        p1 = cp_batch[:, 1:2, :]
        p2 = cp_batch[:, 2:3, :]
        p3 = cp_batch[:, 3:4, :]

        pts = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
        pts_left = pts[:, :-1, :]
        pts_right = pts[:, 1:, :]
        mus = (pts_left + pts_right) / 2.0

        pts_px = pts * self.res_tensor
        pts_left_px = pts_px[:, :-1, :]
        pts_right_px = pts_px[:, 1:, :]
        AB_px = pts_right_px - pts_left_px
        L_px = torch.sqrt(torch.sum(AB_px**2, dim=-1, keepdim=True) + 1e-8)
        tangents = AB_px / L_px

        T_left = torch.cat([tangents[:, 0:1, :], tangents[:, :-1, :]], dim=1)
        T_right = torch.cat([tangents[:, 1:, :], tangents[:, -1:, :]], dim=1)
        cos_prev = torch.sum(tangents * T_left, dim=-1, keepdim=True)
        cos_next = torch.sum(tangents * T_right, dim=-1, keepdim=True)
        cos_theta = torch.clamp(torch.min(cos_prev, cos_next), -1.0, 1.0)

        curvature_penalty = 1.0 - cos_theta
        overlap_ratio = 0.10 + 1.0 * curvature_penalty
        sigma_x_px = (L_px / 2.0) * (1.0 + overlap_ratio)

        N_rect = num_curves * self.steps
        means2d_rect = (mus.reshape(N_rect, 2) * 2.0 - 1.0)
        tangents_rect = tangents.reshape(N_rect, 2)
        sigma_x_rect = sigma_x_px.reshape(N_rect, 1)
        thickness_px = thickness_batch * self.W
        thicknesses_rect = thickness_px.view(-1, 1, 1).expand(-1, self.steps, 1).reshape(N_rect, 1)

        colors_rect = color_batch.view(-1, 1, 3).expand(-1, self.steps, 3).reshape(N_rect, 3)
        opacity_rect = opacity_batch.view(-1, 1, 1).expand(-1, self.steps, 1).reshape(N_rect, 1)
        types_rect = torch.zeros((N_rect, 1), dtype=torch.int32, device=self.device)
        z_order_ids = torch.arange(num_curves - 1, -1, -1, dtype=torch.int32, device=self.device)
        curve_ids_rect = z_order_ids.view(-1, 1, 1).expand(-1, self.steps, 1).reshape(N_rect, 1)

        t_caps = torch.tensor([0.0, 1.0], device=self.device).view(1, -1, 1)
        caps_mu = (1-t_caps)**3 * p0 + 3*(1-t_caps)**2 * t_caps * p1 + 3*(1-t_caps)*t_caps**2 * p2 + t_caps**3 * p3
        dp_caps = 3*(1-t_caps)**2 * (p1 - p0) + 6*(1-t_caps)*t_caps * (p2 - p1) + 3*t_caps**2 * (p3 - p2)
        dp_caps_px = dp_caps * self.res_tensor
        dp_caps_norm = torch.sqrt(torch.sum(dp_caps_px**2, dim=-1, keepdim=True) + 1e-8)
        caps_tangents = dp_caps_px / dp_caps_norm

        N_caps = num_curves * 2
        means2d_caps = (caps_mu.reshape(N_caps, 2) * 2.0 - 1.0)
        tangents_caps = caps_tangents.reshape(N_caps, 2)
        sigma_x_caps = torch.zeros((N_caps, 1), device=self.device)
        thicknesses_caps = thickness_px.view(-1, 1, 1).expand(-1, 2, 1).reshape(N_caps, 1)
        colors_caps = color_batch.view(-1, 1, 3).expand(-1, 2, 3).reshape(N_caps, 3)
        opacity_caps = opacity_batch.view(-1, 1, 1).expand(-1, 2, 1).reshape(N_caps, 1)
        types_caps = torch.ones((N_caps, 1), dtype=torch.int32, device=self.device)
        curve_ids_caps = z_order_ids.view(-1, 1, 1).expand(-1, 2, 1).reshape(N_caps, 1)

        means2d = torch.cat([means2d_rect, means2d_caps], dim=0)
        tangents_all = torch.cat([tangents_rect, tangents_caps], dim=0)
        sigma_x_all = torch.cat([sigma_x_rect, sigma_x_caps], dim=0)
        thicknesses_all = torch.cat([thicknesses_rect, thicknesses_caps], dim=0)
        colors_all = torch.cat([colors_rect, colors_caps], dim=0)
        opacities_all = torch.cat([opacity_rect, opacity_caps], dim=0).view(-1)
        primitive_types = torch.cat([types_rect, types_caps], dim=0)
        curve_ids = torch.cat([curve_ids_rect, curve_ids_caps], dim=0)

        mu_tensor = mus.reshape(N_rect, 2)
        sigma_edge_px = sigma_edge * self.W

        BLOCK_X, BLOCK_Y = 16, 16
        tile_bounds = ((self.W + BLOCK_X - 1) // BLOCK_X, (self.H + BLOCK_Y - 1) // BLOCK_Y, 1)

        xys, depths, radii, num_tiles_hit = project_strokes_2d(
            means2d, tangents_all, sigma_x_all, thicknesses_all,
            primitive_types, curve_ids, sigma_edge_px, self.H, self.W, tile_bounds)

        background = torch.ones(3, dtype=torch.float32, device=self.device)
        final_rgb = rasterize_strokes(
            xys, depths, radii, num_tiles_hit,
            means2d, tangents_all, sigma_x_all, thicknesses_all, colors_all,
            primitive_types, curve_ids, sigma_edge_px,
            self.H, self.W, BLOCK_H=BLOCK_Y, BLOCK_W=BLOCK_X,
            background=background, opacities=opacities_all)

        return final_rgb, mu_tensor