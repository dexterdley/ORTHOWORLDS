"""
Shared model pieces. For now: Wan 2.1 VAE helpers (moved unchanged from the training scripts).
"""
import torch
import torch.nn as nn


def _get_wan_latent_stats(vae, target_device, target_dtype):
    """Retrieves per-channel mean and std from AutoencoderKLWan config."""
    if hasattr(vae.config, "latents_mean") and vae.config.latents_mean is not None:
        mean = torch.tensor(vae.config.latents_mean, device=target_device, dtype=target_dtype).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=target_device, dtype=target_dtype).view(1, -1, 1, 1, 1)
    return mean, std

def encode_views_to_latents(vae, views_tensor, out_dtype=None):
    """
    Encodes 4 orthographic views into normalized Wan 3D VAE latents ~ N(0, I).
    views_tensor: (B, 4, 3, H, W) in [-1, 1] -> Latents: (B, 16, 4, h, w)
    """
    b, num_views, c, h, w = views_tensor.shape
    flat_views = views_tensor.view(b * num_views, c, 1, h, w)

    with torch.no_grad():
        latent_dist = vae.encode(flat_views.to(dtype=vae.dtype)).latent_dist
        latents = latent_dist.mode() if hasattr(latent_dist, "mode") else latent_dist.sample()
        mean, std = _get_wan_latent_stats(vae, latents.device, latents.dtype)
        latents = (latents - mean) / std

    _, c_lat, _, h_lat, w_lat = latents.shape
    latents = latents.view(b, num_views, c_lat, h_lat, w_lat).permute(0, 2, 1, 3, 4)
    if out_dtype is not None:
        latents = latents.to(dtype=out_dtype)
    return latents

def decode_latents_to_views(vae, latents):
    """
    Unnormalizes and decodes 4 orthographic views latents back into image tensors.
    latents: (B, C_lat, 4, h_lat, w_lat) -> (B, 4, 3, H, W) in [0, 1]
    """
    b, c_lat, num_views, h_lat, w_lat = latents.shape
    flat_latents = latents.permute(0, 2, 1, 3, 4).reshape(b * num_views, c_lat, 1, h_lat, w_lat)
    flat_latents = flat_latents.to(dtype=vae.dtype)

    mean, std = _get_wan_latent_stats(vae, flat_latents.device, flat_latents.dtype)
    flat_latents = flat_latents * std + mean

    with torch.no_grad():
        decoded = vae.decode(flat_latents).sample  # (B * num_views, 3, 1, H, W)

    decoded = decoded.squeeze(2)  # (B * num_views, 3, H, W)
    views = decoded.view(b, num_views, 3, decoded.shape[-2], decoded.shape[-1])
    views = torch.clamp((views + 1.0) / 2.0, 0.0, 1.0)
    return views

class LinearProjector(nn.Module):
    def __init__(self, h_lat, w_lat, c_lat=16, n_src=3):
        super().__init__()
        self.h_lat, self.w_lat = h_lat, w_lat
        hw = h_lat * w_lat
        n_in = n_src * 3 # each view x [1, sin(yaw), cos(yaw)]
        self.spatial = nn.Parameter(torch.eye(hw).repeat(n_in, 1, 1))
        self.channel = nn.Linear(n_in * c_lat, c_lat)
        self.bias = nn.Parameter(torch.zeros(c_lat, hw))
        with torch.no_grad(): # init = copy plain Rear (unchanged behaviour)
            self.channel.weight.zero_(); self.channel.weight[:, (n_src - 1) * c_lat : n_src * c_lat] = torch.eye(c_lat)
            self.channel.bias.zero_()

    def forward(self, latents_ortho, cam):
        x = latents_ortho.float()
        s, c_ = cam[:, 0].view(-1, 1, 1, 1, 1), cam[:, 1].view(-1, 1, 1, 1, 1)
        latents_ortho = torch.cat([x, x * s, x * c_], dim=2)

        b, c, v = latents_ortho.shape[:3]
        x = torch.einsum("bcvp,vqp->bvcq", latents_ortho.float().reshape(b, c, v, -1), self.spatial)  # (B,9,C,hw).
        x = self.channel(x.reshape(b, v * c, -1).transpose(1, 2)).transpose(1, 2) + self.bias          # (B,C,hw)
        return x.view(b, c, 1, self.h_lat, self.w_lat)

class SoftZProjector(nn.Module):
    """FPV sketch from Top/Side/Rear by a soft z-buffer: per FPV cell, softmax over source cells of
    (learned ray logit(yaw) + occupancy score of the cell). Same interface as LinearProjector."""
    def __init__(self, h_lat, w_lat, c_lat=16, n_src=3, init_peak=8.0, hidden=32):
        super().__init__()
        self.h_lat, self.w_lat = h_lat, w_lat
        hw = h_lat * w_lat
        P0 = torch.zeros(hw, n_src, hw)
        P0[:, n_src - 1] = torch.eye(hw) * init_peak                        # init ~ copy Rear
        P0 = P0.view(hw, n_src * hw)
        self.pos = nn.Parameter(torch.stack([P0] + [torch.zeros_like(P0)] * 4))  # [1, sin/cos yaw, sin/cos pitch] x (hw, 3hw)
        self.null_logit = nn.Parameter(torch.zeros(hw))                      # "ray hits nothing"
        self.bg = nn.Parameter(torch.zeros(hw, c_lat))                       # background for empty rays
        self.score = nn.ModuleList([nn.Sequential(nn.Linear(c_lat, hidden), nn.GELU(), nn.Linear(hidden, 1)) for _ in range(n_src)])
        self.value = nn.ModuleList([nn.Linear(c_lat, c_lat) for _ in range(n_src)])
        with torch.no_grad():
            for s in self.score: s[-1].weight.zero_(); s[-1].bias.zero_()    # start purely positional
            for v in self.value: v.weight.copy_(torch.eye(c_lat)); v.bias.zero_()

    def forward(self, latents_ortho, cam):
        x = latents_ortho.float()                                            # (B, C, 3, h, w)
        b, c, v = x.shape[:3]
        tok = x.reshape(b, c, v, -1).permute(0, 2, 3, 1)                     # (B, 3, hw, C)
        score = torch.cat([self.score[i](tok[:, i]) for i in range(v)], 1).squeeze(-1)   # (B, 3hw)  occupancy
        val = torch.cat([self.value[i](tok[:, i]) for i in range(v)], 1)                 # (B, 3hw, C)
        basis = torch.cat([torch.ones_like(cam[:, :1]), cam.float()], 1)     # (B, 3)
        logits = torch.einsum("bk,kqn->bqn", basis, self.pos) + score[:, None, :]       # (B, hw, 3hw)
        null = self.null_logit.view(1, -1, 1).expand(b, -1, 1)
        attn = torch.softmax(torch.cat([logits, null], -1), -1)              # (B, hw, 3hw + 1)
        out = attn[..., :-1] @ val + attn[..., -1:] * self.bg                # (B, hw, C)
        return out.transpose(1, 2).reshape(b, c, 1, self.h_lat, self.w_lat)

def cam_feats(cam_states, device="cuda" if torch.cuda.is_available() else "cpu"):
    yaw = torch.tensor([float(cs.get("yaw", 0.0)) for cs in cam_states], device=device)
    pitch = torch.tensor([float(cs.get("pitch", 0.0)) for cs in cam_states], device=device)
    return torch.stack([yaw.sin(), yaw.cos(), pitch.sin(), pitch.cos()], dim=1)

class CrossViewProjector(nn.Module):
    """Renders view `tgt` from the other three by a soft z-buffer (leave-one-out): per target cell, softmax
    over source cells of (learned ray logit(yaw, pitch) + occupancy score of the cell).
    The per-view encoders (score, value) are shared across targets; ray logits, null logit and background
    are per target. A 3-view input (Top, Side, Rear) renders the FPV, as SoftZProjector did.
    Views: 0 Top, 1 Side, 2 Rear, 3 FPV."""
    def __init__(self, h_lat, w_lat, c_lat=16, n_views=4, init_peak=8.0, hidden=32, init_src=(1, 2, 1, 2)):
        super().__init__()
        self.h_lat, self.w_lat = h_lat, w_lat
        hw = h_lat * w_lat
        self.src = [[j for j in range(n_views) if j != i] for i in range(n_views)]   # sources of each target
        P0 = torch.zeros(n_views, hw, n_views - 1, hw)
        for i, j in enumerate(init_src):                                    # init ~ copy init_src[i] (FPV <- Rear, as before)
            P0[i, :, self.src[i].index(j)] = torch.eye(hw) * init_peak
        P0 = P0.view(n_views, hw, (n_views - 1) * hw)
        self.pos = nn.Parameter(torch.stack([P0] + [torch.zeros_like(P0)] * 4, 1))  # (tgt, [1, sin/cos yaw, sin/cos pitch], hw, 3hw)
        self.null_logit = nn.Parameter(torch.zeros(n_views, hw))            # "ray hits nothing", per target
        self.bg = nn.Parameter(torch.zeros(n_views, hw, c_lat))             # background for empty rays, per target
        self.score = nn.ModuleList([nn.Sequential(nn.Linear(c_lat, hidden), nn.GELU(), nn.Linear(hidden, 1)) for _ in range(n_views)])  # shared by all targets
        self.value = nn.ModuleList([nn.Linear(c_lat, c_lat) for _ in range(n_views)])
        with torch.no_grad():
            for s in self.score: s[-1].weight.zero_(); s[-1].bias.zero_()    # start purely positional
            for v in self.value: v.weight.copy_(torch.eye(c_lat)); v.bias.zero_()

    def forward(self, views, cam, tgt=3):
        x = views.float()                                                    # (B, C, 4, h, w), or (B, C, 3, h, w) for FPV
        if x.shape[2] == 3:                                                  # FPV slot is never a source for tgt=3
            x = torch.cat([x, torch.zeros_like(x[:, :, :1])], 2)
        b, c, v = x.shape[:3]
        src = self.src[tgt]
        tok = x.reshape(b, c, v, -1).permute(0, 2, 3, 1)                     # (B, 4, hw, C)
        score = torch.cat([self.score[i](tok[:, i]) for i in src], 1).squeeze(-1)   # (B, 3hw)  occupancy
        val = torch.cat([self.value[i](tok[:, i]) for i in src], 1)                 # (B, 3hw, C)
        basis = torch.cat([torch.ones_like(cam[:, :1]), cam.float()], 1)     # (B, 5)
        logits = torch.einsum("bk,kqn->bqn", basis, self.pos[tgt]) + score[:, None, :]   # (B, hw, 3hw)
        null = self.null_logit[tgt].view(1, -1, 1).expand(b, -1, 1)
        attn = torch.softmax(torch.cat([logits, null], -1), -1)              # (B, hw, 3hw + 1)
        out = attn[..., :-1] @ val + attn[..., -1:] * self.bg[tgt]           # (B, hw, C)
        return out.transpose(1, 2).reshape(b, c, 1, self.h_lat, self.w_lat)