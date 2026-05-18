from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

def uv_to_lonlat_torch(u: torch.Tensor, v: torch.Tensor, w: int, h: int):
    pi = torch.pi
    lon = (u / w) * (2.0 * pi) - pi
    lat = (0.5 - v / h) * pi
    return lon, lat


def lonlat_to_uv_torch(lon: torch.Tensor, lat: torch.Tensor, w: int, h: int):
    pi = torch.pi
    u = (lon + pi) / (2.0 * pi) * w
    v = (0.5 - lat / pi) * h
    return u, v


def lonlat_to_dir_torch(lon: torch.Tensor, lat: torch.Tensor):
    clat = torch.cos(lat)
    x = clat * torch.cos(lon)
    y = clat * torch.sin(lon)
    z = torch.sin(lat)
    return x, y, z


def dir_to_lonlat_torch(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    lon = torch.atan2(y, x)
    lat = torch.atan2(z, torch.sqrt(x * x + y * y))
    return lon, lat

def stereographic_from_sphere_torch(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, eps: float = 1e-6):
    denom = 1.0 - z
    safe = torch.abs(denom) >= eps
    denom_safe = torch.where(safe, denom, torch.ones_like(denom))
    w_real = x / denom_safe
    w_imag = y / denom_safe
    return torch.complex(w_real, w_imag), safe


def stereographic_to_sphere_torch(w: torch.Tensor):
    wr = torch.real(w)
    wi = torch.imag(w)
    r2 = wr * wr + wi * wi
    denom = r2 + 1.0
    x = 2.0 * wr / denom
    y = 2.0 * wi / denom
    z = (r2 - 1.0) / denom
    return x, y, z

def _to_complex(val: torch.Tensor) -> torch.Tensor:
    if torch.is_complex(val):
        return val
    return torch.complex(val, torch.zeros_like(val))


def _clamp_complex(z: torch.Tensor, max_gain: float) -> torch.Tensor:
    mag = torch.abs(z)
    scale = torch.clamp(mag, max=max_gain) / (mag + 1e-8)
    return z * scale


def mobius_params_torch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor, eps: float = 1e-3, max_gain: float = 4.0):
    is_complex = any(torch.is_complex(t) for t in (a, b, c, d))
    if not is_complex:
        det = a * d - b * c
        safe_det = torch.sign(det) * torch.clamp(det.abs(), min=eps)
        scale = torch.sqrt(safe_det.abs())
        a = a / scale
        b = b / scale
        c = c / scale
        d = d / scale
        a = torch.clamp(a, -max_gain, max_gain)
        b = torch.clamp(b, -max_gain, max_gain)
        c = torch.clamp(c, -max_gain, max_gain)
        d = torch.clamp(d, -max_gain, max_gain)
        return a, b, c, d

    a = _to_complex(a)
    b = _to_complex(b)
    c = _to_complex(c)
    d = _to_complex(d)
    det = a * d - b * c
    det_mag = torch.abs(det)
    det_phase = det / torch.where(det_mag > 0, det_mag, torch.ones_like(det_mag))
    safe_det_mag = torch.clamp(det_mag, min=eps)
    safe_det = det_phase * safe_det_mag
    scale = torch.sqrt(torch.abs(safe_det))
    a = a / scale
    b = b / scale
    c = c / scale
    d = d / scale
    a = _clamp_complex(a, max_gain)
    b = _clamp_complex(b, max_gain)
    c = _clamp_complex(c, max_gain)
    d = _clamp_complex(d, max_gain)
    return a, b, c, d


def invert_mobius_params_torch(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor):
    ai, bi, ci, di = d, -b, -c, a
    return mobius_params_torch(ai, bi, ci, di)

def clamp_theta(theta: torch.Tensor, bounds: Dict[str, Tuple[float, float]], dim_names: list[str]) -> torch.Tensor:
    theta = theta.clone()
    for i, name in enumerate(dim_names):
        if name in bounds:
            lo, hi = bounds[name]
            theta[..., i] = torch.clamp(theta[..., i], lo, hi)
    return theta


def _ensure_theta_batch(theta: torch.Tensor, batch: int) -> torch.Tensor:
    if not isinstance(theta, torch.Tensor):
        theta = torch.tensor(theta, dtype=torch.float32)
    if theta.dim() == 1:
        theta = theta.unsqueeze(0)
    if theta.shape[0] == 1 and batch > 1:
        theta = theta.expand(batch, -1)
    if theta.shape[0] != batch:
        raise ValueError(f"Theta batch {theta.shape[0]} must match frames batch {batch}")
    return theta


def _theta_to_abcd(theta: torch.Tensor, dim_names: Optional[list[str]]):
    if dim_names is None:
        dim_names = ["cx", "cy", "log_scale", "rot"]

    name_set = set(dim_names)
    if {"a", "b", "c", "d"}.issubset(name_set):
        a = theta[:, dim_names.index("a")]
        b = theta[:, dim_names.index("b")]
        c = theta[:, dim_names.index("c")]
        d = theta[:, dim_names.index("d")]
        log_aniso = None
        if "log_aniso" in name_set:
            log_aniso = theta[:, dim_names.index("log_aniso")]
        return _to_complex(a), _to_complex(b), _to_complex(c), _to_complex(d), log_aniso

    cx = theta[:, dim_names.index("cx")]
    cy = theta[:, dim_names.index("cy")]
    log_scale = theta[:, dim_names.index("log_scale")]
    rot = theta[:, dim_names.index("rot")]
    log_aniso = None
    if "log_aniso" in name_set:
        log_aniso = theta[:, dim_names.index("log_aniso")]

    c = torch.complex(cx, cy)
    scale = torch.exp(log_scale)
    rot_c = torch.complex(torch.cos(rot), torch.sin(rot))
    r = scale * rot_c
    a = r
    b = -r * c
    c_param = -torch.conj(c)
    d = torch.ones_like(a)
    return a, b, c_param, d, log_aniso

def _make_base_grid(h: int, w: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    v_coords = torch.linspace(0, h - 1, h, device=device, dtype=dtype)
    u_coords = torch.linspace(0, w - 1, w, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(v_coords, u_coords, indexing="ij")
    lon_out, lat_out = uv_to_lonlat_torch(uu, vv, w, h)
    x_out, y_out, z_out = lonlat_to_dir_torch(lon_out, lat_out)
    w_out, safe_stereo = stereographic_from_sphere_torch(x_out, y_out, z_out)
    return {
        "w_out": w_out,
        "safe_stereo": safe_stereo,
        "h": h,
        "w": w,
    }


def _warp_erp(
    frame: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    inverse: bool,
    base_grid: Optional[Dict[str, torch.Tensor]],
    log_aniso: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if frame.dim() == 3:
        frame = frame.unsqueeze(0)
        squeeze_back = True
    else:
        squeeze_back = False
    bsz, _, h, w = frame.shape

    if base_grid is None or base_grid.get("h", -1) != h or base_grid.get("w", -1) != w:
        base_grid = _make_base_grid(h, w, frame.device, frame.dtype)

    w_out = base_grid["w_out"]
    safe_stereo = base_grid["safe_stereo"]

    a, b, c, d = mobius_params_torch(a, b, c, d)

    w_out = w_out.unsqueeze(0)
    safe_stereo = safe_stereo.unsqueeze(0)
    a = a.view(-1, 1, 1)
    b = b.view(-1, 1, 1)
    c = c.view(-1, 1, 1)
    d = d.view(-1, 1, 1)

    if inverse:
        a, b, c, d = invert_mobius_params_torch(a, b, c, d)

    denom = (-c * w_out + a)
    num = (d * w_out - b)
    eps_div = 1e-6
    safe_mob = torch.abs(denom) >= eps_div
    denom_safe = torch.where(
        safe_mob,
        denom,
        torch.complex(torch.full_like(denom.real, eps_div), torch.zeros_like(denom.imag)),
    )
    z_src = num / denom_safe

    if log_aniso is not None:
        an = torch.exp(log_aniso).view(-1, 1, 1)
        z_src = torch.complex(z_src.real * an, z_src.imag / an)

    x_src, y_src, z_src_sph = stereographic_to_sphere_torch(z_src)

    nan_mask = (~safe_stereo) | (~safe_mob)

    lon_src, lat_src = dir_to_lonlat_torch(x_src, y_src, z_src_sph)
    lon_wrapped = torch.atan2(torch.sin(lon_src), torch.cos(lon_src))
    u_src, v_src = lonlat_to_uv_torch(lon_wrapped, lat_src, w, h)
    v_src = torch.clamp(v_src, 0.0, h - 1 - 1e-6)

    x_norm = (u_src + 0.5) / w * 2.0 - 1.0
    y_norm = (v_src + 0.5) / h * 2.0 - 1.0

    x_norm = torch.where(nan_mask, torch.full_like(x_norm, 2.0), x_norm)
    y_norm = torch.where(nan_mask, torch.full_like(y_norm, 2.0), y_norm)
    x_norm = torch.nan_to_num(x_norm, nan=2.0, posinf=2.0, neginf=2.0)
    y_norm = torch.nan_to_num(y_norm, nan=2.0, posinf=2.0, neginf=2.0)

    grid = torch.stack([x_norm, y_norm], dim=-1)
    out = F.grid_sample(frame, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.squeeze(0) if squeeze_back else out


def apply_mobius(
    frame: torch.Tensor,
    theta: torch.Tensor,
    dim_names: Optional[list[str]] = None,
    base_grid: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    if frame.dim() == 3:
        batch = 1
    else:
        batch = frame.shape[0]
    theta = _ensure_theta_batch(theta, batch)
    a, b, c, d, log_aniso = _theta_to_abcd(theta, dim_names)
    return _warp_erp(frame, a, b, c, d, inverse=False, base_grid=base_grid, log_aniso=log_aniso)


def inverse_mobius(
    frame: torch.Tensor,
    theta: torch.Tensor,
    dim_names: Optional[list[str]] = None,
    base_grid: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    if frame.dim() == 3:
        batch = 1
    else:
        batch = frame.shape[0]
    theta = _ensure_theta_batch(theta, batch)
    a, b, c, d, log_aniso = _theta_to_abcd(theta, dim_names)
    return _warp_erp(frame, a, b, c, d, inverse=True, base_grid=base_grid, log_aniso=log_aniso)
