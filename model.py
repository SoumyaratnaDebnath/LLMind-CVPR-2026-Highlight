
import torch
import torch.nn as nn

class MLPParamNet(nn.Module):
    def __init__(self, z_dim=8, hidden=64, mobius_layers=1, param_limits=None):
        super().__init__()
        self.mobius_layers = mobius_layers
        self.z = nn.Parameter(torch.randn(z_dim) * 0.1)
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden*2), nn.ReLU(inplace=True),
            nn.Linear(hidden*2, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 4*mobius_layers), nn.Tanh()
        )
        if param_limits is None:
            self.register_buffer("param_limits", None)
        else:
            limits = torch.tensor(param_limits, dtype=torch.float32)
            if limits.numel() not in (4, 4 * mobius_layers):
                raise ValueError("param_limits must have length 4 or 4 * mobius_layers")
            self.register_buffer("param_limits", limits)

    def _expanded_param_limits(self, raw):
        if self.param_limits is None:
            return None
        limits = self.param_limits
        if limits.numel() == 4:
            limits = limits.repeat(self.mobius_layers)
        return limits.to(device=raw.device, dtype=raw.dtype)

    def forward(self):
        raw = self.net(self.z)
        limits = self._expanded_param_limits(raw)
        if limits is not None:
            raw = raw * limits
        return list(raw)
