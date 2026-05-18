
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

def build_scorer(name="mse", device=None):

    if name == "mse":
        return mse_loss
    
    if name in ["lhuman", "human", "vsi+dists"]:
        vsi = VSI(device=device)
        dists = DISTS(device=device)
        def lhuman(x, y):
            x, y = normalize_img(x), normalize_img(y)
            v = vsi(x, y)
            d = dists(x, y)
            loss = 0.25 * (1 - v) + 0.25 * (1 - d) + 0.5 * F.mse_loss(x, y)
            return loss
        return lhuman

    print("#-----------------------------")
    print(f"[warn] unknown loss '{name}', falling back to MSE")
    print("#-----------------------------")
    return mse_loss

def mse_loss(x, y):
    return F.mse_loss(x, y)

def normalize_img(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

class DISTS(nn.Module):
    def __init__(self, device=None):
        assert device is not None, "Please specify device (e.g., 'cuda' or 'cpu')"
        super().__init__()
        vgg_pretrained = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES).features[:16].to(device)
        for p in vgg_pretrained.parameters():
            p.requires_grad = False
        self.vgg = vgg_pretrained.eval()
        self.device = device
        self.layer_weights = [0.1, 0.2, 0.3, 0.4]

    def forward(self, x, y):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device)[None, :, None, None]
        x = (x - mean) / std
        y = (y - mean) / std

        feats_x, feats_y = [], []
        out_x, out_y = x, y
        for i, layer in enumerate(self.vgg):
            out_x, out_y = layer(out_x), layer(out_y)
            if isinstance(layer, nn.ReLU):
                feats_x.append(out_x)
                feats_y.append(out_y)

        dists_score = 0
        for w, fx, fy in zip(self.layer_weights, feats_x, feats_y):
            mu_x, mu_y = fx.mean([2,3]), fy.mean([2,3])
            sigma_x, sigma_y = fx.var([2,3]), fy.var([2,3])
            ssim_map = (2*mu_x*mu_y + 1e-5) / (mu_x**2 + mu_y**2 + 1e-5)
            texture_map = (2*sigma_x.sqrt()*sigma_y.sqrt() + 1e-5) / (sigma_x + sigma_y + 1e-5)
            dists_score += w * (ssim_map * texture_map).mean()
        return dists_score.clamp(0, 1)

class VSI(nn.Module):
    def __init__(self, device=None):
        assert device is not None, "Please specify device (e.g., 'cuda' or 'cpu')"
        super().__init__()
        self.device = device

    def forward(self, x, y):
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=self.device).float().view(1,1,3,3)
        sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], device=self.device).float().view(1,1,3,3)
        if x.shape[1] > 1:
            sobel_x = sobel_x.repeat(x.shape[1],1,1,1)
            sobel_y = sobel_y.repeat(x.shape[1],1,1,1)

        grad_x_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        grad_y_x = F.conv2d(x, sobel_y, padding=1, groups=x.shape[1])
        grad_x_y = F.conv2d(y, sobel_x, padding=1, groups=y.shape[1])
        grad_y_y = F.conv2d(y, sobel_y, padding=1, groups=y.shape[1])

        sal_x = torch.sqrt(grad_x_x**2 + grad_y_x**2 + 1e-6)
        sal_y = torch.sqrt(grad_x_y**2 + grad_y_y**2 + 1e-6)
        sal_w = (sal_x + sal_y) / 2.0

        mu_x = F.avg_pool2d(x, 3, 1, 1)
        mu_y = F.avg_pool2d(y, 3, 1, 1)
        sigma_x = F.avg_pool2d(x*x, 3, 1, 1) - mu_x**2
        sigma_y = F.avg_pool2d(y*y, 3, 1, 1) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, 3, 1, 1) - mu_x*mu_y
        C1, C2 = 0.01**2, 0.03**2
        ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
        vsi_score = (ssim_map * sal_w).mean()
        return vsi_score.clamp(0, 1)
