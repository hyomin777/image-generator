import torch


def normalize(
        tensor,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    ):
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def rgb_to_hsv(img):
    r, g, b = img[0], img[1], img[2]
    maxc = torch.max(img, dim=0).values
    minc = torch.min(img, dim=0).values
    v = maxc
    s = (maxc - minc) / (maxc + 1e-6)
    s[maxc == 0] = 0

    deltac = maxc - minc
    h = torch.zeros_like(maxc)
    mask = deltac != 0
    idx = (maxc == r) & mask
    h[idx] = ((g - b)[idx] / deltac[idx]) % 6
    idx = (maxc == g) & mask
    h[idx] = (2 + (b - r)[idx] / deltac[idx])
    idx = (maxc == b) & mask
    h[idx] = (4 + (r - g)[idx] / deltac[idx])
    h = h / 6.0
    h[h < 0] += 1
    return torch.stack([h, s, v], dim=0)


def hsv_to_rgb(img):
    h, s, v = img[0], img[1], img[2]
    h = h * 6
    i = torch.floor(h).long()
    f = h - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    i = i % 6
    conditions = [
        (i == 0, torch.stack((v, t, p), dim=0)),
        (i == 1, torch.stack((q, v, p), dim=0)),
        (i == 2, torch.stack((p, v, t), dim=0)),
        (i == 3, torch.stack((p, q, v), dim=0)),
        (i == 4, torch.stack((t, p, v), dim=0)),
        (i == 5, torch.stack((v, p, q), dim=0)),
    ]
    out = torch.zeros_like(img)
    for cond, val in conditions:
        out[:, cond] = val[:, cond]
    return out


def color_jitter(img, brightness=0, contrast=0, saturation=0, hue=0):
    img = img.clone()

    # Brightness
    if brightness > 0:
        factor = torch.empty(1, device=img.device).uniform_(1 - brightness, 1 + brightness).item()
        img = img * factor

    # Contrast
    if contrast > 0:
        mean = img.mean(dim=(1, 2), keepdim=True)
        factor = torch.empty(1, device=img.device).uniform_(1 - contrast, 1 + contrast).item()
        img = (img - mean) * factor + mean

    # Saturation and Hue - convert to HSV first
    if saturation > 0 or hue > 0:
        img = rgb_to_hsv(img)
        if saturation > 0:
            factor = torch.empty(1, device=img.device).uniform_(1 - saturation, 1 + saturation).item()
            img[1] = (img[1] * factor).clamp(0, 1)
        if hue > 0:
            factor = torch.empty(1, device=img.device).uniform_(-hue, hue).item()
            img[0] = (img[0] + factor) % 1.0
        img = hsv_to_rgb(img)

    return img.clamp(0, 1)
