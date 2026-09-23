import torch, torch.nn as nn

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None: min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v: new_v += divisor
    return new_v

NET_CONFIG = {
    "blocks2": [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]],
}

class ConvBNLayer(nn.Module):
    def __init__(self, num_channels, filter_size, num_filters, stride, num_groups=1):
        super().__init__()
        self.conv = nn.Conv2d(num_channels, num_filters, filter_size, stride,
                              padding=(filter_size - 1) // 2, groups=num_groups, bias=False)
        self.bn = nn.BatchNorm2d(num_filters)
        self.hardswish = nn.Hardswish()
    def forward(self, x):
        return self.hardswish(self.bn(self.conv(x)))

class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channel, channel // reduction, 1, 1, 0)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel // reduction, channel, 1, 1, 0)
        self.hardsigmoid = nn.Hardsigmoid()
    def forward(self, x):
        identity = x
        x = self.hardsigmoid(self.conv2(self.relu(self.conv1(self.avg_pool(x)))))
        return identity * x

class DepthwiseSeparable(nn.Module):
    def __init__(self, num_channels, num_filters, stride, dw_size=3, use_se=False):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = ConvBNLayer(num_channels, dw_size, num_channels, stride, num_groups=num_channels)
        if use_se: self.se = SEModule(num_channels)
        self.pw_conv = ConvBNLayer(num_channels, 1, num_filters, 1)
    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se: x = self.se(x)
        return self.pw_conv(x)

class PPLCNetDocOrientation(nn.Module):
    def __init__(self, in_channels=3, scale=1.0, class_dim=4):
        super().__init__()
        self.conv1 = ConvBNLayer(in_channels, 3, make_divisible(16 * scale), 2)
        for name in ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]:
            setattr(self, name, nn.Sequential(*[
                DepthwiseSeparable(make_divisible(in_c * scale), make_divisible(out_c * scale), s, k, se)
                for k, in_c, out_c, s, se in NET_CONFIG[name]]))
        last_conv_in = make_divisible(512 * scale)
        last_conv_out = int(last_conv_in * 2.5)
        self.last_conv = nn.Conv2d(last_conv_in, last_conv_out, 1, 1, 0, bias=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(last_conv_out, class_dim)
    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks6(self.blocks5(self.blocks4(self.blocks3(self.blocks2(x)))))
        x = self.pool(self.last_conv(x))
        return self.fc(torch.flatten(x, 1))

def load_model(path, device='cpu'):
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck['state_dict'] if 'state_dict' in ck else ck
    meta = {k: v for k, v in ck.items() if k != 'state_dict'} if 'state_dict' in ck else {}
    model = PPLCNetDocOrientation(3, meta.get('scale', 1.0), meta.get('num_classes', 4))
    conv = {}
    for k, v in sd.items():
        nk = k.replace('._mean', '.running_mean').replace('._variance', '.running_var')
        if nk == 'fc.weight' and v.shape[0] != meta.get('num_classes', 4):
            v = v.t()
        conv[nk] = v
    missing, unexpected = model.load_state_dict(conv, strict=False)
    model.eval().to(device)
    return model, meta, missing, unexpected
