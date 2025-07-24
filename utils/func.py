import torch
import numpy as np
import torch.nn.functional
import torch.nn as nn
import cigvis
from cigvis import colormap

class NormTensor:
    def __init__(self):
        pass
    
    def min_max_norm(self, x):
        x = torch.clip(x, min=-3.2, max=3.2)
        x_min = torch.min(x)
        x_max = torch.max(x)
        
        return (x - x_min) / (x_max - x_min)
    
    def __call__(self, x):
        x_mean = torch.mean(x)
        x_std = torch.std(x)

        x = (x - x_mean) / (x_std)
        return self.min_max_norm(x)

def _pad(img, infer_size):
    _, _, t, h, w = img.size()
    pad_t = t % infer_size[0]
    pad_h = h % infer_size[1]
    pad_w = w % infer_size[2]
    pad3d = nn.ReflectionPad3d(padding=(0, infer_size[2] - pad_w if pad_w != 0 else 0,
                                        0, infer_size[1] - pad_h if pad_h != 0 else 0,
                                        0, infer_size[0] - pad_t if pad_t != 0 else 0))

    new_img = pad3d(img)

    return new_img


def plot_result(seismic, pred, remain_prob=False):
    if not remain_prob:
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
    seismic = (seismic - np.mean(seismic)) / np.std(seismic)
    seismic = np.clip(seismic, -3.2, 3.2)
    seismic = (seismic - np.min(seismic)) / (np.max(seismic) - np.min(seismic))
    seismic = np.transpose(seismic, axes=(1, 2, 0))
    pred = np.transpose(pred, axes=(1, 2, 0)) 
    
    fg_cmap = colormap.set_alpha_except_min('jet', alpha=1)
    node = cigvis.create_overlay(seismic,
                                pred,
                                pos=[[0], [0], [105]],
                                bg_cmap='gray',
                                fg_cmap=fg_cmap,
                                fg_interpolation='nearest')
    
    cigvis.plot3D(node, size=(800, 800))

def model_infer(model, img, infer_size, device, norm=NormTensor()):
    img = torch.from_numpy(img)[None, None]
    if infer_size is None:
        img = img.to(device)
        with torch.no_grad() and torch.amp.autocast('cuda'):
            img = norm(img)
            _, pred = model(img)
            
        pred = torch.softmax(pred, dim=1)
        pred = pred[0, 1, ...].detach().cpu().numpy()
        return pred

    if isinstance(infer_size, int):
        infer_size = (infer_size, infer_size, infer_size)
    
    _, _, ori_d, ori_h, ori_w = img.shape
    img = _pad(img, infer_size)
    
    result = np.zeros(img.shape, np.single)
    _, _, D, H, W = img.shape

    ds = [idx for idx in range(0, D+1, infer_size[0]) if idx + infer_size[0] <= D]
    hs = [idx for idx in range(0, H+1, infer_size[1]) if idx + infer_size[1] <= H]
    ws = [idx for idx in range(0, W+1, infer_size[2]) if idx + infer_size[2] <= W]
    
    for d in ds:
        for h in hs:
            for w in ws:
                cube = img[:, :, d:d+infer_size[0],
                                h:h+infer_size[1], w:w+infer_size[2]]
                cube = norm(cube, show=True).to(device)
                with torch.amp.autocast('cuda') and torch.no_grad():
                    _, pred = model(cube)
                    
                pred = torch.softmax(pred, dim=1)
                pred = pred[:, 1, ...].detach().cpu().numpy()
                result[:, 0, d:d+infer_size[0], h:h+infer_size[1], w:w+infer_size[2]] = pred
    return result[0, 0, :ori_d, :ori_h, :ori_w]

