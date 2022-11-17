import numpy as np
import matplotlib.pyplot as plt
import torch
import skimage.metrics
from torchvision.utils import make_grid

    
def fftn_center(img):
    return np.fft.fftshift(np.fft.fftn(img))


def fft2_center(img):
    return np.fft.fftshift(np.fft.fft2(img), axes=(-2, -1))


def ifft2_center(img):
    return np.fft.ifft2(np.fft.ifftshift(img, axes=(-2, -1)))


def torch_fft2_center(img):
    return torch.fft.fftshift(torch.fft.fft2(img), dim=(-2, -1))


def torch_ifft2_center(img):
    return torch.fft.ifft2(torch.fft.ifftshift(img, dim=(-2, -1)))


def fsc_from_file(path, plot=False, plot_path=None, apix=1):
    with open(path) as f:
        lines = f.readlines()
        half_D = len(lines) + 1
        xs = []
        fscs = []
        for line in lines:
            x, fsc = line.split()
            xs.append(float(x))
            fscs.append(float(fsc))
        fscs = np.asarray(fscs)
        w = np.where(fscs < 0.5)[0]
        if w.size:
            res = 1/xs[w[0]]*apix
        else:
            res = 1/xs[-1]*apix
        if plot:
            x_values = ["DC"] + ["{:.2f}A".format(1/xs[i]*apix) for i in range(half_D // 8, half_D, half_D // 8)]
            x_locs = [i/(2 * half_D) for i in range(0, half_D, half_D // 8)]
            plt.plot(xs,fscs)
            plt.axhline(y=0.5, color='r', linestyle='-')
            plt.title("FSC Curve, Resolution={:.2f}".format(res))
            plt.ylim((0, 1))
            plt.xlim((0, 0.5))
            plt.xticks(x_locs, x_values)
            if plot_path is not None:
                plt.savefig(plot_path)
                plt.close()

     
def fsc(vol1, vol2, plot=False, plot_path=None, apix=1):
    D = vol1.shape[0]
    x = np.arange(-D//2, D//2)
    x2, x1, x0 = np.meshgrid(x,x,x, indexing='ij')
    coords = np.stack((x0,x1,x2), -1)
    r = (coords**2).sum(-1)**.5

    vol1 = fftn_center(vol1)
    vol2 = fftn_center(vol2)

    prev_mask = np.zeros((D,D,D), dtype=bool)
    fsc = [1.0]
    for i in range(1,D//2):
        mask = r < i
        shell = np.where(mask & np.logical_not(prev_mask))
        v1 = vol1[shell]
        v2 = vol2[shell]
        p = np.vdot(v1,v2) / ((np.vdot(v1,v1)*np.vdot(v2,v2))**.5)
        fsc.append(p.real)
        prev_mask = mask
    fsc = np.asarray(fsc)
    x = np.arange(D//2)/D

    w = np.where(fsc < 0.5)[0]
    if w.size:
        res = 1/x[w[0]]*apix
    else:
        res = 1/x[-1]*apix
    
    if plot:
        x_values = ["DC"] + ["{:.2f}A".format(1/x[i]*apix) for i in range(D//16, D//2, D//16)]
        x_locs = [i/D for i in range(0, D//2, D//16)]
        plt.plot(x,fsc)
        plt.axhline(y=0.5, color='r', linestyle='-')
        plt.title("FSC Curve, Resolution={:.2f}".format(res))
        plt.ylim((0, 1))
        plt.xlim((0, 0.5))
        plt.xticks(x_locs, x_values)
        if plot_path is not None:
            plt.savefig(plot_path)
            plt.close()

    return res

def write_imgs_file(pred_img, path):
    pred_img = pred_img.permute((0, 3, 1, 2))
    spectrums = torch_fft2_center(pred_img)
    img_spectrums = torch.cat((pred_img, spectrums), dim=0)
    np_img_spectrums = img_spectrums.detach().cpu().numpy()
    np.save(path, np_img_spectrums)

def write_imgs(pred_img, gt_img, writer, iter, prefix):
    pred_img = pred_img.permute((0, 3, 1, 2))
    gt_img = gt_img.permute((2, 0, 1)).unsqueeze(0)
    imgs = torch.cat([gt_img, pred_img], dim=0)

    spectrums = torch_fft2_center(imgs)
    smax = torch.max(spectrums[0].real)
    spectrums = spectrums / smax * 50
    spectrums = torch.clamp(abs(torch.norm(spectrums, dim=1, keepdim=True)), 0, 1)**(1/2)
    spectrums = spectrums.repeat(1, 3, 1, 1)
    imgs_spectrums = torch.cat((imgs, spectrums), dim=0)
    writer.add_image(prefix + 'imgs_fft', make_grid(imgs_spectrums, scale_each=True,
                         normalize=True, nrow=imgs_spectrums.shape[0]//2), global_step=iter)
