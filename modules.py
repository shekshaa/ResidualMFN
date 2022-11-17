import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
import numpy as np
from torch import nn
from kornia.geometry.transform import warp_affine3d
import torch.fft as fft
from utils import torch_fft2_center, torch_ifft2_center
from ctf import torch_compute_ctf, torch_compute_safe_freqs


def mfn_weights_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input), np.sqrt(6/num_input))


class ContinuousFourierLayer(nn.Module):

    def __init__(self, in_features, out_features, weight_scale):
        super(ContinuousFourierLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        # sample discrete uniform distribution of frequencies
        for i in range(self.linear.weight.data.shape[1]):
            # uniform 
            init = torch.rand_like(self.linear.weight.data[:, i], ) * 2*weight_scale[i] - weight_scale[i]
            self.linear.weight.data[:, i] = init

        self.linear.weight.requires_grad = False
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class ContinuousCustomizedFourierLayer(nn.Module): 
    def __init__(self, in_features, out_features, weight_scale, lambdas):
        super(ContinuousCustomizedFourierLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        lambda1, lambda2 = lambdas
        
        # sample discrete uniform distribution of frequencies
        if in_features == 2:
            p = torch.tensor([[0, 1], [1, 0], [1, 1], [-1, 1]]) # without [0, 0]
        elif in_features == 3:
            lst = []
            for i in range(-1, 2):
                for j in range(-1, 2):
                    lst.append([i, j, 1])
            for i in range(-1, 2):
                lst.append([i, 1, 0])
            lst.append([1, 0, 0])
            p = torch.tensor(lst)
        else:    
            raise NotImplementedError
        idx = torch.randint(0, p.shape[0], size=(self.linear.weight.shape[0],))
        r = p[idx]
        for i in range(self.linear.weight.data.shape[1]):
            init = (torch.rand_like(self.linear.weight.data[:, i]) * 2 * weight_scale[i] - weight_scale[i]) * lambda1 + lambda2 * weight_scale[i] * r[:, i]
            self.linear.weight.data[:, i] = init

        self.linear.weight.requires_grad = False
        print("original uniform: {}".format(lambda1 * weight_scale[i]))
        print("offset: {}".format(lambda2 * weight_scale[i]))
        print("max, min frequency: {}, {}".format(torch.max(self.linear.weight.data), torch.min(self.linear.weight.data)))
        self.linear.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        return torch.sin(self.linear(x))


class BaconLayer(nn.Module):
    def __init__(self, in_size, out_size, hidden_size, band, lambdas, initialization, out_bias=True, quantization_interval=2*np.pi, freeze=True, relu=False):
        super().__init__()
        self.stop_grad = False
        if initialization == 'old':
            self.filter = ContinuousFourierLayer(in_features=in_size, out_features=hidden_size, weight_scale=band)
            self.linear = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=True)
        else:
            self.filter = ContinuousCustomizedFourierLayer(in_features=in_size, out_features=hidden_size, weight_scale=band, lambdas=lambdas)
            self.linear = nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=False)
        self.out = nn.Linear(in_features=hidden_size, out_features=out_size, bias=out_bias)
        self.linear.apply(mfn_weights_init)
        self.out.apply(mfn_weights_init)
        self.freeze = freeze
        self.relu = relu
    
    def forward(self, coords, z):
        if self.freeze:
            with torch.no_grad():    
                filtered_input = self.filter(coords)
                new_z = self.linear(z) * filtered_input
                if self.relu:
                    out = nn.ReLU(inplace=True)(self.out(new_z))
                else:
                    out = self.out(new_z)
        else:
            filtered_input = self.filter(coords)
            new_z = self.linear(z) * filtered_input
            if self.relu:
                if self.stop_grad:
                    out = nn.ReLU(inplace=True)(self.out(new_z.detach()))
                else:
                    out = nn.ReLU(inplace=True)(self.out(new_z))
            else:
                if self.stop_grad:
                    out = self.out(new_z.detach())
                else:
                    out = self.out(new_z)
        return new_z, out


class BACON(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, hidden_layers, staged=False, residual=False, initialization='old', out_bias=True, frequency=(128, 128), lambdas=(0.3, 2), quantization_interval=2*np.pi, all_out=False, relu=False):
        super().__init__()
        self.quantization_interval = quantization_interval
        self.hidden_layers = hidden_layers
        self.frequency = frequency
        self.all_out = all_out
        self.stop_after = 0 if staged else None
        self.staged = staged
        self.residual = residual
        lambda1, lambda2 = lambdas
        partial_scales = [1 / ((1 + lambda1 + lambda2)**(hidden_layers - i)) for i in range(hidden_layers + 1)]
        scales = [partial_scales[0]]
        for i in range(1, len(partial_scales)):
            scales.append(partial_scales[i] - partial_scales[i-1])

        band = [np.pi * freq * scales[0] for freq in frequency]
        self.first_filter = ContinuousFourierLayer(in_size, hidden_size, band)
        
        self.bacon_layers = []
        for i in range(1, hidden_layers + 1):
            if staged:
                if i == 1:
                    freeze = False
                else:
                    freeze = True
            else:
                freeze = False
            if initialization == 'old':
                band = [np.pi * freq * scales[i] for freq in frequency]
            self.bacon_layers.append(BaconLayer(in_size=in_size, 
                                                out_size=out_size, 
                                                hidden_size=hidden_size, 
                                                initialization=initialization,
                                                band=band, 
                                                lambdas=lambdas,
                                                out_bias=out_bias, 
                                                quantization_interval=quantization_interval,
                                                freeze=freeze,
                                                relu=relu))
            if initialization == 'new':
                band = [np.pi * freq * partial_scales[i] for freq in frequency]
            print(partial_scales[i] * np.pi * frequency[0])
        self.bands = partial_scales[1:]
        self.bacon_layers = nn.ModuleList(self.bacon_layers)

    def get_band(self,):
        return self.bands[self.stop_after]

    def forward_just_one(self, coords):
        z = self.first_filter(coords)
        output = 0
        for i in range(self.hidden_layers):
            z, out = self.bacon_layers[i](coords, z)
            if self.residual:
                output += out
            else:
                output = out
            if self.staged and i == self.stop_after:
                return output
    
    def forward_all(self, coords):
        all_outs = []
        z = self.first_filter(coords)
        for i in range(self.hidden_layers):
            z, out = self.bacon_layers[i](coords, z)
            if self.residual and len(all_outs) > 0:
                all_outs.append(out + all_outs[-1])
            else:
                all_outs.append(out)
        return torch.stack(all_outs, dim=0)

    def forward(self, coords):
        if self.all_out:
            return self.forward_all(coords)
        else:
            return self.forward_just_one(coords)


class FrequencyMarchingBACON(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, hidden_layers=3, out_bias=True, frequency=(128, 128), lambdas=(0.3, 2), quantization_interval=2*np.pi, all_out=False, relu=False):
        super().__init__()
        self.quantization_interval = quantization_interval
        self.hidden_layers = hidden_layers
        self.frequency = frequency
        self.all_out = all_out
        self.stop_after = 0
        lambda1, lambda2 = lambdas
        band = [np.pi * freq / ((1 + lambda1 + lambda2)**(hidden_layers)) for freq in frequency]
        self.first_filter = ContinuousFourierLayer(in_size, hidden_size, band)
        self.bacon_layers = []
        for i in range(1, hidden_layers + 1):
            if i == 1:
                freeze = False
            else:
                freeze = True
            self.bacon_layers.append(BaconLayer(in_size=in_size, 
                                                out_size=out_size, 
                                                hidden_size=hidden_size, 
                                                band=band, 
                                                initialization='new',
                                                lambdas=lambdas,
                                                out_bias=out_bias, 
                                                quantization_interval=quantization_interval,
                                                freeze=freeze,
                                                relu=relu))
            band = [np.pi * freq / ((1 + lambda1 + lambda2)**(hidden_layers - i)) for freq in frequency]
            print(band)
        self.bands = [1 / ((1 + lambda1 + lambda2)**(hidden_layers - i)) for i in range(1, hidden_layers + 1)]
        self.bacon_layers = nn.ModuleList(self.bacon_layers)
    
    def get_band(self,):
        return self.bands[self.stop_after]

    def forward_just_one(self, coords):
        output = 0
        z = self.first_filter(coords)
        for i in range(self.hidden_layers):
            z, out = self.bacon_layers[i](coords, z)
            output = output + out
            if i == self.stop_after:
                break
        return output
    
    def forward_all(self, coords):
        output = 0
        all_outs = []
        z = self.first_filter(coords)
        for i in range(self.hidden_layers):
            z, out = self.bacon_layers[i](coords, z)
            output = output + out
            all_outs.append(output)
        return torch.stack(all_outs, dim=0)

    def forward(self, coords):
        if self.all_out:
            return self.forward_all(coords)
        else:
            return self.forward_just_one(coords)


def generate_random_axis(n_points):
    u = np.random.normal(loc=0, scale=1, size=(n_points, 3))
    u /= np.sqrt(np.sum(u ** 2, axis=-1, keepdims=True))
    return u


class PoseModel(nn.Module):
    def __init__(self, n_data, gt_rotations=None, g_dist=[np.pi/6, np.pi/3]):
        super(PoseModel, self).__init__()
        if gt_rotations is not None:
            random_axis = torch.tensor(generate_random_axis(n_data))
            random_angle = g_dist[0] + torch.rand(n_data, 1) * (g_dist[1] - g_dist[0])
            self.rotations = torch.nn.Parameter(torch.matmul(self.convert(random_axis, random_angle).float(), gt_rotations), requires_grad=False)
        else:
            random_axis = torch.tensor(generate_random_axis(n_data))
            random_angle = torch.rand(n_data, 1) * np.pi
            self.rotations = torch.nn.Parameter(self.convert(random_axis, random_angle).float(), requires_grad=False)
            
        self.perturbations_axis = torch.nn.Parameter(torch.randn(n_data, 3) * 1e-6, requires_grad=True)
        self.perturbations_angle = torch.nn.Parameter(torch.rand(n_data, 1) * 1e-6, requires_grad=True)
        
    def convert(self, axis, angle):
        assert axis.shape[0] == angle.shape[0]
        batch_size = axis.shape[0]
        z_skew = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], device=axis.device)
        y_skew = torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], device=axis.device)
        x_skew = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]], device=axis.device)
        x_skew = x_skew.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)
        y_skew = y_skew.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)
        z_skew = z_skew.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)
        axis_skew = axis[:, 0].view(batch_size, 1, 1, 1) * x_skew + axis[:, 1].view(batch_size, 1, 1, 1) * y_skew + axis[:, 2].view(batch_size, 1, 1, 1) * z_skew
        axis_skew = axis_skew.squeeze(-1)
        angle = angle.view(-1, 1, 1)
        R = torch.eye(3, device=axis.device).unsqueeze(0).expand(batch_size, -1, -1) + torch.sin(angle) * axis_skew + (1 - torch.cos(angle)) * torch.matmul(axis_skew, axis_skew)
        return R
    
    def constrain_angle(self, idx):
        with torch.no_grad():
            self.perturbations_angle.data[idx].clamp_(min=0., max=np.pi)
    
    def constrain_axis(self, idx):
        with torch.no_grad():
            self.perturbations_axis.data[idx] /= torch.sqrt(torch.sum(self.perturbations_axis.data[idx] ** 2, dim=-1, keepdim=True))
            
    def update(self, idx):
        with torch.no_grad():
            self.rotations.data[idx] = torch.matmul(self.convert(self.perturbations_axis.data[idx], self.perturbations_angle.data[idx]), self.rotations.data[idx])
            self.perturbations_axis.data.normal_(0, 1e-6)
            self.perturbations_angle.data.uniform_(0, 1e-6)

    def forward(self, idx):
        perturb_rotations = self.convert(self.perturbations_axis[idx], self.perturbations_angle[idx])
        return torch.matmul(perturb_rotations, self.rotations[idx])


class RotateProject(nn.Module):
    def __init__(self, sidelen, apix):
        super(RotateProject, self).__init__()
        self.sidelen = sidelen
        self.freqs_mag, self.angle_rad = torch_compute_safe_freqs(sidelen, apix)
    
    def dissect_ctf_params(self, ctf_params):
        dfu = ctf_params[:, 0].view(-1, 1, 1)
        dfv = ctf_params[:, 1].view(-1, 1, 1)
        dfang_deg = ctf_params[:, 2].view(-1, 1, 1)
        kv = ctf_params[:, 3].view(-1, 1, 1)
        cs = ctf_params[:, 4].view(-1, 1, 1)
        w = ctf_params[:, 5].view(-1, 1, 1)
        phase = ctf_params[:, 6].view(-1, 1, 1)
        return (dfu, dfv, dfang_deg, kv, cs, w, phase)

    def forward(self, vol, rotations, gt_img=None, ctf_params=None):
        dv = rotations.device
        sidelen = vol.shape[-1]
        rotated_offset = sidelen / 2 * (torch.ones(3, 1, device=dv) - torch.matmul(rotations, torch.ones(3, 1, device=dv)))
        warp_affine = torch.cat([rotations, rotated_offset], axis=-1)
        warped_vols = warp_affine3d(vol, warp_affine, dsize=(sidelen, sidelen, sidelen), flags='bilinear')
        projections = warped_vols.sum(2).squeeze(1)
        if gt_img is None:
            return projections
        else:
            if ctf_params is None:
                img_loss = ((projections - gt_img) ** 2).mean()
            else:
                ctf = torch_compute_ctf(self.freqs_mag.to(dv), self.angle_rad.to(dv), *self.dissect_ctf_params(ctf_params))
                ctf_corrupted_fourier_images = ctf * torch_fft2_center(projections)
                ctf_corrupted_real_images = torch_ifft2_center(ctf_corrupted_fourier_images).real
                img_loss = ((ctf_corrupted_real_images - gt_img) ** 2).mean()
        return img_loss


class GaussianFilter(nn.Module):
    def __init__(self, sidelen):
        super(GaussianFilter, self).__init__()
        self.sidelen = sidelen
    
    def forward(self, img, band):
        fft_img = fft.fftshift(fft.fft2(img), dim=(-2, -1))
        sigma = (band * self.sidelen) / 3
        mean = self.sidelen / 2
        X = torch.linspace(0, self.sidelen, self.sidelen, device=img.device)
        Y = torch.linspace(0, self.sidelen, self.sidelen, device=img.device)
        X, Y = torch.meshgrid(X, Y)
        gmask = torch.exp(-((X - mean)**2/(2 * sigma**2) + (Y - mean)**2/(2 * sigma**2)))
        filtered_fft_img = fft_img * gmask.unsqueeze(0).expand(fft_img.shape[0], -1, -1)
        filtered_img = fft.ifft2(fft.ifftshift(filtered_fft_img, dim=(-2, -1)))
        return filtered_img.real


class AllModels(nn.Module):
    def __init__(self, sidelen, hidden_dim, hidden_layers, n_data, gt_rotations=None, g_dist=None, apix=None, gabor=False):
        super(AllModels, self).__init__()
        self.model = FrequencyMarchingBACON(in_size=3, 
                                            hidden_size=hidden_dim, 
                                            out_size=1, 
                                            hidden_layers=hidden_layers, 
                                            out_bias=True,
                                            frequency=(sidelen, sidelen, sidelen), 
                                            lambdas=(0.3, 2.), 
                                            quantization_interval=2*np.pi,
                                            gabor=gabor,
                                            all_out=False,
                                            relu=True)
        self.filter = GaussianFilter(sidelen=sidelen)
        self.poses = PoseModel(n_data=n_data, gt_rotations=gt_rotations, g_dist=g_dist)
        self.projector = RotateProject(sidelen=sidelen, apix=apix)
        self.sidelen = sidelen
        points = np.linspace(-0.5, 0.5, num=sidelen, endpoint=False) + 1/(2*sidelen)
        full_mgrid = np.stack(np.meshgrid(points, points, points), axis=-1).reshape(-1, 3)
        self.full_mgrid = torch.tensor(full_mgrid).float()

    def setup_alternate(self, optimize="volume"):
        if optimize == "volume":
            for params in self.poses.parameters():
                params.requires_grad = False
            for params in self.model.parameters():
                params.requires_grad = True
        else:
            for params in self.poses.parameters():
                params.requires_grad = True
            for params in self.model.parameters():
                params.requires_grad = False

    def get_rotations(self):
        return self.poses.rotations.detach()
    
    def apply_constraints(self, idx):
        self.poses.constrain_angle(idx)
        self.poses.constrain_axis(idx)

    def update_poses(self, idx):
        self.poses.update(idx)

    def evaluate_model(self):
        predicted_vol = self.model.forward_all(self.full_mgrid.cuda())
        return predicted_vol

    def forward(self, gt_img, idx, ctf_params=None):
        n_imgs = gt_img.shape[0]
        rotations = self.poses(idx)
        band = self.model.get_band()
        bandlimited_gt_projections = self.filter(img=gt_img, band=band)
        predicted_vol = self.model(self.full_mgrid.to(gt_img.device)).view(self.sidelen, self.sidelen, self.sidelen).permute(2, 1, 0)
        img_loss = self.projector(predicted_vol.view(1, 1, self.sidelen, self.sidelen, self.sidelen).expand(n_imgs, -1, -1, -1, -1), 
                                    rotations, bandlimited_gt_projections, ctf_params)
        return img_loss
