from mrc import parse_mrc
import torch
from torch.utils.data import Dataset, DataLoader
from modules import RotateProject
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
from ctf import generate_random_ctf_params, compute_ctf, compute_safe_freqs
from utils import fft2_center, ifft2_center
import argparse


def random_quaternion(N):
    u = np.random.uniform(0, 1, size=(N,))
    v = np.random.uniform(0, 1, size=(N,))
    w = np.random.uniform(0, 1, size=(N,))
    q = np.stack([np.sqrt(1 - u) * np.sin(2*np.pi*v), 
                  np.sqrt(1 - u) * np.cos(2*np.pi*v), 
                  np.sqrt(u) * np.sin(2*np.pi*w),
                  np.sqrt(u) * np.cos(2*np.pi*w)], axis=-1)
    return q

    
def quaternion2Rmatrix(q):
    qi = q[:, 0]
    qj = q[:, 1]
    qk = q[:, 2]
    qr = q[:, 3]
    r11 = 1 - 2*(qj**2 + qk**2)
    r12 = 2*(qi*qj - qk*qr)
    r13 = 2*(qi*qk + qj*qr)
    r21 = 2*(qi*qj + qk*qr)
    r22 = 1 - 2*(qi**2 + qk**2)
    r23 = 2*(qj*qk - qi*qr)
    r31 = 2*(qi*qk - qj*qr)
    r32 = 2*(qj*qk + qi*qr)
    r33 = 1 - 2*(qi**2 + qj**2)
    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    return R


class AffineGenerator(Dataset):
    def __init__(self, n_projections):
        rotations = np.transpose(quaternion2Rmatrix(random_quaternion(n_projections)), axes=(2, 0, 1))
        self.rotations = torch.tensor(rotations).float()
        self.n_projections = n_projections
        
    def __len__(self):
        return self.n_projections
    
    def __getitem__(self, idx):
        return self.rotations[idx]


def generate_data(path, n_projections=50000, snr=0.1, apply_ctf=True, batch_size=300):
    vol, hdr = parse_mrc(os.path.join(path, 'volume.mrc'))
    apix = hdr.get_apix()
    sidelen = vol.shape[0]
    print('apix:', apix)
    print('volume size:', vol.shape)
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    vol_tensor = torch.tensor(np.transpose(vol, axes=(2, 1, 0))).view(1, 1, sidelen, sidelen, sidelen).to(device)

    affine_generator = AffineGenerator(n_projections)
    affine_data_loader = DataLoader(affine_generator, batch_size=batch_size)

    rotate_project = RotateProject(sidelen, apix)
    rotate_project = nn.DataParallel(rotate_project)
    rotate_project.to(device)
    
    particle_stack = []
    all_ctfs = []
    for rotations in tqdm(affine_data_loader):
        rotations = rotations.to(device)
        batch = rotations.shape[0]
        clean_projected_images = rotate_project(vol_tensor.expand(batch, -1, -1, -1, -1), rotations).detach().cpu().numpy()
        if apply_ctf:
            ctf_params = generate_random_ctf_params(batch)
            all_ctfs.append(np.concatenate(ctf_params, axis=1).squeeze())
            freqs_mag, angles_rad = compute_safe_freqs(sidelen, apix)
            ctf = compute_ctf(freqs_mag, angles_rad, *ctf_params).reshape(batch, sidelen, sidelen)
            ctf_corrupted_fourier_images = ctf * fft2_center(clean_projected_images)
            ctf_corrupted_real_images = ifft2_center(ctf_corrupted_fourier_images).real
        else:
            ctf_corrupted_real_images = clean_projected_images
        if snr is not None:
            noise_std = np.sqrt(np.var(ctf_corrupted_real_images, axis=(-2, -1), keepdims=True) / snr)
            expand_noise_std = np.tile(noise_std, (1, sidelen, sidelen))
            ctf_corrupted_noisy_images = np.random.normal(ctf_corrupted_real_images, expand_noise_std)
            particle_stack.append(ctf_corrupted_noisy_images)
        else:
            particle_stack.append(ctf_corrupted_real_images)
    particle_stack = np.concatenate(particle_stack, axis=0)
    if not apply_ctf:
        save_path = os.path.join(path, 'N{}_snr{}'.format(n_projections, snr))
    else:
        all_ctfs = np.concatenate(all_ctfs, axis=0)
        save_path = os.path.join(path, 'N{}_snr{}_ctf'.format(n_projections, snr))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(save_path, "particle_stack.npy"), 'wb') as f:
        np.save(f, particle_stack)
    with open(os.path.join(save_path, "poses.npy"), "wb") as f:
        np.save(f, affine_generator.rotations.detach().cpu().numpy())
    if apply_ctf:
        with open(os.path.join(save_path, "ctf.npy"), 'wb') as f:
            np.save(f, all_ctfs)
    print("Synthetic Data Generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=os.path.abspath, help="Volume path")
    parser.add_argument('--ctf', action='store_true', default=False, help="Apply CTF")
    parser.add_argument('--snr', type=float, default=0.1, help="Signal to Noise Ratio")
    parser.add_argument('--n-projections', type=int, default=50000, help="Number of images")
    args = parser.parse_args()
    generate_data(path=args.path, snr=args.snr, apply_ctf=args.ctf, n_projections=args.n_projections)
    