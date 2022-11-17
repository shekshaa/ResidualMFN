import torch
from torch.utils.data import Dataset
import numpy as np
import os
from mrc import parse_mrc
from PIL import Image


class ParticleImages(Dataset):
    def __init__(self, vol_path, image_path):
        self.particle_images = np.load(os.path.join(image_path, "particle_stack.npy"), 'r')
        self.rotations = np.load(os.path.join(image_path, "poses.npy"), 'r')
        self.apply_ctf = False
        ctf_path = os.path.join(image_path, "ctf.npy")
        if os.path.exists(ctf_path):
            self.ctf = np.load(os.path.join(image_path, "ctf.npy"), 'r')
            self.apply_ctf = True
        self.n_projections = self.rotations.shape[0]
        self.vol, hdr = parse_mrc(os.path.join(vol_path, 'volume.mrc'))
        self.apix = hdr.get_apix()
        self.sidelen = self.vol.shape[0]
        freqs = np.fft.fftshift(np.fft.fftfreq(self.sidelen))
        self.freqs = np.stack(np.meshgrid(freqs, freqs), axis=-1).reshape(-1, 2) / self.apix
    
    def __len__(self):
        return self.n_projections

    def __getitem__(self, idx):
        projected_image = torch.tensor(self.particle_images[idx]).float()
        if self.apply_ctf:
            ctf_params = torch.tensor(self.ctf[idx]).float()
        else:
            ctf_params = torch.zeros((7,)) # dummy ctf
        return projected_image, ctf_params, idx


class ImageDataset(Dataset):
    def __init__(self, path, just_one=True):
        imgs = np.load(path)
        imgs = np.concatenate([imgs['train_data'], imgs['test_data']], axis=0)
        self.n_imgs = imgs.shape[0]
        self.train_res = imgs.shape[1] // 2
        self.test_res = imgs.shape[1]
        self.just_one = just_one
        train_imgs = []
        for i in range(self.n_imgs):
            img = Image.fromarray(imgs[i]).resize((self.train_res, self.train_res), Image.ANTIALIAS)
            train_imgs.append(np.array(img) / 255.)
        self.train_imgs = np.stack(train_imgs, axis=0)
        self.test_imgs = imgs / 255.
    
    def __len__(self):
        if self.just_one:
            return 1
        return self.n_imgs

    def __getitem__(self, idx):
        return self.train_imgs[idx], self.test_imgs[idx]