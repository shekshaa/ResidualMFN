import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import warnings

from torch.utils.tensorboard import SummaryWriter
from modules import AllModels, FrequencyMarchingBACON
from mrc import write_mrc
from utils import fsc
from load_data import ParticleImages
from tqdm import tqdm
from kornia.geometry.conversions import rotation_matrix_to_quaternion


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('vol_path', type=os.path.abspath, help="Volume path")
    parser.add_argument('image_path', type=os.path.abspath, help="Particle images path")
    parser.add_argument('outdir', type=os.path.abspath, help="Output directory")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")

    parser.add_argument('--bacon-lr', type=float, default=1e-3, help="Learning rate of bacon")
    parser.add_argument('--bacon-iter', type=int, default=5, help="Number of iterations updating the structure")
    parser.add_argument('--bacon-hidden-dim', type=int, default=128, help="Hidden dimension of coordinate network")
    parser.add_argument('--bacon-hidden-layers', type=int, default=3, help="Number of hidden layers of coordinate network")

    parser.add_argument('--pose-lr', type=float, default=1e-2, help="Learning rate of poses optimizer")
    parser.add_argument('--pose-beta1', type=float, default=0.9, help="Beta1 for adam optimizer of poses")
    parser.add_argument('--pose-beta2', type=float, default=0.9, help="Beta2 for adam optimizer of poses")
    parser.add_argument('--pose-lr-decay', type=float, default=1., help="If use piecewise decay for pose learning rate")
    parser.add_argument('--pose-iter', type=int, default=20, help="Number of iterations updating poses")
    
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    n_gpu = torch.cuda.device_count()

    # train_schedule = [15, 30] # pdb1ol5
    train_schedule = [25, 50] # pdb4ake
    
    logs_path = args.outdir
    writer = SummaryWriter(os.path.join(logs_path, "summaries"))

    particle_images_dataset = ParticleImages(args.vol_path, args.image_path)
    apix = particle_images_dataset.apix
    sidelen = particle_images_dataset.sidelen
    batch_size = 600
    particle_images = DataLoader(particle_images_dataset, batch_size=batch_size, shuffle=True)

    tensor_gt_rotations = torch.tensor(particle_images_dataset.rotations).float()
    gt_q = rotation_matrix_to_quaternion(tensor_gt_rotations)
    g_dist = [np.pi/4, np.pi/2]
    model = AllModels(sidelen=sidelen, 
                        hidden_dim=args.bacon_hidden_dim, 
                        hidden_layers=args.bacon_hidden_layers,
                        n_data=particle_images_dataset.n_projections,
                        gt_rotations=tensor_gt_rotations,
                        g_dist=g_dist,
                        apix=apix)
    optim1 = torch.optim.Adam(lr=args.bacon_lr, params=list(model.model.parameters()))
    optim2 = torch.optim.Adam(lr=args.pose_lr, betas=(args.pose_beta1, args.pose_beta2), 
                                params=list(filter(lambda p: p.requires_grad, model.poses.parameters())))
    
    model = nn.DataParallel(model)
    model.to(device)

    vol_iterations = args.bacon_iter
    pose_iterations = args.pose_iter
    
    pose_lr_decay = False
    if args.pose_lr_decay != 1.:
        pose_lr_decay = True
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim2, milestones=train_schedule, gamma=args.pose_lr_decay)

    log = "Canonical Model Config:\n"
    log += "bacon lr {}, pose lr={}\n".format(args.bacon_lr, args.pose_lr)
    log += "hidden dim={}, layers={}\n".format(args.bacon_hidden_dim, args.bacon_hidden_layers)
    log += '-------------\n\n'
    with open(os.path.join(args.outdir, "logs.txt"), "w") as f:
        f.write(log)
    print(log, end="")
    
    for epoch in range(args.epochs):
        avg_img_loss = 0
        log = 'Epoch {}:\n'.format(epoch + 1)
        distances = []
        if (epoch + 1) in train_schedule:
            model.module.model.stop_after += 1
            log += "Unfreezing layer {} and start training on its output\n".format(model.module.model.stop_after + 1)
            if isinstance(model.module.model, FrequencyMarchingBACON):
                model.module.model.bacon_layers[model.module.model.stop_after].freeze = False

        est_q = rotation_matrix_to_quaternion(model.module.get_rotations().cpu())
        distances = 2 * torch.arccos(torch.abs(torch.sum(gt_q * est_q, axis=-1))) / np.pi * 180.
        if epoch == args.epochs - 1:
            np_distances = distances.detach().cpu().numpy()
            with open(os.path.join(args.outdir, "final_distances.npy"), 'wb') as f:
                np.save(f, np_distances)
            tmp = np_distances > 60
            sample_est_rotations = est_q[tmp].cpu().numpy()[:10]
            sample_gt_rotations = gt_q[tmp].detach().cpu().numpy()[:10]
            np.savez(os.path.join(args.outdir, "sample_wrong_rotations.npz"), est_rot=sample_est_rotations, gt_rot=sample_gt_rotations)
            tmp = np_distances < 15
            sample_est_rotations = est_q[tmp].cpu().numpy()[:10]
            sample_gt_rotations = gt_q[tmp].detach().cpu().numpy()[:10]
            np.savez(os.path.join(args.outdir, "sample_right_rotations.npz"), est_rot=sample_est_rotations, gt_rot=sample_gt_rotations)
                
        writer.add_histogram("geodesic_distance", distances, epoch+1)

        for batch_n, (gt_img, ctf_params, idx) in tqdm(enumerate(particle_images), total=len(particle_images)):
            gt_img = gt_img.to(device)
            idx = idx.to(device)
            if particle_images_dataset.apply_ctf:
                ctf_params = ctf_params.to(device)
            else:
                ctf_params = None
            
            model.module.setup_alternate(optimize="volume")
            for iteration in range(vol_iterations):
                optim1.zero_grad(set_to_none=True)
                optim2.zero_grad(set_to_none=True)
                img_loss = model(gt_img, idx, ctf_params)
                img_loss = img_loss.mean()
                avg_img_loss += img_loss.item()
                img_loss.backward()
                optim1.step()

            model.module.setup_alternate(optimize="poses")
            for iteration in range(pose_iterations):
                optim1.zero_grad(set_to_none=True) 
                optim2.zero_grad(set_to_none=True)
                img_loss = model(gt_img, idx, ctf_params)
                img_loss = img_loss.mean()
                avg_img_loss += img_loss.item()
                img_loss.backward()
                optim2.step()
                model.module.apply_constraints(idx)
            model.module.update_poses(idx)
        
        avg_img_loss /= len(particle_images)
        log += "Average image l2 loss = {:.5f}\n".format(avg_img_loss)
        epoch_logs_path = os.path.join(logs_path, 'epoch_{}'.format(epoch + 1))
        if not os.path.exists(epoch_logs_path):
            os.makedirs(epoch_logs_path)
        with torch.no_grad():
            predicted_vol = model.module.evaluate_model().view(-1, sidelen, sidelen, sidelen).detach().cpu().numpy()
            for i in range(3):
                predicted_vol_path = os.path.join(epoch_logs_path, 'predicted_vol_scale{}.mrc'.format(i))
                write_mrc(predicted_vol_path, predicted_vol[i], apix=apix)
            selected_predicted_vol = predicted_vol[model.module.model.stop_after]
            vol_l2_loss = np.mean((selected_predicted_vol - particle_images_dataset.vol) ** 2)
            log += "Vol l2 loss = {:.5f}".format(vol_l2_loss)
        log += '----------\n'
        print(log, end="")
        with open(os.path.join(args.outdir, "logs.txt"), "a") as f:
            f.write(log)
        if pose_lr_decay:
            scheduler.step()
    writer.close()
