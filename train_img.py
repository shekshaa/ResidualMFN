import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import warnings

from torch.utils.tensorboard import SummaryWriter
from modules import BACON
from utils import write_imgs, write_imgs_file
from load_data import ImageDataset
from tqdm import tqdm
from math import log10


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', type=os.path.abspath, help="Output directory")
    parser.add_argument('--staged', action='store_true', default=False, help="staged training")
    parser.add_argument('--fair', action='store_true', default=False, help="To make vanilla bacon fair")
    parser.add_argument('--residual', action='store_true', default=False, help="residual connection")
    parser.add_argument('--init', type=str, default='old', choices=['old', 'new'], help="choice of initialization")
    parser.add_argument('--dataset', type=str, default='natural', help="Natural or Text images", choices=['natural', 'text'])
    parser.add_argument('--epochs', type=int, default=10000, help="Number of epochs")
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--save-epoch', type=int, default=100, help="Saving logs and model epochs")
    parser.add_argument('--just-one', action='store_true', default=False, help="Train only on first image")

    parser.add_argument('--lr', type=float, default=5e-3, help="Learning rate of bacon")
    parser.add_argument('--hidden-dim', type=int, default=256, help="Hidden dimension")
    parser.add_argument('--hidden-layers', type=int, default=4, help="Number of hidden layers")
    
    args = parser.parse_args()

    warnings.filterwarnings("ignore")

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.gpu) if use_cuda else 'cpu')
    n_gpu = torch.cuda.device_count()

    logs_path = os.path.join(args.outdir, args.dataset)
    if not args.staged:
        method = "vanilla"
    elif (not args.residual) and (args.init == "old"):
        if args.fair:
            method = "staged_fair"
        else:
            method = "staged_unfair"
    elif (args.residual) and (args.init == "old"):
        method = "staged_residual"
    elif (args.residual) and (args.init == "new"):
        method = "staged_residual_newinit"
    else:
        raise NotImplemented("wrong combination!")
    logs_path = os.path.join(args.outdir, method)
    writer = SummaryWriter(os.path.join(logs_path, "summaries"))
    if args.dataset == 'natural':
        images_path = './data/images/data_div2k.npz'
    else:
        images_path = './data/images/data_2d_text.npz'
    img_dataset = ImageDataset(images_path, just_one=args.just_one)
    train_sidelen = img_dataset.train_res
    test_sidelen = img_dataset.test_res
    img_loader = DataLoader(img_dataset, batch_size=1, shuffle=False)

    train_points = np.linspace(-0.5, 0.5, num=train_sidelen, endpoint=False) + 1/(2*train_sidelen)
    train_mgrid = np.stack(np.meshgrid(train_points, train_points), axis=-1).reshape(-1, 2)
    train_mgrid = torch.tensor(train_mgrid, device=device).float()

    test_points = np.linspace(-0.5, 0.5, num=test_sidelen, endpoint=False) + 1/(2*test_sidelen)
    test_mgrid = np.stack(np.meshgrid(test_points, test_points), axis=-1).reshape(-1, 2)
    test_mgrid = torch.tensor(test_mgrid, device=device).float()

    log = "Canonical Model Config:\n"
    log += "lr {}\n".format(args.lr)
    log += "hidden dim={}, layers={}\n".format(args.hidden_dim, args.hidden_layers)
    log += '-------------\n\n'
    with open(os.path.join(logs_path, "config.txt"), "w") as f:
        f.write(log)
    save_img_path = os.path.join(logs_path, "images")
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
    lambdas = (0.3, 2.)
    train_schedule = [500, 1000, 2000] 

    final_train_psnr = []
    final_test_psnr = []
    final_train_mse = []
    final_test_mse = []
    for img_id, (train_img, test_img) in enumerate(img_loader):
        train_img = train_img.squeeze()
        test_img = test_img.squeeze()
        model = BACON(in_size=2, 
                    hidden_size=args.hidden_dim,
                    out_size=3,
                    hidden_layers=args.hidden_layers, 
                    staged=args.staged,
                    initialization=args.init,
                    residual=args.residual,
                    out_bias=True,
                    frequency=(train_sidelen, train_sidelen), 
                    lambdas = lambdas,
                    quantization_interval=2*np.pi,
                    all_out=True,
                    relu=False)

        optim = torch.optim.Adam(lr=args.lr, params=list(model.parameters()))
        model.to(device)
        train_img = train_img.to(device)
        test_img = test_img.to(device)
        
        for epoch in tqdm(range(args.epochs)):
            if args.staged:
                if (epoch + 1) in train_schedule:
                    model.stop_after += 1
                    model.bacon_layers[model.stop_after].freeze = False
                    if args.fair:
                        model.bacon_layers[model.stop_after-1].stop_grad = True # fair comparsion with stop gradient between z and out.
            optim.zero_grad(set_to_none=True)
            out = model(train_mgrid).view(-1, train_sidelen, train_sidelen, 3)
            total_mse = 0
            for i in range(args.hidden_layers):
                train_mse = ((out[i] - train_img) ** 2).mean()
                if (not args.staged) or (args.fair):
                    total_mse += train_mse
                train_psnr = 10 * log10(1 / train_mse.item())
                writer.add_scalar('id{}/train/mse/scale{}'.format(img_id, i + 1), train_mse.item(), epoch + 1)
                writer.add_scalar('id{}/train/psnr/scale{}'.format(img_id, i + 1), train_psnr, epoch + 1)
                if i == args.hidden_layers - 1:
                    writer.add_scalar('id{}/train/final_mse'.format(img_id), train_mse.item(), epoch + 1)
                    writer.add_scalar('id{}/train/final_psnr'.format(img_id), train_psnr, epoch + 1)
            if args.staged and (not args.fair):
                total_mse = ((out[model.stop_after] - train_img) ** 2).mean()
            total_mse.backward()
            optim.step()
            with torch.no_grad():
                # train log
                if not ((epoch + 1) % args.save_epoch):
                    write_imgs(out, train_img, writer, epoch + 1, 'id{}/train/'.format(img_id))
                if ((epoch + 1) in train_schedule) or (epoch == args.epochs - 1):
                    path = os.path.join(save_img_path, "train_img_{}_ep{}.npy".format(img_id, epoch + 1))
                    write_imgs_file(out, path)
                # test log
                out = model(test_mgrid).view(-1, test_sidelen, test_sidelen, 3)
                for i in range(args.hidden_layers):
                    test_mse = ((out[i] - test_img) ** 2).mean().item()
                    test_psnr = 10 * log10(1 / test_mse)
                    writer.add_scalar('id{}/test/mse/scale{}'.format(img_id, i + 1), test_mse, epoch + 1)
                    writer.add_scalar('id{}/test/psnr/scale{}'.format(img_id, i + 1), test_psnr, epoch + 1)
                    if i == args.hidden_layers - 1:
                        writer.add_scalar('id{}/test/final_mse'.format(img_id), test_mse, epoch + 1)
                        writer.add_scalar('id{}/test/final_psnr'.format(img_id), test_psnr, epoch + 1)
                if not ((epoch + 1) % args.save_epoch):
                    write_imgs(out, test_img, writer, epoch + 1, 'id{}/test/'.format(img_id))
                if ((epoch + 1) in train_schedule) or (epoch == args.epochs - 1):
                    path = os.path.join(save_img_path, "test_img_{}_ep{}.npy".format(img_id, epoch + 1))
                    write_imgs_file(out, path)

        final_train_psnr.append(train_psnr)
        final_test_psnr.append(test_psnr)
        final_train_mse.append(train_mse.item())
        final_test_mse.append(test_mse)
    writer.close()

    with open(os.path.join(args.outdir, "train_psnr.txt"), "w") as f:
        f.write('\n'.join(str(psnr) for psnr in final_train_psnr))
    with open(os.path.join(args.outdir, "test_psnr.txt"), "w") as f:
        f.write('\n'.join(str(psnr) for psnr in final_test_psnr))

    with open(os.path.join(args.outdir, "train_mse.txt"), "w") as f:
        f.write('\n'.join(str(mse) for mse in final_train_mse))
    with open(os.path.join(args.outdir, "test_mse.txt"), "w") as f:
        f.write('\n'.join(str(mse) for mse in final_test_mse))
