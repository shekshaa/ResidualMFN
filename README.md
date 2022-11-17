# Residual Multiplicative Filter Networks | NeurIPS (2022)

### [Project Page](https://shekshaa.github.io/ResidualMFN/) | [Paper](https://arxiv.org/abs/2206.00746)
Official PyTorch implementation of Residual MFN.<br>

[Residual Multiplicative Filter Networks for Multiscale Reconstruction]()<br>
[Shayan Shekarforoush](https://shekshaa.github.io),
[David B. Lindell](https://davidlindell.com),
[David Fleet](http://www.cs.toronto.edu/~fleet/),
[Marcus Brubaker](https://mbrubake.github.io/)<br>
University of Toronto, Vector Institute, York University <br>

<img src='./media/teaser.png'/>

## Dependencies
First create a conda environment and activate it:
```
conda env create -f environment.yml
conda activate resmfn
```
We also use [EMAN2](https://cryoem.bcm.edu/cryoem/downloads/view_eman2_versions) command line interface to align the predicted map with the ground truth before computing Fourier Shell Correlation (FSC) and resolution.

## Datasets
For the 2D image fitting task, download Natural and Text images from Google Drive and put them under `/data/images/`:
```
cd ./data/images/
gdown --id 1TtwlEDArhOMoH18aUyjIMSZ3WODFmUab  # Natural dataset
gdown --id 1V-RQJcMuk9GD4JCUn70o7nwQE0hEzHoT  # Text dataset
```

For cryo-EM experiment, use following script to generate synthetic dataset from 3D density map stored in `/path/to/volume`:
```
python3 generate_data.py /path/to/volume --ctf --snr 0.1 --n-projections 50000
```
To remove the CTF effect, omit `--ctf`. Also, options `--snr` and `--n-projections` specify the snr level and number of projections, respectively. Last command stores particle images, their orientation and ctf parameters within a new directory created next to `/path/to/volume`.

## Experiments

### 2D image Fitting
We first evaluate our proposal and compare it with the baseline [BACON](https://github.com/computational-imaging/bacon) on 2D image generalization task. We find this useful for didactic purposes, allowing us to clearly illustrate the benefit of skip connections and our new initialization scheme in the context of coarse-to-fine image reconstruction. 
To run BACON in a staged coarse-to-fine optimization stratgey, use following command:
```
python train_img.py /path/to/outdir --staged --init old --dataset dataset-name --gpu gpu-id
```
The first argument is the path to save output results, `--dataset` and `--gpu` specify dataset name (either `natural` or `text`) and gpu id, respectively. By adding `--fair`, in a more fair scenario, we let all the output layers to be optimized and allow the gradient backpropagate from the ouput of current stage:
```
python train_img.py /path/to/outdir --staged --init old --dataset dataset-name --fair --gpu gpu-id
```
To run our proposal in the same optimization scheme:
```
# only skip connection
python train_img.py /path/to/outdir --staged --init old --dataset dataset-name --residual --gpu gpu-id
# both skip connection and new frequency initialization
python train_img.py /path/to/outdir --staged --init new --dataset dataset-name --residual --gpu gpu-id 
```
Note that in the last command, the option `--init` is changed to `new` in order to use our specialized initialization scheme which allows explicit control over the amount of Fourier spectra support shared across different outputs. Note that, in all previous commands, `--init old` is used, namely the initialization proposed by BACON. Finally, remove `--staged` option to run the simplest baseline BACON without coarse-to-fine optimization:
```
python train_img.py /path/to/outdir --init old --dataset dataset-name --gpu gpu-id
```
### cryo-EM 3D Reconstruction
Simply run following script to examine the proposed network in default settings on cryo-EM 3D reconstruction, which also saves the reconstructed maps at all scales for each epoch:
```
python train_cryoem.py /path/to/volume /path/to/particle-images/ /path/to/outdir
```
Here are the arguments for the script `train_cryoem.py` and their descriptions:
```
usage: train_cryoem.py [-h] [--epochs EPOCHS] [--bacon-lr BACON_LR] [--bacon-iter BACON_ITER]
                       [--bacon-hidden-dim BACON_HIDDEN_DIM] [--bacon-hidden-layers BACON_HIDDEN_LAYERS] [--pose-lr POSE_LR]
                       [--pose-beta1 POSE_BETA1] [--pose-beta2 POSE_BETA2] [--pose-lr-decay POSE_LR_DECAY]
                       [--pose-iter POSE_ITER]
                       vol_path image_path outdir

positional arguments:
  vol_path              Volume path
  image_path            Particle images path
  outdir                Output directory

optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of epochs
  --bacon-lr BACON_LR   Learning rate of bacon
  --bacon-iter BACON_ITER
                        Number of iterations updating the structure
  --bacon-hidden-dim BACON_HIDDEN_DIM
                        Hidden dimension of coordinate network
  --bacon-hidden-layers BACON_HIDDEN_LAYERS
                        Number of hidden layers of coordinate network
  --pose-lr POSE_LR     Learning rate of poses optimizer
  --pose-beta1 POSE_BETA1
                        Beta1 for adam optimizer of poses
  --pose-beta2 POSE_BETA2
                        Beta2 for adam optimizer of poses
  --pose-lr-decay POSE_LR_DECAY
                        If use piecewise decay for pose learning rate
  --pose-iter POSE_ITER
                        Number of iterations updating poses
``` 
After obtaining reconstructions, use following script to first align them with the ground truth map and then compute Fourier Shell Correlation: 
```
export PATH="path/to/EMAN2:$PATH"
e2proc3d.py /path/to/reconstruction /path/to/aligned-reconstruction --align rotate_translate_3d_tree --alignref /path/to/ground-truth-map
e2proc3d.py /path/to/aligned-reconstruction /path/to/fsc.txt --calcfsc /path/to/ground-truth-map
```
Don't forget to put EMAN2 in the `PATH` variable before running `e2proc3d.py`. The last command generates an `fsc.txt` file which stores FSC along shells at different radius.
## Notebooks
Coming Soon...!
## Citation
```
@article{shekarforoush2022residual,
  title={Residual Multiplicative Filter Networks for Multiscale Reconstruction},
  author={Shekarforoush, Shayan and Lindell, David B and Fleet, David J and Brubaker, Marcus A},
  journal={arXiv preprint arXiv:2206.00746},
  year={2022}
}
```
## Acknowledgements
We thank Ali Punjani for numerous valuable discussions and for use of the cryoSPARC software package. 
This research was supported in part by the Province of Ontario, 
the Government of Canada, through NSERC, CIFAR, 
and the Canada First Research Excellence Fund for the Vision: Science to Applications (VISTA) programme, 
and by companies sponsoring the Vector Institute.
