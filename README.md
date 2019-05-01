# Towards Learning of Filter-Level Heterogeneous Compression of Convolutional Neural Networks

Code repository for Slimmable NAS (Chapter 4): https://arxiv.org/abs/1904.09872

## Installation
We recommend using TensorFlow with pip virtual environment.
Installing instructions can be found in the following link: https://www.tensorflow.org/install/pip

After the virtual environment activation, we have to install the required packages:
```
pip install -r requirements.txt
```
Make sure the current directory is the repository main directory.

## Datasets
We worked with CIFAR-10 and CIFAR-100.

Both can be automatically downloaded by torchvision.

## Usage


### Search
To carry out Slimmable search, use the following command:
```
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=0,1 python3 ./search.py --type block_binomial --data ../datasets --width 0.25,0.5,0.75,1.0 --gpu 0,0,1 --nSamples 3 --workers 0 --search_learning_rate 1E-4 --lmbda 0.0 --search_learning_rate_min 1e-5 --modelFlops ./flops.pth.tar
```
Make sure the current directory is the repository main directory.
Notice it is possible to train multiple configurations on the same GPU. the --gpu flag determines how many configurations we train simultaneously. Therefore, --gpu 0,0,1 as in the command line example mean we train 3 configurations simultaneously, two configurations on GPU #0 and another configuration on GPU #1.

### Checkpoint evaluation
During the search, we sample configurations from the current distribution.
Use the following command in order to train the sampled configurations and evaluate their quality.
```
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=0 python3 train_model.py --folderPath ../results_block_binomial/checkpoints/ --data ../datasets/ --repeatNum 1 --optimal_epochs 150
```
The argument --folderPath holds the path to the folder containing the checkpoints we would like to train.
It is possible to train different checkpoints from the same folder on different GPUs simultaneously, just replace CUDA_VISIBLE_DEVICES value. 

## Plot checkpoints command ???

## Acknowledgments  
The research was funded by ERC StG RAPID.  
  
## Citation  
If our work helped you in your research, please consider cite us.  
```
@ARTICLE{2019arXiv190409872Z,
       author = {{Zur}, Yochai and {Baskin}, Chaim and {Zheltonozhskii}, Evgenii and
         {Chmiel}, Brian and {Evron}, Itay and {Bronstein}, Alex M. and
         {Mendelson}, Avi},
        title = "{Towards Learning of Filter-Level Heterogeneous Compression of Convolutional Neural Networks}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning, Computer Science - Neural and Evolutionary Computing},
         year = "2019",
        month = "Apr",
          eid = {arXiv:1904.09872},
        pages = {arXiv:1904.09872},
archivePrefix = {arXiv},
       eprint = {1904.09872},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190409872Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
    
This work is licensed under the Creative Commons Attribution-NonCommercial  
4.0 International License. To view a copy of this license, visit  
[http://creativecommons.org/licenses/by-nc/4.0/](http://creativecommons.org/licenses/by-nc/4.0/) or send a letter to  
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.