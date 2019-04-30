# Slimmable

## Search command
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=0 python3  ./search.py --type block_binomial --data ../datasets --width 0.25,0.5,0.75,1.0 --gpu 0 --nSamples 2 --workers 0 --search_learning_rate 1E-4 --lmbda 0.0 --search_learning_rate_min 1e-5 --modelFlops ./flops.pth.tar

## Training checkpoint command
PYTHONPATH=./ CUDA_VISIBLE_DEVICES=0 python3 train_model.py --folderPath ../results_block_binomial/checkpoints/ --data ../datasets/ --repeatNum 1 --optimal_epochs 150

## Plot checkpoints command ???