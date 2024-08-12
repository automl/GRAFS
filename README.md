# Gradient-based Activation Function Search (GRAFS)

This is a repository for reproducing the experiments mentioned in the paper _"Searching for Customized Activation Functions
Efficiently with Gradient Descent"_.
This Repository features two main sections:

1. ResNet experiments:`rn`, parts of code originates form [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10)
2. ViT experiments:`vit`, parts of code originates form [vision-transformers-cifar10](https://github.com/kentaroy47/vision-transformers-cifar10)
3. GPT Experiments: `gpt`, parts of code originates form [nanoGPT](https://github.com/karpathy/nanoGPT)


## Get Data
1. For Computer Vision (ResNet or ViT) experiments data run

```
cd rn
python download_data.py
```

2. For GPT experiments data run

```
cd gpt

cd data/tinystories
python tinystories.py
cd ../..
```

## ResNet Experiments

Below we provide scripts to perform search and evaluation of activation functions in ResNet models.

### Search

To run a search on ResNet20 / CIFAR10:\
`python search_arch.py --model=res20 --data cifar10 --epochs 50 --start_shrinking 2 --warmstart_epoch 1 --batch-size=64 --grad_acumm=2 --id=<seed> --comp relu --arch_lr=0.0006 --workers=1 --results_folder=<folder_to_save_discovered_activations>`\


### Evaluation

To evaluate discovered activation with id=i on ResNet20 / CIFAR10:\
`python train_test.py --model=res20 --data cifar10 --epochs 200 --batch-size=128 --grad_acumm=1 --id=i --seed=<seed> --ac=discovered --workers=1 --results_folder=<folder_where_discovered_activations_is_saved>`\

To evaluate baseline activation SiLU on ResNet32 / SVHN_Core:\
`python train_test.py --model=res32 --data svhn_core --epochs 200 --batch-size=128 --grad_acumm=1 --seed=<seed> --ac=silu --workers=1 --results_folder=<folder_where_discovered_activations_is_saved>`\


## ViT Experiments

Below we provide scripts to perform search and evaluation of activation functions on Computer Vision tasks.

### Search

To run a search on ViT-tiny / CIFAR10:\
`python search_arch.py --search_epochs 50 --start_shrinking 2 --warmstart_epoch 1 --net vit_tiny --bs 128 --grad_acumm 4 --data cifar10 --lr 0.001 --aug --n_epochs 50 --id=<seed> --comp gelu --results_folder <folder_to_save_discovered_activations>`\

--comp specifies the warm-starting activation


### Evaluation


To evaluate discovered activation with id=i on ViT-tiny / CIFAR10:\
`python train_test.py --eval --net vit_tiny --bs 512 --data cifar10 --lr 0.001 --aug --n_epochs 500 --id=i --seed=<seed> --ac discovered --results_folder <folder_where_discovered_activations_is_saved>`\

To evaluate baseline activation ELU on ViT-small / CIFAR100:\
`python train_test.py --eval --net vit_small --bs 512 --data cifar100 --lr 0.001 --aug --n_epochs 500 --seed=<seed> --ac elu --results_folder <folder_where_discovered_activations_is_saved>`\



## GPT Experiments

Below we provide scripts to perform search and evaluation of activation functions on Language Modelling tasks.

### Search

To run a search on miniGPT / TinyStories:\
`python train_search.py config/train_shakespeare.py --compile=False --dtype=float16 --dataset=tinystories --n_layer=3 --n_head=3 --n_embd=192 --log_interval=10 --eval_interval=10 --batch_size=4 --gradient_accumulation_steps=160 --max_iters=1000 --lr_decay_iters=1000 --warmup_iters=0 --warmstart_iters=100 --start_shrinking=200 --train_portion=0.75 --id=<seed> --results_folder <folder_to_save_discovered_activations>`\


### Evaluation


To evaluate discovered activation with id=i on tinyGPT / TinyStories:\
`python train.py config/train_shakespeare_char.py --compile=False --dtype=float16 --dataset=tinystories --ac=discovered --n_layer=6 --n_head=6 --n_embd=384 --log_interval=10 --eval_interval=10 --batch_size=16 --gradient_accumulation_steps=40 --max_iters=10000 --lr_decay_iters=10000 --warmup_iters=100 --id=i --seed=<seed> --results_folder <folder_where_discovered_activations_is_saved>`\

To evaluate baseline activation GELU on smallGPT / TinyStories:\
`python train.py config/train_shakespeare_char.py --compile=False --dtype=float16 --dataset=tinystories --n_layer=9 --n_head=9 --n_embd=576 --log_interval=10 --eval_interval=10 --batch_size=16 --gradient_accumulation_steps=40 --max_iters=10000 --lr_decay_iters=10000 --warmup_iters=100 --seed=<seed>`\
