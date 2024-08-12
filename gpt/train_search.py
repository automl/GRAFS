"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import types
import random

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

from custom_functions.utils import replace_ac_function

from architecture.architecture import Architecture
from activation_cell.ac_cell import ActivationCell

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
ac_func = 'gelu'
id = 42
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
resume = False
# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2'  # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # used to simulate larger batch sizes
batch_size = 12  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 600000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl'  # 'nccl', 'gloo', etc.
check_min_train_loss = False
min_train_loss = 1e6
# system
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster
train_portion = 0.75
arch_weight_decay = 0.001
# lr = 1e-3
arch_learning_rate = 1e-3
warmstart_iters = 100
start_shrinking = 100
results_folder = 'exps_gpt'
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# os.environ["WANDB_MODE"]="offline"


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

def random_seed(seed=1337, rank=0):
    np.random.seed(seed + rank)
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    # if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed + rank)


if master_process:
    os.makedirs(out_dir, exist_ok=True)
random_seed(id, rank=seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

search_data = train_data
n = len(search_data)
train_data = search_data[:int(n*train_portion)]
val_data = search_data[int(n*train_portion):]
############################

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


ac_func = ActivationCell().to(device)


iter_num = 0
best_val_loss = 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)  # start with model_args from command line
if init_from == 'scratch' and not resume:
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif resume:
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, f'ckpt_{type(ac_func).__name__}_{id}_{n_head}_{n_layer}_{init_from}.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2') and not resume:
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size  # so that the checkpoint will have the right value


n_params = sum(p.numel() for p in model.parameters())
print('n_params before:', n_params)

replace_ac_function(model, nn.ReLU, ac_func)
replace_ac_function(model, nn.GELU, ac_func)

condition_array = start_shrinking - 1 + np.logspace(np.log10(1), np.log10(max_iters-start_shrinking), len(ac_func) - 6, base=10).astype(int)
print(condition_array)
condition_idx = 0
print(model)
model.to(device)

n_params = sum(p.numel() for p in model.parameters())
print('n_params after:', n_params)

if not os.path.exists(f"results/{results_folder}") or not os.path.isdir(f"results/{results_folder}"):
    os.makedirs(f"results/{results_folder}", exist_ok=True)



# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

criterion = nn.CrossEntropyLoss()
arch = Architecture(model, ac_func, criterion, scaler, lr=arch_learning_rate, arch_weight_decay=arch_weight_decay)

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if resume:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        ac_func.forward_type = "drnas" if split=='train' else 'discretized'
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log and master_process:
    import wandb

    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train')  # fetch the very first batch
X_arch, Y_arch = get_batch('val')
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0

t = time.time()


os.makedirs("plots", exist_ok=True)
os.makedirs(f"results/{results_folder}/search_{dataset}_{'gpt'}_{start_shrinking}_{arch_weight_decay>0}", exist_ok=True)
with open(
        f"results/{results_folder}/search_{dataset}_{'gpt'}_{start_shrinking}_{arch_weight_decay>0}/arch_{id}_{arch_learning_rate}_{max_iters}_{warmstart_iters}_{'gelu'}.csv",
        "w") as f:
    f.write(f"epoch,train_loss,val_loss,activation,best_activation,time")

ac_func.best_genotype = ac_func.genotype()

while True:

    if iter_num in condition_array:
            ac_func.drop_op(np.count_nonzero(condition_array == iter_num))

    if all([sum(m)==1 for m in ac_func.mask]):
        print('fully discretized architecture found before max epochs reached')
        break

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num == 0 and eval_only:
        break

    ac_func.forward_type = "drnas"
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            arch.step(model, X_arch, Y_arch, False, False, False, False, gradient_accumulation_steps, micro_step)

        # immediately async prefetch next batch while model is doing the forward pass on the GPU 
        X_arch, Y_arch = get_batch('val')
    
    optimizer.zero_grad(set_to_none=True)

    # ac_func.forward_type == "discretized"
    if iter_num <= warmstart_iters:  # and args.comp == "gelu":
        ac_func.forward_type = "gelu"
    """elif iter_num <= args.warmstart_epoch:  # and args.comp == "relu":
        ac_func.forward_type = "relu"
    elif iter_num <= args.warmstart_epoch:  # and args.comp == "elu":
        ac_func.forward_type = "elu"
    elif iter_num <= args.warmstart_epoch:  # and args.comp == "silu":
        ac_func.forward_type = "silu"
    elif iter_num <= args.warmstart_epoch:  # and args.comp == "leakyrelu":
        ac_func.forward_type = "leakyrelu"
    else:
        ac_func.forward_type = "drnas"""

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)


    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")


    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()

        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            ac_func.best_genotype = ac_func.genotype()
        
    if iter_num % log_interval == 0 and master_process:

        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "best_val_loss": best_val_loss,
                "lr": lr,
                "mfu": running_mfu * 100,  # convert to percentage
                "search_runtime": time.time() - t,
                "id": id,
                "num remaining ops": sum([sum(m) for m in ac_func.mask]),
                "num remaining unary ops 0": sum(ac_func.mask[0]),
                "num remaining unary ops 1": sum(ac_func.mask[1]),
                "num remaining binary ops 2": sum(ac_func.mask[2]),
                "num remaining unary ops 3": sum(ac_func.mask[3]),
                "num remaining unary ops 1": sum(ac_func.mask[4]),
                "num remaining binary ops 2": sum(ac_func.mask[5]),
                "run_name": wandb_run_name
                # "chart": fig
            })

    iter_num += 1
    local_iter_num += 1

    if iter_num % eval_interval == 0 and master_process:

        torch.save(ac_func.best_genotype, f"results/{results_folder}/search_{id}.pth")
        activation = [type(op).__name__ if len(tuple((float(p.cpu()) for p in op.parameters())))==0 else type(op).__name__+f"-{tuple((float(p.cpu()) for p in op.parameters()))}" for op in ac_func.genotype()]
        best_activation = [type(op).__name__ if len(tuple((float(p.cpu()) for p in op.parameters())))==0 else type(op).__name__+f"-{tuple((float(p.cpu()) for p in op.parameters()))}" for op in ac_func.best_genotype]
        # if iter_num >= start_shrinking:
        print(f'activation at iter {iter_num} =', activation)
        print(f'best activation till iter {iter_num} =', best_activation)

        with open(
            f"results/{results_folder}/search_{dataset}_{'gpt'}_{start_shrinking}_{arch_weight_decay>0}/arch_{id}_{arch_learning_rate}_{max_iters}_{warmstart_iters}_{'gelu'}.csv",
            "a") as f:
            f.write(
                f"\n{iter_num},{losses['train']},{losses['val']},{activation},{best_activation},{time.time()-t}"
            )


    # termination conditions
    if iter_num > max_iters:
        break

print(f"best val loss: {best_val_loss}, ac_func: {type(ac_func).__name__}")
if ddp:
    destroy_process_group()
