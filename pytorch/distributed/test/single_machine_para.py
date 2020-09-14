import os
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
# Simple code of running DDP on multiple CPU, this code is changed from above sample multiple GPU DDP code.

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # comment out below code if running on multiple GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

# TODO, verify each parameter on distributed models.
def demo_basic(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    model = ToyModel()
    ddp_model = DDP(model)
    # model = ToyModel().to(rank)
    # ddp_model = DDP(model, device_ids=[rank])

    #print(f"Model's initial parameters on rank {rank}.")
    #print_named_parameters(ddp_model, rank)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Sample code to save model
    # save_load_model

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    # labels = torch.randn(20, 5).to(rank)
    labels = torch.randn(20, 5)

    # Gradient synchronization communications take place during the backward pass and overlap with the backward computation. 
    # When the backward() returns, param.grad already contains the synchronized gradient tensor. 
    loss_fn(outputs, labels).backward()
    #print(f"Model's parameters after backward on rank {rank}.")
    #print_named_parameters(ddp_model, rank)

    optimizer.step()
    #print(f"Model's parameters after synchronization on rank {rank}.")
    #print_named_parameters(ddp_model, rank)

    cleanup()
    print(f"Finish basic DDP example on rank {rank}.")

def save_load_model(ddp_model, rank):
    CHECKPOINT_PATH = tempfile.gettempdir() + "/model.checkpoint"
    if rank == 0:
        # All processes should see same parameters as they all start from same
        # random parameters and gradients are synchronized in backward passes.
        # Therefore, saving it in one process is sufficient.
        torch.save(ddp_model.state_dict(), CHECKPOINT_PATH)

    # Use a barrier() to make sure that process 1 loads the model after process
    # 0 saves it.
    dist.barrier()
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    ddp_model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=map_location))

    if rank == 0:
        os.remove(CHECKPOINT_PATH)

def print_named_parameters(model, rank):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}, {param.data}, rank={rank}")

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    run_demo(demo_basic, 2)