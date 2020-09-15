import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

# Sample code to show multi process communication with DDP.
# https://pytorch.org/tutorials/intermediate/dist_tuto.html


"""Blocking point-to-point communication."""
def run_send_recv_blocking(rank, size):

    print("Distributed function ...")

    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        print(f"send data from {rank}")
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        print(f"received data from {rank}")
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])

    pass

"""Non-blocking point-to-point communication."""
def run_send_recv_nonblocking(rank, size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        req = dist.isend(tensor=tensor, dst=1)
        print('Rank 0 started sending')
    else:
        # Receive tensor from process 0
        req = dist.irecv(tensor=tensor, src=0)
        print('Rank 1 started receiving')
    req.wait()
    print('Rank ', rank, ' has data ', tensor[0])

""" All-Reduce example."""
def run_allreduce_sum(rank, size):
    """ Simple point-to-point communication. """
    group = dist.new_group(list(range(size)))
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.reduce_op.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    # dist.init_process_group allows processes to communicate with each other by sharing their locations.
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 4
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run_allreduce_sum))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
