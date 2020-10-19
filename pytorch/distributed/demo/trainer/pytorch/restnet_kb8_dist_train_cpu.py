"""

This code is based on https://github.com/pytorch/tutorials/blob/master/beginner_source/aws_distributed_training_tutorial.py,
**Chi Wang** changed it to CPU mode, so it can run distributedly without the GPU requirements. 
For detail please checkout https://pytorch.org/tutorials/beginner/aws_distributed_training_tutorial.html 

(advanced) PyTorch 1.0 Distributed Trainer with Amazon AWS
=============================================================

**Author**: `Nathan Inkawhich <https://github.com/inkawhich>`_

**Edited by**: `Teng Li <https://github.com/teng-li>`_

"""

######################################################################
# Distributed Training Code
# -------------------------
#
# With the instances running and the environments setup we can now get
# into the training code. Most of the code here has been taken from the
# `PyTorch ImageNet
# Example <https://github.com/pytorch/examples/tree/master/imagenet>`__
# which also supports distributed training. This code provides a good
# starting point for a custom trainer as it has much of the boilerplate
# training loop, validation loop, and accuracy tracking functionality.
# However, you will notice that the argument parsing and other
# non-essential functions have been stripped out for simplicity.
#
# In this example we will use
# `torchvision.models.resnet18 <https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.resnet18>`__
# model and will train it on the
# `torchvision.datasets.STL10 <https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.STL10>`__
# dataset. To accomodate for the dimensionality mismatch of STL-10 with
# Resnet18, we will resize each image to 224x224 with a transform. Notice,
# the choice of model and dataset are orthogonal to the distributed
# training code, you may use any dataset and model you wish and the
# process is the same. Lets get started by first handling the imports and
# talking about some helper functions. Then we will define the train and
# test functions, which have been largely taken from the ImageNet Example.
# At the end, we will build the main part of the code which handles the
# distributed training setup. And finally, we will discuss how to actually
# run the code.
#


######################################################################
# Imports
# ~~~~~~~
#
# The important distributed training specific imports here are
# `torch.nn.parallel <https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel>`__,
# `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`__,
# `torch.utils.data.distributed <https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler>`__,
# and
# `torch.multiprocessing <https://pytorch.org/docs/stable/multiprocessing.html>`__.
# It is also important to set the multiprocessing start method to *spawn*
# or *forkserver* (only supported in Python 3),
# as the default is *fork* which may cause deadlocks when using multiple
# worker processes for dataloading.
#

import time
import sys
import torch
import os
import socket
import requests

import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from pathlib import Path

from torch.multiprocessing import Pool, Process
from torch.nn.parallel import DistributedDataParallel as DDP



######################################################################
# Helper Functions
# ~~~~~~~~~~~~~~~~
#
# We must also define some helper functions and classes that will make
# training easier. The ``AverageMeter`` class tracks training statistics
# like accuracy and iteration count. The ``accuracy`` function computes
# and returns the top-k accuracy of the model so we can track learning
# progress. Both are provided for training convenience but neither are
# distributed training specific.
#

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


######################################################################
# Train Functions
# ~~~~~~~~~~~~~~~
#
# To simplify the main loop, it is best to separate a training epoch step
# into a function called ``train``. This function trains the input model
# for one epoch of the *train\_loader*. The only distributed training
# artifact in this function is setting the
# `non\_blocking <https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers>`__
# attributes of the data and label tensors to ``True`` before the forward
# pass. This allows asynchronous GPU copies of the data meaning transfers
# can be overlapped with computation. This function also outputs training
# statistics along the way so we can track progress throughout the epoch.
#
# The other function to define here is ``adjust_learning_rate``, which
# decays the initial learning rate at a fixed schedule. This is another
# boilerplate trainer function that is useful to train accurate models.
#

def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    batch_begin_at = time.time()
    for i, (input, target) in enumerate(train_loader):
        batch_start_at = time.time()
        batch_prepare_time = batch_start_at - batch_begin_at
        # estimate time the train_loader takes
        print(f"batch {i} prepare time {batch_prepare_time}s")

        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        batch_begin_at = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def adjust_learning_rate(initial_lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = initial_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

######################################################################
# Validation Function
# ~~~~~~~~~~~~~~~~~~~
#
# To track generalization performance and simplify the main loop further
# we can also extract the validation step into a function called
# ``validate``. This function runs a full validation step of the input
# model on the input validation dataloader and returns the top-1 accuracy
# of the model on the validation set. Again, you will notice the only
# distributed training feature here is setting ``non_blocking=True`` for
# the training data and labels before they are passed to the model.
#

def validate(val_loader, model, criterion):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


######################################################################
# Inputs
# ~~~~~~
#
# With the helper functions out of the way, now we have reached the
# interesting part. Here is where we will define the inputs for the run.
# Some of the inputs are standard model training inputs such as batch size
# and number of training epochs, and some are specific to our distributed
# training task. The required inputs are:
#
# -  **batch\_size** - batch size for *each* process in the distributed
#    training group. Total batch size across distributed model is
#    batch\_size\*world\_size
#
# -  **workers** - number of worker processes used with the dataloaders in
#    each process
#
# -  **num\_epochs** - total number of epochs to train for
#
# -  **starting\_lr** - starting learning rate for training
#
# -  **world\_size** - number of processes in the distributed training
#    environment
#
# -  **dist\_backend** - backend to use for distributed training
#    communication (i.e. NCCL, Gloo, MPI, etc.). In this tutorial, since
#    we are using several multi-gpu nodes, NCCL is suggested.
#
# -  **dist\_url** - URL to specify the initialization method of the
#    process group. This may contain the IP address and port of the rank0
#    process or be a non-existant file on a shared file system. Here,
#    since we do not have a shared file system this will incorporate the
#    **node0-privateIP** and the port on node0 to use.
#

print("Print out env")
print(os.environ)

print("Collect Inputs...")

# Batch Size for training and testing
batch_size = 32

# Number of epochs to train for
num_epochs = 4

# Starting Learning Rate
starting_lr = 0.1

# Distributed backend type
#dist_backend = 'nccl'
# use gloo for CPU and nccl for GPU
dist_backend = 'gloo'

# Url used to setup distributed training
# we are not using url but using sync server address and port to create process group.
# dist_url = "tcp://172.31.22.234:23456"
#dist_url = "tcp://127.0.0.1:23456"


######################################################################
# Initialize process group
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# One of the most important parts of distributed training in PyTorch is to
# properly setup the process group, which is the **first** step in
# initializing the ``torch.distributed`` package. To do this, we will use
# the ``torch.distributed.init_process_group`` function which takes
# several inputs. First, a *backend* input which specifies the backend to
# use (i.e. NCCL, Gloo, MPI, etc.). An *init\_method* input which is
# either a url containing the address and port of the rank0 machine or a
# path to a non-existant file on the shared file system. Note, to use the
# file init\_method, all machines must have access to the file, similarly
# for the url method, all machines must be able to communicate on the
# network so make sure to configure any firewalls and network settings to
# accomodate. The *init\_process\_group* function also takes *rank* and
# *world\_size* arguments which specify the rank of this process when run
# and the number of processes in the collective, respectively.
# The *init\_method* input can also be "env://". In this case, the address
# and port of the rank0 machine will be read from the following two
# environment variables respectively: MASTER_ADDR, MASTER_PORT.  If *rank*
# and *world\_size* arguments are not specified in the *init\_process\_group*
# function, they both can be read from the following two environment
# variables respectively as well: RANK, WORLD_SIZE.
#
# Another important step, especially when each node has multiple gpus is
# to set the *local\_rank* of this process. For example, if you have two
# nodes, each with 8 GPUs and you wish to train with all of them then
# :math:`world\_size=16` and each node will have a process with local rank
# 0-7. This local\_rank is used to set the device (i.e. which GPU to use)
# for the process and later used to set the device when creating a
# distributed data parallel model. It is also recommended to use NCCL
# backend in this hypothetical environment as NCCL is preferred for
# multi-gpu nodes.
#

def getProcessGroupInfo():
    local_ip = socket.gethostbyname(socket.gethostname())

    trainer_port = '12355'
    if 'TRAINER_PORT' in os.environ and os.getenv('TRAINER_PORT') is not None:
        trainer_port = os.getenv('TRAINER_PORT')
    
    sync_server_address="http://localhost:5000"
    if 'SYNC_SERVER' in os.environ and os.getenv('SYNC_SERVER') is not None:
        sync_server_address = os.getenv('SYNC_SERVER')

    world_size = os.getenv('WORLD_SIZE')

    group_id = os.getenv('GROUP_ID')

    print(f"send registration info from host, local ip: {local_ip}, trainer port: {trainer_port}, sync server: {sync_server_address}, world size:{world_size}, group_id:{group_id}.")
    # send sync request to sync server.

    # Step 1: register from host A.
    hostSyncRequest = {
        'address': local_ip,
        'port': trainer_port,
        'world': world_size,
        'groupId': group_id
    }

    response = requests.post(sync_server_address + '/register', json=hostSyncRequest)

    print("Status code: ", response.status_code)
    print("Printing Entire Post Request")
    print(response.json())

    # TODO, implement with request and test.
    return response.json()['globalRank'], world_size, response.json()['leadServerAddress'], response.json()['leadServerPort']

# Number of distributed processes
rank, world_size, master_address, master_port = getProcessGroupInfo()

world_size = int(world_size)
# Number of additional worker processes for dataloading
workers = world_size

print("Initialize Process Group...")
# Initialize Process Group
# v1 - init with url
# dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=int(sys.argv[1]), world_size=world_size)
#dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=int(sys.argv[1]), world_size=int(sys.argv[2]))
# v2 - init with file
# dist.init_process_group(backend="nccl", init_method="file:///home/ubuntu/pt-distributed-tutorial/trainfile", rank=int(sys.argv[1]), world_size=world_size)
# v3 - init with environment variables
# dist.init_process_group(backend="nccl", init_method="env://", rank=int(sys.argv[1]), world_size=world_size)

# set env for init process group.
os.environ['MASTER_ADDR'] = master_address
os.environ['MASTER_PORT'] = master_port

data_path = "./data"
if 'DATA_PATH' in os.environ and os.getenv('DATA_PATH') is not None:
    data_path = os.getenv('DATA_PATH')
    Path(data_path).mkdir(parents=True, exist_ok=True)

model_path = "model.ts"
if 'MODEL_PATH' in os.environ and os.getenv('MODEL_PATH') is not None:
    model_path = os.getenv('MODEL_PATH') 
    Path(model_path).mkdir(parents=True, exist_ok=True)
    model_path = model_path + "model.ts"

print(f"Initialize process group, world_size={world_size}, process rank={rank}, master address={master_address}:{master_port}, data:{data_path}, model:{model_path}")


dist.init_process_group(backend=dist_backend, init_method="env://", rank=rank, world_size=world_size)


# Establish Local Rank and set device on this node
# local_rank = int(sys.argv[2])
# dp_device_ids = [local_rank]
# torch.cuda.set_device(local_rank)


######################################################################
# Initialize Model
# ~~~~~~~~~~~~~~~~
#
# The next major step is to initialize the model to be trained. Here, we
# will use a resnet18 model from ``torchvision.models`` but any model may
# be used. First, we initialize the model and place it in GPU memory.
# Next, we make the model ``DistributedDataParallel``, which handles the
# distribution of the data to and from the model and is critical for
# distributed training. The ``DistributedDataParallel`` module also
# handles the averaging of gradients across the world, so we do not have
# to explicitly average the gradients in the training step.
#
# It is important to note that this is a blocking function, meaning
# program execution will wait at this function until *world\_size*
# processes have joined the process group. Also, notice we pass our device
# ids list as a parameter which contains the local rank (i.e. GPU) we are
# using. Finally, we specify the loss function and optimizer to train the
# model with.
#

print("Initialize Model...")
# Construct Model
#model = models.resnet18(pretrained=False).cuda()
model = models.resnet18(pretrained=False)
# Make model DistributedDataParallel
# model = torch.nn.parallel.DistributedDataParallel(model, device_ids=dp_device_ids, output_device=local_rank)
# device_ids must be empty for CPU or single GPU machine.
ddp_model = DDP(model)

# define loss function (criterion) and optimizer
#criterion = nn.CrossEntropyLoss().cuda()
# Avoid GPU.
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(ddp_model.parameters(), starting_lr, momentum=0.9, weight_decay=1e-4)


######################################################################
# Initialize Dataloaders
# ~~~~~~~~~~~~~~~~~~~~~~
#
# The last step in preparation for the training is to specify which
# dataset to use. Here we use the `STL-10
# dataset <https://cs.stanford.edu/~acoates/stl10/>`__ from
# `torchvision.datasets.STL10 <https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.STL10>`__.
# The STL10 dataset is a 10 class dataset of 96x96px color images. For use
# with our model, we resize the images to 224x224px in the transform. One
# distributed training specific item in this section is the use of the
# ``DistributedSampler`` for the training set, which is designed to be
# used in conjunction with ``DistributedDataParallel`` models. This object
# handles the partitioning of the dataset across the distributed
# environment so that not all models are training on the same subset of
# data, which would be counterproductive. Finally, we create the
# ``DataLoader``'s which are responsible for feeding the data to the
# processes.
#
# The STL-10 dataset will automatically download on the nodes if they are
# not present. If you wish to use your own dataset you should download the
# data, write your own dataset handler, and construct a dataloader for
# your dataset here.
#

print("Initialize Dataloaders...")
# Define the transform for the data. Notice, we must resize to 224x224 with this dataset and model.
transform = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Initialize Datasets. STL10 will automatically download if not present
trainset = datasets.STL10(root=data_path, split='train', download=True, transform=transform)
valset = datasets.STL10(root=data_path, split='test', download=True, transform=transform)

# Create DistributedSampler to handle distributing the dataset across nodes when training
# This can only be called after torch.distributed.init_process_group is called
# Distributed Sampler will query distribute group to figure out rank and world_size.
train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

# Create the Dataloaders to feed data to the training and validation steps
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=workers, pin_memory=False, sampler=train_sampler)
val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=False)


######################################################################
# Training Loop
# ~~~~~~~~~~~~~
#
# The last step is to define the training loop. We have already done most
# of the work for setting up the distributed training so this is not
# distributed training specific. The only detail is setting the current
# epoch count in the ``DistributedSampler``, as the sampler shuffles the
# data going to each process deterministically based on epoch. After
# updating the sampler, the loop runs a full training epoch, runs a full
# validation step then prints the performance of the current model against
# the best performing model so far. After training for num\_epochs, the
# loop exits and the tutorial is complete. Notice, since this is an
# exercise we are not saving models but one may wish to keep track of the
# best performing model then save it at the end of training (see
# `here <https://github.com/pytorch/examples/blob/master/imagenet/main.py#L184>`__).
#

best_prec1 = 0

start_time = time.time()
print(f"training start at {start_time}")

for epoch in range(num_epochs):
    # Set epoch count for DistributedSampler
    train_sampler.set_epoch(epoch)

    # Adjust learning rate according to schedule
    adjust_learning_rate(starting_lr, optimizer, epoch)

    # train for one epoch
    print("\nBegin Training Epoch {}".format(epoch+1))
    train(train_loader, ddp_model, criterion, optimizer, epoch)

    # evaluate on validation set
    print("Begin Validation @ Epoch {}".format(epoch+1))
    prec1 = validate(val_loader, ddp_model, criterion)

    # remember best prec@1 and save checkpoint if desired
    # is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    print("Epoch Summary: ")
    print("\tEpoch Accuracy: {}".format(prec1))
    print("\tBest Accuracy: {}".format(best_prec1))

end_time = time.time()
duration_in_seconds = end_time - start_time
print(f"total training time: {duration_in_seconds}s")

# save model by the rank0 process.
if rank == 0:
    print(f"save model at {model_path}")
    torch.save(model.state_dict(), model_path)

group_id = os.getenv('GROUP_ID')
print(f"training {group_id} finished.")

try:
    input("Press Enter to exit...")
except EOFError as e:
    print(e)

######################################################################
# Running the Code with sync server
# ----------------
#   1. set up sync server: docker run -d -p 5000:8080 alex1005/dist_syncserver:1.0 sync_server
#   2. set below environment variables.
#   3. run instances == world_size, they will figure out rank and master server by communicating with sync_server.
#
#   export TRAINER_PORT=12355
#   export WORLD_SIZE=2
#   export GROUP_ID=pytorch1
#   export SYNC_SERVER=http://localhost:5000/
#   export DATA_PATH=/home/chi/workspace/ai_basics/pytorch/data
#   export MODEL_PATH=models
#
