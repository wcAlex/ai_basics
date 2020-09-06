import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# A simple example to use DataParallel, this distribute library is almost deprecated,
# but I think it still worth the time to read and run, since the code here can demonstrate 
# the distributed training concept clearly and in a extremely simple way, it's good
# for understanding cross machine training - DistributedDataParallel. 

# DataParallel is not recommend to use, although it enables single-machine 
# multi-GPU parallelism with the lowest coding hurdle and it only requires 
# a one-line change to the application code, it usually does not offer the best performance. 

# This is because the implementation of DataParallel replicates the model in every forward pass, and its single-process multi-thread parallelism naturally suffers from GIL contentions. 
# To get better performance, please consider using DistributedDataParallel.
# https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)

class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model = nn.DataParallel(model)

model.to(device)

for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())

