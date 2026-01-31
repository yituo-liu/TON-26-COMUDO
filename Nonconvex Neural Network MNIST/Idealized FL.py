import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU, Softmax, CrossEntropyLoss
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.models import alexnet
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import random
import pickle
from tqdm import tqdm, trange
from collections import defaultdict
from scipy.io import loadmat

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def generate_WeightMatrix():
    P = np.zeros((N, N))
    for i in range(N - 1):
        P[i + 1, i] = 1 / N
        P[i, i + 1] = 1 / N

    for i in range(N):
        for j in range(i + 1, N):
            if np.random.rand() <= 0.4:
                P[i, j] = 1 / N
                P[j, i] = 1 / N
        P[i, i] = 1 - np.sum(P[i, :])
    return P

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_tranform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = MNIST(root='./data', train=True, download=True, transform=dataset_tranform)
test_set = MNIST(root='./data', train=False, download=True, transform=dataset_tranform)


def preload_data(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, targets = next(iter(loader))
    return data.to(device), targets.to(device)

train_images_gpu, train_targets_gpu = preload_data(train_set)
test_images_gpu, test_targets_gpu = preload_data(test_set)

test_subset_indices = list(range(1000))
test_images_subset = test_images_gpu[test_subset_indices]
test_targets_subset = test_targets_gpu[test_subset_indices]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Sequential(
            Conv2d(1, 10, 7),
            ReLU(),
            Flatten(),
            Linear(4840, 10),
            Softmax(dim=1)
        )
    def forward(self, x):
        x = self.model1(x)
        return x

T = 1000
N = 15

weight_matrix = []
for t in range(T):
    weight_matrix.append(generate_WeightMatrix())

label_indices = defaultdict(list)
for idx, (_, label) in enumerate(train_set):
    label_indices[label].append(idx)

D = 32
b = 0.2
num_labels = 10
main_labels = [random.randint(0, num_labels-1) for _ in range(N)]
device_main_labels = {i: main_labels[i] for i in range(N)}

ind_n = {t: {n: [] for n in range(N)} for t in range(T)}
for t in range(T):
    if t % 100 == 0:
        print(f"Time slot {t}")
    for i in range(N):
        selected_indices = []
        main_label = device_main_labels[i]

        num_main_samples = int(b * D)
        main_samples = np.random.choice(label_indices[main_label], num_main_samples, replace=False)
        selected_indices.extend(main_samples)

        num_other_samples = D - num_main_samples
        other_labels = [label for label in np.arange(num_labels) if label != main_label]
        other_samples = np.random.choice(np.concatenate([label_indices[label] for label in other_labels]), num_other_samples, replace=False)
        selected_indices.extend(other_samples)
        
        ind_n[t][i] = selected_indices

local_models = [Net().to(device) for i in range(N)]
models_agg = [copy.deepcopy(model.state_dict()) for model in local_models]

loss = CrossEntropyLoss()
loss.to(device)

local_optimizers = []

for i in range(N):
    optimizer = torch.optim.SGD(local_models[i].parameters(), lr=0.5)
    local_optimizers.append(optimizer)

plot_accuracy = []
plot_loss = []

for t in range(T):
    transmit_power = 0
    weights_local = []
    loss_local = []

    for i in range(N):
        local_models[i].train()
        indices = ind_n[t][i]
        images = []
        targets = []
        sampled_images = train_images_gpu[indices]
        sampled_targets = train_targets_gpu[indices]

        outputs = local_models[i](sampled_images)
        result_loss = loss(outputs, sampled_targets)
        loss_local.append(result_loss.item())

        local_optimizers[i].zero_grad()
        result_loss.backward()
        with torch.no_grad():
            for param_tensor, param in local_models[i].named_parameters():
                if param.requires_grad and param.grad is not None:
                    param.data.copy_(models_agg[i][param_tensor] - param.grad * 0.5)

    current_weights = weight_matrix[t]
    temp_trained_states = [copy.deepcopy(model.state_dict()) for model in local_models]
    for i in range(N):
        neighbors = np.where(current_weights[i] > 0)[0]
        
        state_dict = models_agg[i]
        with torch.no_grad():
            for key in state_dict.keys():
                new_param = torch.zeros_like(state_dict[key])
                for j in neighbors:
                    w = current_weights[i][j]
                    new_param.add_(temp_trained_states[j][key], alpha=float(w))
                state_dict[key].copy_(new_param)

    if (t+1) % 1 == 0:
        total_correct = 0
        with torch.no_grad():
            for i in range(N):
                local_models[i].eval()
                outputs = local_models[i](test_images_subset)
                pred = outputs.argmax(dim=1)
                total_correct += (pred == test_targets_subset).sum().item()
        if t == 0:
            plot_accuracy.append(total_correct / 15000 * 100)
        else:
            plot_accuracy.append((plot_accuracy[t-1] * t + total_correct / 15000 * 100) / (t+1))
        print("t: {} Accuracy: {}".format(t+1, plot_accuracy[t]))