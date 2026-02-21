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
        torch.backends.cudnn.deterministic = True

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

T = 1000
N = 15

weight_matrix = []
for t in range(T):
    weight_matrix.append(generate_WeightMatrix())

dataset_tranform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = MNIST(root='./data', train=True, download=True, transform=dataset_tranform)
test_set = MNIST(root='./data', train=False, download=True, transform=dataset_tranform)

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

dist = np.random.uniform(80, 120, size=(N, N))
distances = (dist + dist.T) / 2
gamma = 33.0
psi = 8.0
Halpha = torch.tensor(0.997).cuda()

def compute_epsilon(distance):
    db_values = -31.54 - 33 * np.log10(distance) - np.sqrt(8) * np.random.randn()
    epsilon_com = 10 ** (db_values / 10)
    return epsilon_com

epsilon_device = [[compute_epsilon(distances[i][j])
                   for j in range(N)]
                  for i in range(N)]

h = [
    [{
        'model1.0.weight': torch.randn(10, 1, 7, 7, dtype=torch.complex32)*np.sqrt(epsilon_device[i][j]/2),
        'model1.0.bias': torch.randn(10, dtype=torch.complex32)*np.sqrt(epsilon_device[i][j]/2),
        'model1.3.weight': torch.randn(10, 4840, dtype=torch.complex32)*np.sqrt(epsilon_device[i][j]/2),
        'model1.3.bias': torch.randn(10, dtype=torch.complex32)*np.sqrt(epsilon_device[i][j]/2),
    }
    for j in range(N)]
    for i in range(N)
]

h_gpu = []
h_gpu = [
    [{k: v.to(device) for k, v in h[i][j].items()} for j in range(N)]
    for i in range(N)
]

N0 = -174
NF = 10
BW = 15e3
noise_dBm = N0 + 10 * np.log10(BW) + NF
noise = 10 ** (noise_dBm/10) * 1e-3

Z = [{
        'model1.0.weight': torch.randn(10, 1, 7, 7, dtype=torch.complex32)*np.sqrt(noise),
        'model1.0.bias': torch.randn(10, dtype=torch.complex32)*np.sqrt(noise),
        'model1.3.weight': torch.randn(10, 4840, dtype=torch.complex32)*np.sqrt(noise),
        'model1.3.bias': torch.randn(10, dtype=torch.complex32)*np.sqrt(noise),
    }
    for _ in range(N)
]
Z_gpu = [{k: v.to(device) for k, v in z.items()} for z in Z]

epsilon_device = torch.tensor(epsilon_device, dtype=torch.float32).to(device)

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

Lambda = torch.tensor(1e-5, device=device)
Gamma = torch.tensor(7e-5, device=device)
Qtn = torch.ones(N, device=device)
Gtn = torch.zeros(N, device=device)
Pnbar = torch.tensor((10 ** ((18-30)/10)) * 48910/500, device=device)
Q_Low = torch.tensor(0.0, device=device)

transmit_power_history = [0]
transmit_power_history_dBm = [0]
Power_Separate = np.zeros((T+1, N))

g_gradient = {
        'model1.0.weight': torch.zeros(10, 1, 7, 7, device=device),
        'model1.0.bias': torch.zeros(10, device=device),
        'model1.3.weight': torch.zeros(10, 4840, device=device),
        'model1.3.bias': torch.zeros(10, device=device),
    }

for t in range(T-1):
    transmit_power = 0
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
        
        previous_weights = weight_matrix[t]
        neighbors = np.where((previous_weights[:,i] > 0) & (np.arange(N) != i))[0]
        with torch.no_grad():
            for param_tensor, param in local_models[i].named_parameters():
                g_gradient[param_tensor] = torch.tensor(torch.zeros_like(param), dtype=torch.float32, device=device)
                for j in neighbors:
                    if param.grad is not None:
                        h_data = h_gpu[j][i][param_tensor]
                        h_abs_inverse = 1.0 / torch.abs(h_data)
                        g_gradient[param_tensor] += 2 * (Lambda * (1/N) * h_abs_inverse) ** 2
        with torch.no_grad():
            for param_tensor, param in local_models[i].named_parameters():
                if param.grad is not None:
                    param.data.copy_((models_agg[i][param_tensor] - 0.5 * param.grad) /(1 + 0.5 * Gamma * Qtn[i] * g_gradient[param_tensor]))
    
        gn = torch.tensor(0, device=device)
        state_dict = local_models[i].state_dict()
        current_weights_n = weight_matrix[t+1]
        neighbors = np.where((current_weights_n[:,i] > 0) & (np.arange(N) != i))[0]
        for param_tensor in state_dict:
            for j in range(N):
                param_data = state_dict[param_tensor]
                h_data = h_gpu[j][i][param_tensor]

                Hz = torch.sqrt(epsilon_device[j][i] * (1 - Halpha ** 2) / 2) * torch.randn(size=h_data.shape, dtype=torch.complex64).to(device)
                h_gpu[j][i][param_tensor] = Halpha * h_data + Hz

                if j in neighbors:
                    h_abs_inverse = 1.0 / torch.abs(h_gpu[j][i][param_tensor])
                    power_compute = torch.sum((Lambda * (1/N) * param_data * h_abs_inverse) ** 2)
                    power = power_compute.item()
                    transmit_power = transmit_power + power
                    gn = gn + power_compute
        cons = gn - Pnbar
        Qtn[i] = torch.maximum(Qtn[i] + cons, Q_Low)
        Power_Separate[t+1][i] = gn

    tranPower_toHistory = (transmit_power_history[t] * (t+1) + transmit_power/(N*48910/500))/(t+2)
    transmit_power_history.append(tranPower_toHistory)
    transmit_power_history_dBm.append(np.log10(transmit_power_history[t+1]*1000) * 10)

    for param_tensor in Z_gpu[i]:
        Z_gpu[i][param_tensor] = torch.randn(size=Z_gpu[i][param_tensor].shape, device=device) * torch.sqrt(torch.tensor(noise, device=device))

    current_weights = weight_matrix[t+1]
    temp_trained_states = [{k: v.clone() for k,v in model.state_dict().items()} for model in local_models]
    for i in range(N):
        neighbors = np.where(current_weights[i] > 0)[0]
        
        state_dict = models_agg[i]
        with torch.no_grad():
            for key in state_dict.keys():
                new_param = torch.zeros_like(state_dict[key])
                for j in neighbors:
                    w = current_weights[i][j]

                    new_param.add_(temp_trained_states[j][key], alpha=float(w))
                state_dict[key].copy_(new_param + torch.div(Z_gpu[i][key], Lambda))

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

        print("t: {} Accuracy: {}, Power: {}, VQ: {}".format(t+1, plot_accuracy[t], transmit_power_history_dBm[t+1], torch.mean(Qtn).item()))

