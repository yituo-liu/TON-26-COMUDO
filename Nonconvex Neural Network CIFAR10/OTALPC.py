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

dataset_tranform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=dataset_tranform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=dataset_tranform)

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


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class CifarResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarResNet20, self).__init__()
        self.inplanes = 16
        block = BasicBlock
        layers = [3, 3, 3]

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet20(num_classes=10):

    return CifarResNet20(num_classes=num_classes)

initial_model = resnet20(num_classes=10) 

state_dict = torch.load("resnet20_cifar10.pth", map_location='cpu')
initial_model.load_state_dict(state_dict, strict=True)


def create_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if ("layer3.0.conv" in name) or ("layer3.1.conv1" in name):
            param.requires_grad = True

    for name, module in model.named_modules():
        if ("layer3.0.conv" in name) or ("layer3.1.conv1" in name):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    for name, param in model.named_parameters():
        if name.startswith("fc"):
            param.requires_grad = True

    nn.init.kaiming_normal_(model.fc.weight)
    nn.init.constant_(model.fc.bias, 0)

    return model

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
        'layer3.0.conv1.weight': torch.randn(64, 32, 3, 3, dtype=torch.complex32)*np.sqrt(epsilon_device[i][j]/2),
        'layer3.0.conv2.weight': torch.randn(64, 64, 3, 3, dtype=torch.complex32)*np.sqrt(epsilon_device[i][j]/2),
        'layer3.1.conv1.weight': torch.randn(64, 64, 3, 3, dtype=torch.complex32)*np.sqrt(epsilon_device[i][j]/2),
        'fc.weight': torch.randn(10, 64, dtype=torch.complex32)*np.sqrt(epsilon_device[i][j]/2),
        'fc.bias': torch.randn(10, dtype=torch.complex32)*np.sqrt(epsilon_device[i][j]/2),
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
        'layer3.0.conv1.weight': torch.randn(64, 32, 3, 3, dtype=torch.complex32)*np.sqrt(noise),
        'layer3.0.conv2.weight': torch.randn(64, 64, 3, 3, dtype=torch.complex32)*np.sqrt(noise),
        'layer3.1.conv1.weight': torch.randn(64, 64, 3, 3, dtype=torch.complex32)*np.sqrt(noise),
        'fc.weight': torch.randn(10, 64, dtype=torch.complex32)*np.sqrt(noise),
        'fc.bias': torch.randn(10, dtype=torch.complex32)*np.sqrt(noise),
    }
    for _ in range(N)
]
Z_gpu = [{k: v.to(device) for k, v in z.items()} for z in Z]

epsilon_device = torch.tensor(epsilon_device, dtype=torch.float32).to(device)



local_models = [copy.deepcopy(initial_model).to(device) for i in range(N)]
local_models = [create_model(model) for model in local_models]

models_agg = [
    {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad} 
    for m in local_models
]

loss = CrossEntropyLoss()
loss.to(device)

local_optimizers = []

for i in range(N):
    optimizer = torch.optim.SGD(local_models[i].parameters(), lr=0.05)
    local_optimizers.append(optimizer)

weight_matrix_gpu = torch.tensor(weight_matrix, dtype=torch.float32).to(device)

plot_accuracy = []
plot_loss = []

Pnbar = torch.tensor((10 ** ((18-30)/10)) * 92810/500, device=device)

Kappa = torch.tensor(1e-3, device=device)

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
    Lambda = torch.zeros((N,N), device=device)
    E = torch.zeros(N, device=device)
    Xi = torch.zeros(N, device=device)

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
                    param.data.copy_(models_agg[i][param_tensor] - (param.grad + 2 * Kappa * param.data)* 0.05)

    current_weights_matrix = weight_matrix_gpu[t+1]
    ratio = torch.zeros(N, device=device)
    for i in range(N):
        neighbors = torch.where((current_weights_matrix[:,i] > 0) & (torch.arange(N, device=device) != i))[0]
        for param_tensor, param in local_models[i].named_parameters():
            if not param.requires_grad:
                continue
            param_data = param.data
            for j in range(N):
                h_data = h_gpu[j][i][param_tensor]
                Hz = torch.sqrt(epsilon_device[j][i] * (1 - Halpha ** 2) / 2) * torch.randn(size=h_data.shape, dtype=torch.complex64).to(device)
                h_gpu[j][i][param_tensor] = Halpha * h_data + Hz

                if j in neighbors:
                    h_abs_inverse = 1.0 / torch.abs(h_gpu[j][i][param_tensor])
                    ratio[i] += torch.sum(((1/N) * param_data * h_abs_inverse) ** 2)
        E[i] = torch.sqrt(Pnbar / ratio[i])
        Lambda[:,i] = E[i] * current_weights_matrix[:,i] * N
        Lambda[i,i] = torch.tensor(0.0, device=device)

    for i in range(N):
        state_dict = local_models[i].state_dict()
        neighbors = torch.where((current_weights_matrix[:,i] > 0) & (torch.arange(N, device=device) != i))[0]
        Xi[i] = torch.sum(Lambda[i,:]) / torch.maximum(torch.count_nonzero(Lambda[i,:]) - 1, torch.tensor(1, device=device))
        transmit_power += E[i] ** 2 * ratio[i]

    tranPower_toHistory = (transmit_power_history[t] * (t+1) + transmit_power.item()/(N*92810/500))/(t+2)
    transmit_power_history.append(tranPower_toHistory)
    transmit_power_history_dBm.append(np.log10(transmit_power_history[t+1]*1000) * 10)

    for param_tensor in Z_gpu[i]:
        Z_gpu[i][param_tensor] = torch.randn(size=Z_gpu[i][param_tensor].shape, device=device) * torch.sqrt(torch.tensor(noise, device=device))

    temp_trained_states = [{k: v.clone() for k,v in model.state_dict().items()} for model in local_models]
    for i in range(N):
        neighbors = torch.where(current_weights_matrix[i,:] > 0)[0]
        
        state_dict = models_agg[i]
        with torch.no_grad():
            for key in state_dict.keys():
                new_param = torch.zeros_like(state_dict[key])
                for j in neighbors:
                    w = current_weights_matrix[i][j]
                    if j == i:
                        new_param.add_(temp_trained_states[j][key], alpha=float(w))
                    else:
                        new_param.add_(temp_trained_states[j][key], alpha=float(w * Lambda[i][j] / Xi[i]))
                state_dict[key].copy_(new_param + torch.div(Z_gpu[i][key], Xi[i]))

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
        print("t: {} Accuracy: {}, Power: {}".format(t+1, plot_accuracy[t], transmit_power_history_dBm[t+1]))
