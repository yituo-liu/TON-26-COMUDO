import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import random
import math
import pickle
from collections import defaultdict

rho = 0.4
def generate_WeightMatrix():
    P = np.zeros((N, N))
    for i in range(N - 1):
        P[i + 1, i] = 1 / N
        P[i, i + 1] = 1 / N

    for i in range(N):
        for j in range(i + 1, N):
            if np.random.rand() <= rho:
                P[i, j] = 1 / N
                P[j, i] = 1 / N
        P[i, i] = 1 - np.sum(P[i, :])
    return P

def Ideal(N, total_step, D, d, C, Data, Label):

    xtn = {t: np.zeros((N, d * C)) for t in range(0, total_step)}
    ytn = {t: np.zeros((N, d * C)) for t in range(0, total_step)}

    for step in range(1, total_step):
        alpha = 1e5
        GD_tnc_f = np.zeros((N, d * C))
        for n in range(N):
            for i in range(D):
                d_tni = Data[step - 1][n][i]
                b_tni = Label[step - 1][n][i]
                p_tni = np.zeros(C)
                for k in range(C):
                    idx = slice(d * k, d * (k + 1))
                    p_tni[k] = np.exp(d_tni @ xtn[step - 1][n][idx])
                hsum_tn = np.sum(p_tni)

                for c in range(0, C):
                    idx = slice(d * c, d * (c + 1))
                    GD_tnc_f[n][idx] -= (1 / D) * ((b_tni == c) - p_tni[c] / hsum_tn) * d_tni

            xtn[step][n] = ytn[step - 1][n] - 1 / (2 * alpha) * GD_tnc_f[n]
        
        ytn[step] = np.dot(WeightMatrix[step], xtn[step])
        if step % 50 == 0:
            print(f'Ideal, t:{step}')

    Accuracy = []
    for t in range(total_step):
        wrong = 0
        for n in range(N):
            for i in range(1000):
                d_i = Testdata[:, i]
                h_ti = np.zeros(10)
                for k in range(10):
                    idx = slice(d * k, d * (k + 1))
                    h_ti[k] = np.exp(d_i @ xtn[t][n][idx])
                hsum_ti = np.sum(h_ti)
                TorF = (h_ti / hsum_ti).argmax()
                if Testlabels[i] != TorF:
                    wrong += 1
        At = (1 - wrong / 1000 / N) * 100
        if t == 0:
            Accuracy.append(At)
        else:
            Accuracy.append((Accuracy[t - 1] * t + At) / (t + 1))
        print(f'Accuracy: {Accuracy[t]}, t:{t + 1}')

    return Accuracy


total_step = 1000
N = 10
D = 32
d = 784
C = 10 

WeightMatrix = []
for t in range(total_step):
    WeightMatrix.append(generate_WeightMatrix())

dataset_tranform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

train_set = torchvision.datasets.MNIST(root="./dataset", train=True, transform=dataset_tranform, download=True)
test_set = torchvision.datasets.MNIST(root="./dataset", train=False, transform=dataset_tranform, download=True)

Data = np.zeros((total_step, N, D, 784))
Label = np.zeros((total_step, N, D))

label_indices = defaultdict(list)
for idx, (_, label) in enumerate(train_set):
    label_indices[label].append(idx)

b = 0.2
num_labels = 10
main_labels = [random.randint(0, num_labels-1) for _ in range(N)]
device_main_labels = {i: main_labels[i] for i in range(N)}

ind_n = {t: {n: [] for n in range(N)} for t in range(total_step)}
for t in range(total_step):
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

for t in range(0, total_step):
    for n in range(N):
        Sample =  ind_n[t][n]
        for i, idx in enumerate(Sample):
            img, tar = train_set[idx]
            Data[t][n][i] = img.view(-1).numpy()
            Label[t][n][i] = tar

test_samples = []
test_labels = []
for i in range(1000):
    img, label = test_set[i]
    test_samples.append(img.view(-1).numpy())
    test_labels.append(label)

Testdata = np.array(test_samples).T
Testlabels = np.array(test_labels)


Acc_Ideal = Ideal(N, total_step, D, d, C, Data, Label)
