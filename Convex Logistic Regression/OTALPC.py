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

def OTALPC(N, total_step, D, d, C, Data, Label, H, Z):
    slot = 7840 / 500
    lamda = np.zeros((N, N))
    Xi = np.zeros((N,))

    xtn = {t: np.zeros((N, d * C)) for t in range(0, total_step)}
    ytn = {t: np.zeros((N, d * C)) for t in range(0, total_step)}

    Pn_bar = 10 ** ((14 - 30) / 10) * slot

    Power_Summary = [0]
    Power_Summary_dbm = [0]

    for step in range(1, total_step):
        alpha = 1e5
        Reg = 1e2
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
                    GD_tnc_f[n][idx] -= (1 / D) * ((b_tni == c) - p_tni[c] / hsum_tn) * d_tni  + 2 * Reg * xtn[step - 1][n][idx]

            xtn[step][n] = ytn[step - 1][n] - 1 / (2 * alpha) * GD_tnc_f[n]

        Hz = scale_innov * (np.random.randn(N, N, 7840) + 1j * np.random.randn(N, N, 7840))
        H = Kappa * H + Hz
        for n in range(N):
            ratio = 0
            w = WeightMatrix[step][:, n]
            for k in range(N):
                if w[k] !=0 and k != n:
                    term = w[k] * (1.0 / np.abs(H[k, n, :])) * xtn[step][n]
                    ratio += np.linalg.norm(term)**2

            E = np.sqrt(Pn_bar / ratio)
            lamda[:,n] = E * w * N
            lamda[n,n] = 0
        for n in range(N):
            Xi[n] = np.sum(lamda[n,:]) / np.count_nonzero(lamda[n, :])
        Power = 0
        for n in range(N):
            w = WeightMatrix[step][:, n]
            for k in range(N):
                if w[k] !=0 and k != n:
                    H_tnc = H[k, n, :]            
                    e_tnc = 1.0 / np.abs(H_tnc)        

                    term = (w[k] * lamda[k,n]) * e_tnc * xtn[step][n]
                    Power_n = np.linalg.norm(term)**2
                    Power += Power_n

        power = (Power_Summary[step - 1] * step + Power / (N * slot))  / (step + 1)
        Power_Summary.append(power)
        Power_Summary_dbm.append(10 * np.log10(Power_Summary[step] * 1000))
        
        Z = np.sqrt(noise) * np.random.randn(N, 7840)
        for n in range(N):
            if WeightMatrix[step][n, n] !=1:
                for k in range(N):
                    if WeightMatrix[step][n, k] !=0 and k != n:
                        ytn[step][n] = ytn[step][n] +  WeightMatrix[step][n, k] * xtn[step][k] * lamda[n,k]
                ytn[step][n] = ytn[step][n] / Xi[n] + WeightMatrix[step][n, n] * xtn[step][n] + Z[n] / Xi[n]
            else:
                ytn[step][n] = xtn[step][n]
        if step % 50 == 0:
            print(f'OTALPC, t:{step}')

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

    return Accuracy, Power_Summary_dbm


total_step = 1000
N = 10
D = 32
d = 784
C = 10 

WeightMatrix = []
for t in range(total_step):
    WeightMatrix.append(generate_WeightMatrix())

Distance = np.random.uniform(80, 120, size=(N, N))
Distance = (Distance + Distance.T) // 2

psi = 8
PathLoss_dB = -31.54 - 33 * np.log10(np.abs(Distance)) - np.sqrt(psi) * np.random.randn(N, N)
PathLoss = 10 ** (PathLoss_dB / 10)
epsilon_device = PathLoss

BW = 15e3 
NF = 10

noise_dBm = -174 + 10 * np.log10(BW) + NF
noise = 10**((noise_dBm - 30) / 10)

Kappa = 0.997
H = np.zeros((N, N, 7840), dtype=complex)
Z = np.zeros((N, 7840), dtype=float)
scale_init = np.sqrt(epsilon_device[:, :, None] / 2.0) 
scale_innov = np.sqrt((1.0 - Kappa**2) * epsilon_device[:, :, None] / 2.0)

H = scale_init * (np.random.randn(N, N, 7840) + 1j * np.random.randn(N, N, 7840))
Z = np.sqrt(noise) * np.random.randn(N, 7840)

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


Acc_OTALPC, Power_OTALPC = OTALPC(N, total_step, D, d, C, Data, Label, H, Z)
