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

def COMUDO(N, total_step, D, d, C, Data, Label, H, Z):
    slot = 7840 / 500

    lamda = 2e-4
    gamma = 6e-1
    eta = 1e-3
    V = 2

    xtn = {t: np.zeros((N, d * C)) for t in range(0, total_step)}
    ytn = {t: np.zeros((N, d * C)) for t in range(0, total_step)}
    gtn = {t: np.zeros((N,)) for t in range(0, total_step)}

    Qtn = {t: np.ones((N,)) * V for t in range(0, total_step)}

    Pn_bar = 10 ** ((14 - 30) / 10) * slot

    Power_Vio_Summary = [0]
    Power_Vio_Summary_dbm = [0]

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

            H_tnc = H[:, n, :]
            e_tnc = 1.0 / np.abs(H_tnc)
            w = WeightMatrix[step - 1][:, n]
            w[n] = 0.0
            g_Power_test = np.linalg.norm((w[:, None] * lamda) * e_tnc * (xtn[step][n][None, :]))**2 - Pn_bar
                
            if g_Power_test > 0:
                H_tnc_abs = np.abs(H_tnc)**2 
                mask = np.arange(N) != n
                w_masked = w[mask]
                H_tnc_masked = H_tnc_abs[mask]

                temp = np.sum(w_masked[:, None]**2 / H_tnc_masked, axis=0)
                GD_g = 2 * (lamda**2) * temp 

                xtn[step][n] =(ytn[step - 1][n] - 1 / (2 * alpha) * GD_tnc_f[n]) / (1 + 1 / (2 * alpha) * Qtn[step - 1][n] * gamma * GD_g)

        Hz = scale_innov * (np.random.randn(N, N, 7840) + 1j * np.random.randn(N, N, 7840))
        H = Kappa * H + Hz
        Power_Vio = 0
        for n in range(N):
            H_tnc = H[:, n, :]
            e_tnc = 1.0 / np.abs(H_tnc)
            w =  WeightMatrix[step][:, n]
            w[n] = 0.0 

            term = (w[:, None] * lamda) * e_tnc * xtn[step][n][None, :]
            Power_n = np.linalg.norm(term)**2
            Power_Vio += max(Power_n - Pn_bar, 0)

            gtn[step][n] = max(Power_n - Pn_bar, 0)
            Qtn[step][n] = max((1 - eta) * Qtn[step - 1][n] + gamma * gtn[step][n], V)

        viopower = (Power_Vio_Summary[step - 1] * step + Power_Vio / (N * slot) / 10**(-1.6) )  / (step + 1)
        Power_Vio_Summary.append(viopower)
        Power_Vio_Summary_dbm.append(10 * np.log10(Power_Vio_Summary[step]))

        ytn[step] = np.dot(WeightMatrix[step], xtn[step])
        Z = np.sqrt(noise) * np.random.randn(N, 7840)
        for n in range(N):
            ytn[step][n] = ytn[step][n] + Z[n] / lamda
        if step % 50 == 0:
            print(f'COMUDO, t:{step}')

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

    return Accuracy, Power_Vio_Summary_dbm


total_step = 1000
N = 15
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


Acc_COMUDO, Power_Vio_COMUDO = COMUDO(N, total_step, D, d, C, Data, Label, H, Z)

