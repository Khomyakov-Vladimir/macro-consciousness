#!/usr/bin/env python3
"""
# scripts/full_consciousness_interactive.py

"Full Adaptive Consciousness Simulation with Weight Dynamics, Correlation, and Interactive 3D Network"

Author: Vladimir Khomyakov
License: MIT
This version corresponds to the publication 
*A Multi-Level Computational Model of Macro-Consciousness with Self-Organizing Inter-Cluster Networks, Predictive Adaptation, and Reproducible Python Simulations* (August 2025). 
Version-specific DOI: 10.5281/zenodo.16937283
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# ==========================
# Parameters
# ==========================
N = 60
K = 5
M = 6
T = 300

# Initial learning parameters
eta_init = 0.1
gamma_init = 0.1
alpha_init = 0.05
beta_init = 0.05

sigma_init = 0.05
env_strength = 0.05
memory_decay = 0.9
task_influence = 0.05
macro_task_weight = 0.3
eta_W = 0.001
W_max = 0.2

# ==========================
# Initialization
# ==========================
p_micro = np.random.rand(N, K)
p_micro /= p_micro.sum(axis=1, keepdims=True)

E_local = np.random.rand(N, K)
cluster_assignment = np.random.randint(0, M, size=N)

W = np.zeros((M, M))
for i in range(M):
    for j in range(M):
        if i != j:
            W[i,j] = sigma_init

C_mem = np.zeros((M, K))
Macro_mem = np.zeros(K)
E_task = 0.5 + 0.5*np.sin(np.linspace(0, 4*np.pi, T))[:,None]*np.random.rand(1,K)

# ==========================
# Histories
# ==========================
entropy_history = []
macro_projection_history = []
clusters_history = []
W_history = np.zeros((T, M, M))
clusters_prev = np.zeros((M,K))
macro_prev = np.zeros(K)

# ==========================
# Helper functions
# ==========================
def entropy(p_i):
    p_i = np.clip(p_i, 1e-12, 1.0)
    return -np.sum(p_i*np.log(p_i))

def aggregate_clusters(p_micro, cluster_assignment, M):
    clusters = np.zeros((M,K))
    for m in range(M):
        members = p_micro[cluster_assignment==m]
        if len(members)>0:
            clusters[m] = np.mean(members, axis=0)
    return clusters

def update_clusters_with_signals_and_memory(clusters, W, alpha, C_mem, memory_decay):
    clusters_new = clusters.copy()
    for i in range(M):
        clusters_new[i] += alpha*(clusters.mean(axis=0) - clusters[i])
        signal = np.sum([W[i,j]*(clusters[j]-clusters[i]) for j in range(M)], axis=0)
        clusters_new[i] += signal
        clusters_new[i] += (1-memory_decay)*(C_mem[i]-clusters[i])
    return clusters_new

def compute_macro_projection(clusters):
    return clusters.mean(axis=0)

def update_environment(E_local, t, T):
    env_dynamic = 0.5 + 0.5*np.sin(2*np.pi*t/T*np.arange(N)[:,None])
    noise = 0.1*np.random.rand(N,K)
    return E_local*(1-env_strength) + env_dynamic*env_strength + noise*env_strength

# ==========================
# Simulation loop
# ==========================
for t in range(T):
    E_local = update_environment(E_local, t, T)
    
    # Micro-level update
    for i in range(N):
        grad_H = np.log(p_micro[i]+1e-12)+1
        grad_L = 2*(p_micro[i]-E_local[i])
        task_signal = task_influence*(E_task[t]-p_micro[i])
        p_micro[i] -= eta_init*grad_H + gamma_init*grad_L - task_signal
        p_micro[i] = np.clip(p_micro[i],0,1)
        p_micro[i] /= p_micro[i].sum()
    
    # Meso-level aggregation
    clusters = aggregate_clusters(p_micro, cluster_assignment, M)
    clusters = update_clusters_with_signals_and_memory(clusters, W, alpha_init, C_mem, memory_decay)
    C_mem = memory_decay*C_mem + (1-memory_decay)*clusters
    
    # Macro-level
    macro_projection = compute_macro_projection(clusters)
    Macro_mem = memory_decay*Macro_mem + (1-memory_decay)*macro_projection
    macro_projection_pred = (1-macro_task_weight)*((macro_projection+Macro_mem)/2) + macro_task_weight*E_task[t]
    
    macro_projection_history.append(macro_projection_pred)
    clusters_history.append(clusters.copy())
    H_global = np.sum([entropy(p_micro[i]) for i in range(N)])
    entropy_history.append(H_global)
    
    # Macro feedback
    for i in range(N):
        feedback = beta_init*(macro_projection_pred - p_micro[i])
        p_micro[i] += feedback
        p_micro[i] = np.clip(p_micro[i],0,1)
        p_micro[i] /= p_micro[i].sum()
    
    # Self-learning inter-cluster weights
    for i in range(M):
        for j in range(M):
            if i != j:
                delta = eta_W*np.dot((clusters[i]-clusters_prev[i]), (macro_projection_pred - macro_prev))
                W[i,j] += delta
                W[i,j] = np.clip(W[i,j],0,W_max)
    
    W_history[t] = W.copy()
    clusters_prev = clusters.copy()
    macro_prev = macro_projection_pred.copy()

# ==========================
# Convert histories
# ==========================
clusters_history = np.array(clusters_history)
macro_projection_history = np.array(macro_projection_history)
W_history = np.array(W_history)

# ==========================
# Correlation history
# ==========================
window = 20
corr_history = np.zeros((T, M, K))
for m in range(M):
    for k in range(K):
        for t in range(window, T):
            cluster_window = clusters_history[t-window:t, m, k]
            macro_window = macro_projection_history[t-window:t, k]
            if np.std(cluster_window) > 1e-8 and np.std(macro_window) > 1e-8:
                corr_history[t, m, k] = np.corrcoef(cluster_window, macro_window)[0,1]

# ==========================
# 3D Interactive Network
# ==========================
pos3d = np.random.rand(M,3) * 10
fig = plt.figure(figsize=(12,9))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

# Sliders for learning parameters
ax_eta = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_alpha = plt.axes([0.25, 0.1, 0.65, 0.03])
ax_beta = plt.axes([0.25, 0.05, 0.65, 0.03])

slider_eta = Slider(ax_eta, 'eta', 0.01, 0.5, valinit=eta_init)
slider_alpha = Slider(ax_alpha, 'alpha', 0.01, 0.5, valinit=alpha_init)
slider_beta = Slider(ax_beta, 'beta', 0.01, 0.5, valinit=beta_init)

def update(frame):
    ax.clear()
    # Update parameters from sliders
    eta = slider_eta.val
    alpha = slider_alpha.val
    beta = slider_beta.val
    
    # Node colors = mean correlation
    node_colors = [np.mean(corr_history[frame, m, :]) for m in range(M)]
    
    # Edge properties
    edge_colors = []
    edge_widths = []
    for i in range(M):
        for j in range(M):
            if i != j:
                edge_colors.append(W_history[frame,i,j])
                edge_widths.append(W_history[frame,i,j]*5 + 0.1)
    
    # Draw edges
    for idx,(i,j) in enumerate([(i,j) for i in range(M) for j in range(M) if i!=j]):
        x = [pos3d[i,0], pos3d[j,0]]
        y = [pos3d[i,1], pos3d[j,1]]
        z = [pos3d[i,2], pos3d[j,2]]
        ax.plot(x, y, z, color=plt.cm.viridis(edge_colors[idx]), linewidth=edge_widths[idx])
    
    # Draw nodes
    ax.scatter(pos3d[:,0], pos3d[:,1], pos3d[:,2], s=200, c=node_colors, cmap='coolwarm', vmin=-1, vmax=1)
    
    ax.set_title(f'Time step {frame} | eta={eta:.2f}, alpha={alpha:.2f}, beta={beta:.2f}')
    ax.set_xlim(0,10); ax.set_ylim(0,10); ax.set_zlim(0,10)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

ani = FuncAnimation(fig, update, frames=T, interval=100)
plt.show()
