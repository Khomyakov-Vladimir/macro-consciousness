#!/usr/bin/env python3
"""
# scripts/self_learning_consciousness_with_weight_dynamics.py

"Adaptive Predictive Consciousness with Self-learning Cluster Network and Weight Dynamics"

Author: Vladimir Khomyakov
License: MIT
This version corresponds to the publication 
*A Multi-Level Computational Model of Macro-Consciousness with Self-Organizing Inter-Cluster Networks, Predictive Adaptation, and Reproducible Python Simulations* (August 2025). 
Version-specific DOI: 10.5281/zenodo.16937283
"""

import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Parameters
# ==========================
N = 60               # number of micro-elements
K = 5                # number of states per element
M = 6                # number of clusters
T = 300              # number of time steps
eta = 0.1
gamma = 0.1
alpha = 0.05
beta = 0.05
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
# Histories for visualization
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
        p_micro[i] -= eta*grad_H + gamma*grad_L - task_signal
        p_micro[i] = np.clip(p_micro[i],0,1)
        p_micro[i] /= p_micro[i].sum()
    
    # Meso-level aggregation
    clusters = aggregate_clusters(p_micro, cluster_assignment, M)
    clusters = update_clusters_with_signals_and_memory(clusters, W, alpha, C_mem, memory_decay)
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
        feedback = beta*(macro_projection_pred - p_micro[i])
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
# Visualization
# ==========================
# Global entropy
plt.figure(figsize=(10,5))
plt.plot(entropy_history, label='Global Entropy H(S)')
plt.xlabel('Time step')
plt.ylabel('Entropy')
plt.title('Global Observer Entropy')
plt.grid(True)
plt.legend()
plt.show()

# Macro-consciousness predictive states
plt.figure(figsize=(10,5))
for k in range(K):
    plt.plot(macro_projection_history[:,k], label=f'State {k}')
plt.xlabel('Time step')
plt.ylabel('Macro Projection (Predictive)')
plt.title('Predictive Macro-consciousness Dynamics')
plt.grid(True)
plt.legend()
plt.show()

# Cluster states dynamics
plt.figure(figsize=(12,6))
for m in range(M):
    for k in range(K):
        plt.plot(clusters_history[:,m,k], label=f'Cluster {m} State {k}', linestyle='--', alpha=0.7)
plt.xlabel('Time step')
plt.ylabel('Cluster State')
plt.title('Cluster States Dynamics')
plt.grid(True)
plt.legend(loc='upper right', fontsize=8)
plt.show()

# External tasks
plt.figure(figsize=(10,5))
for k in range(K):
    plt.plot(E_task[:,k], label=f'Task State {k}')
plt.xlabel('Time step')
plt.ylabel('External Task Signal')
plt.title('External Predictive Tasks')
plt.grid(True)
plt.legend()
plt.show()

# Inter-cluster weights dynamics
plt.figure(figsize=(12,6))
for i in range(M):
    for j in range(M):
        if i != j:
            plt.plot(W_history[:,i,j], label=f'W[{i},{j}]')
plt.xlabel('Time step')
plt.ylabel('Inter-cluster Weight')
plt.title('Dynamics of Self-learning Inter-cluster Weights')
plt.grid(True)
plt.legend(loc='upper right', fontsize=8)
plt.show()
