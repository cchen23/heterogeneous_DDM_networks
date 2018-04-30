import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time

figure_path = "../Figures/"
axis_fontsize = 12
title_fontsize = 20
legendtitle_fontsize = 16
legendlabel_fontsize = 14

def plot_unique_trajectories(x, indices, leaderindex=None, labels=None):
    epsilon = 0.00001
    num_individuals = x.shape[0]
    unique_trajectories = {} # key is an index of trajectory, value is label.
    included_individuals = [] # Individuals that have been accounted for.
    for i in range(num_individuals):
        if indices[i] in included_individuals:
            continue
        matching_trajectories = [indices[i]]
        for j in range(i+1, num_individuals):
            if indices[j] in included_individuals:
                continue
            elif (abs(x[i,:] - x[j,:]) < epsilon).all():
                matching_trajectories.append(indices[j])
                included_individuals.append(indices[j])
        unique_trajectories[i] = matching_trajectories
    for i in unique_trajectories.keys():
        if labels:
            label = ", ".join("(" + labels[index] + ")" for index in unique_trajectories[i])
        else:
            label = ", ".join("(" + str(index+1) + ")" for index in unique_trajectories[i])
        if leaderindex != None and i == leaderindex:
            plt.plot(x[i,:], label=label, linewidth=3, color='k')
        else:
            plt.plot(x[i,:], label=label)

"""
Simulation Helpers.
"""
def coupled_update(x, beta, L, dt, sigma):
    assert(x.shape[0] == beta.shape[0] == L.shape[0] == L.shape[1] == sigma.shape[0])
    assert(x.shape[1] == beta.shape[1] == 1)
    num_individuals = x.shape[0]
    return x + (beta - np.dot(L, x)) * dt + sigma * np.random.randn(num_individuals, 1)

def run_simulation(x0, beta, L, dt, sigma, T, leaderindex=None, animate=False, title=None, savename=None, ylim=None):
    num_individuals = x0.shape[0]
    num_timesteps = int(T/dt)
    xtick_spacing = int(num_timesteps / 10)
    t_vals = np.arange(0, T, xtick_spacing * dt)
    x = np.empty([num_individuals, num_timesteps])
    x[:,0] = np.squeeze(x0)
    for i in range(num_timesteps-1):
        x[:,i+1] = np.squeeze(coupled_update(np.expand_dims(x[:,i], axis=1), beta, L, dt, sigma))
        if animate:
            plt.plot(x[:,:i+1].T)
            plt.xlim([0, num_timesteps])
            plt.xticks(t_vals)
            # plt.ylim([-10, 10])
            plt.ylabel("x")
            plt.xlabel("t")
            plt.xticks(range(0, num_timesteps, xtick_spacing), t_vals)
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.0001)
    plot_unique_trajectories(x, range(num_individuals), leaderindex)
    # plt.plot(np.sum(x, axis=0), label=r"$\sum x_i$", linestyle=':', color='k')
    plt.xlim([0, num_timesteps])
    ylim = ylim if ylim else np.max(np.absolute(x)) * 1.1
    plt.ylim([-ylim, ylim])
    plt.ylabel("x", fontsize=axis_fontsize)
    plt.xlabel("t", fontsize=axis_fontsize)
    plt.xticks(range(0, num_timesteps, xtick_spacing), t_vals)
    legend = plt.legend(bbox_to_anchor=(1, 1), loc='upper left', ncol=1, fontsize = legendlabel_fontsize)
    plt.setp(legend.get_title(),fontsize=legendtitle_fontsize)
    if title:
        plt.title(title, fontsize=title_fontsize)
    if savename:
        plt.savefig(figure_path + savename, bbox_inches="tight")
    return x

"""
Discrepancy.
"""
def compute_discrepancy(x_t, center, L):
    squared_diffs = np.square(x_t - x_t[center])
    return np.sum(squared_diffs * (L[center,:] != 0))

def compute_network_discrepancy(x, beta, leader_indices, L):
    num_leaders = len(leader_indices)
    discrepancies = np.empty([num_leaders, x.shape[1]])
    for i in range(len(leader_indices)):
        discrepancies[i,:] = [np.abs(beta[leader_indices[i]-1]) * compute_discrepancy(x[:,t], leader_indices[i]-1, L) / np.sum(np.abs(beta)) for t in range(x.shape[1])]
    return np.sum(discrepancies, axis=0)

def plot_discrepancy(x, centers, L, T, ylim=None, title=None, savename=None):
    discrepancies = np.empty([len(centers), x.shape[1]])
    for i in range(len(centers)):
        center = centers[i]
        discrepancies[i,:] = [compute_discrepancy(x[:,t], center-1, L) for t in range(x.shape[1])]
    plot_unique_trajectories(discrepancies, one_to_zero_indexing(centers))
    if title:
        plt.title(title, fontsize=title_fontsize)
    else:
        plt.title("Discrepancy vs Time", fontsize=title_fontsize)
    if ylim:
        plt.ylim([0, ylim])
    plt.xlabel("t", fontsize=axis_fontsize)
    plt.ylabel("Discrepancy", fontsize=axis_fontsize)
    legend = plt.legend(title="Individual", fontsize=legendlabel_fontsize)
    plt.setp(legend.get_title(),fontsize=legendtitle_fontsize)
    t_vals = np.arange(0, T, 1)
    plt.xticks(range(0, x.shape[1], int(x.shape[1] / T)), t_vals)
    if savename:
        plt.savefig(figure_path + savename, bbox_inches="tight")
    plt.show()
    print("Total final discrepancy: %0.2f" % np.sum(discrepancies[:,-1]))
    print("Final discrepancy: %s" % str(discrepancies[:,-1]))
    return discrepancies

def compute_ydot(L, beta):
    E = nx.incidence_matrix(nx.Graph(L), oriented=True).toarray()
    E_t = np.transpose(E)
    y_dot = np.dot(np.linalg.pinv(np.matmul(E_t, E)), np.matmul(E_t, beta))
    y_dot_descriptions = []
    for i in range(E.shape[1]):
        edge = E[:,i]
        if np.max(edge) == 0:
            y_dot_descriptions.append("NA")
        else:
            s = np.where(edge==-1)[0]+1
            t = np.where(edge==1)[0]+1
            y_dot_descriptions.append("x%d-x%d" % (t,s))
    for description, diff in zip(y_dot_descriptions, y_dot):
        print("%s\t: %0.2f" % (description, diff))
    return y_dot

"""
Discrepancy, Fitch & Leonard Model.
"""
def compute_discrepancy_signalmeasuring(x, center, L):
    squared_diffs = np.sum(np.square(x - x[center,:]), axis=1)
    return np.sum(squared_diffs * (L[center,:] != 0))

def compute_totalnetwork_discrepancy_signalmeasuring(x, k, centers, L):
    numerator = 0
    for center in centers:
        numerator += compute_discrepancy_signalmeasuring(x, center, L) * abs(k[center])
    return numerator / sum(abs(k))

"""
Graph Creators.
"""
def connect(L, a, b):
    L = np.copy(L)
    L[a,b] = -1
    L[b,a] = -1
    return L

def fill_laplacian_diagonal(L):
    num_individuals = L.shape[0]
    for i in range(num_individuals):
        degree = np.sum(L[i,:]) - L[i,i]
        L[i,i] = -1 * degree
    return L

def create_fullyconnected_graph(num_individuals):
    L = -1 * np.ones([num_individuals, num_individuals])
    L = fill_laplacian_diagonal(L)
    return L

def create_line_graph(num_individuals):
    L = np.zeros([num_individuals, num_individuals])
    for i in range(1, num_individuals - 1):
        L[i, i - 1] = -1
        L[i, i + 1] = -1
    L[0, 1] = -1
    L[num_individuals - 1, num_individuals - 2] = -1
    L = fill_laplacian_diagonal(L)
    return L

def create_circle_graph(num_individuals):
    L = np.zeros([num_individuals, num_individuals])
    for i in range(num_individuals):
        left_index = (i - 1) if i > 0 else -1
        right_index = (i + 1) if i < (num_individuals - 1) else 0
        L[i,left_index] = -1
        L[i, right_index] = -1
    L = fill_laplacian_diagonal(L)
    return L

def create_paper_graph(num_individuals):
    assert(num_individuals == 9)
    L = np.zeros([num_individuals, num_individuals])
    L = connect(L, 1, 5)
    L = connect(L, 2, 6)
    L = connect(L, 1, 2)
    L = connect(L, 0, 1)
    L = connect(L, 0, 2)
    L = connect(L, 0, 3)
    L = connect(L, 0, 4)
    L = connect(L, 3, 7)
    L = connect(L, 3, 4)
    L = connect(L, 4, 8)
    L = fill_laplacian_diagonal(L)
    return L

def create_erdosreyni_graph(num_individuals, p):
    G = nx.erdos_renyi_graph(num_individuals, p)
    return nx.laplacian_matrix(G).toarray()

def create_tree_graph():
    num_individuals = 10
    L = np.zeros([num_individuals, num_individuals])
    L = connect(L, 0, 1)
    L = connect(L, 0, 2)
    L = connect(L, 1, 3)
    L = connect(L, 1, 4)
    L = connect(L, 2, 5)
    L = connect(L, 2, 6)
    L = connect(L, 3, 7)
    L = connect(L, 3, 8)
    L = connect(L, 4, 9)
    return L

"""
Graph Drawers.
"""
def draw_graph(L, savename=None):
    G = nx.Graph(L)
    num_nodes = L.shape[0]
    labels = {n:(n+1) for n in range(num_nodes)}
    nx.draw(G, labels=labels, with_labels=True)
    if savename:
        plt.savefig(figure_path + savename, bbox_inches="tight")

def draw_paper_graph(L, savename=None):
    G = nx.Graph(L)
    hdist = 0.3
    vdist = 0.05
    num_nodes = L.shape[0]
    assert(num_nodes == 9)
    pos = {
        0:[0,0],
        1:[-hdist,vdist],
        2:[-hdist,-vdist],
        3:[hdist,vdist],
        4:[hdist,-vdist],
        5:[-2*hdist,vdist],
        6:[-2*hdist,-vdist],
        7:[2*hdist,vdist],
        8:[2*hdist,-vdist]
    }
    labels = {n:(n+1) for n in range(num_nodes)}
    nx.draw(G, pos=pos, labels=labels, with_labels=True)
    if savename:
        plt.savefig("../Figures/%s" % savename, bbox_inches="tight")

def draw_circle_graph(L, savename=None, directed=False):
    G = nx.Graph(L)
    if directed:
        G = nx.DiGraph(L)
    pos = nx.circular_layout(G)
    num_nodes = L.shape[0]
    labels = {n:(n+1) for n in range(num_nodes)}
    nx.draw(G, pos=pos, labels=labels, with_labels=True, arrows=True)
    if savename:
        plt.savefig(figure_path + savename, bbox_inches="tight")

def draw_line_graph(L, savename=None, directed=False):
    G = nx.Graph(L)
    if directed:
        G = nx.DiGraph(L)
    hdist = 0.3
    num_nodes = L.shape[0]
    pos = {i:[i * hdist - (num_nodes/2 * hdist), 0] for i in range(num_nodes)}
    labels = {n:(n+1) for n in range(num_nodes)}
    nx.draw(G, pos=pos, labels=labels, with_labels=True, arrows=True)
    if savename:
        plt.savefig(figure_path + savename, bboxinches="tight")

def draw_tree_graph(L, savename=None):
    G = nx.Graph(L)
    hdist = 0.3
    vdist = 0.05
    num_nodes = L.shape[0]
    pos = {
        0:[0,0],
        1:[hdist, vdist],
        2:[hdist, -vdist],
        3:[2*hdist, 1.5*vdist],
        4:[2*hdist, 0.5*vdist],
        5:[2*hdist, -0.5*vdist],
        6:[2*hdist, -1.5*vdist],
        7:[3*hdist, 2*vdist],
        8:[3*hdist, vdist],
        9:[3*hdist, 0.25*vdist]
    }
    labels = {n:(n+1) for n in range(num_nodes)}
    nx.draw(G, pos=pos, labels=labels, with_labels=True)
    if savename:
        plt.savefig("../Figures/%s" % savename, bboxinches="tight")

"""
Beta creators.
"""

def create_homogeneous_beta(num_individuals):
    return np.ones([num_individuals, 1])

def create_leaderfollower_beta(num_individuals, leader_indices, leader_betas):
    leader_indices = one_to_zero_indexing(leader_indices)
    betas = np.zeros([num_individuals, 1])
    for leader_index, leader_beta in zip(leader_indices, leader_betas):
        betas[leader_index] = leader_beta
    return betas

"""
One-indexing util.
"""
def one_to_zero_indexing(indices):
    for index in indices:
        assert(index > 0)
    return [(index-1) for index in indices]

def zero_to_one_indexing(indices):
    return [(index+1) for index in indices]

def get_node_value(index, x):
    return x[index-1,:]
