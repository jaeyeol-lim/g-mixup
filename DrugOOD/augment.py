"""Class-graphon G-Mixup augmentation with DrugOOD feature templates."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data


def _aligned_statistics(graphs, resolution, node_dim, edge_dim):
    aligned_adjacencies = []
    feature_sum = np.zeros((resolution, node_dim), dtype=np.float64)
    feature_count = np.zeros((resolution, 1), dtype=np.float64)
    edge_sum = np.zeros(edge_dim, dtype=np.float64)
    edge_count = 0
    for graph in graphs:
        node_count = int(graph.num_nodes)
        adjacency = np.zeros((node_count, node_count), dtype=np.float32)
        edge_index = graph.edge_index.cpu().numpy()
        adjacency[edge_index[0], edge_index[1]] = 1.0
        order = np.argsort(adjacency.sum(0) + adjacency.sum(1))[::-1]
        take = min(node_count, resolution)
        selected = np.ascontiguousarray(order[:take])
        selected_tensor = torch.from_numpy(selected).long()
        aligned = np.zeros((resolution, resolution), dtype=np.float32)
        aligned[:take, :take] = adjacency[np.ix_(selected, selected)]
        aligned_adjacencies.append(aligned)
        feature_sum[:take] += graph.x.index_select(0, selected_tensor).float().cpu().numpy()
        feature_count[:take] += 1
        if graph.edge_attr.numel():
            edge_sum += graph.edge_attr.float().cpu().numpy().sum(0)
            edge_count += graph.edge_attr.shape[0]
    mean_adjacency = np.mean(aligned_adjacencies, axis=0)
    u, singular_values, vh = np.linalg.svd(mean_adjacency, full_matrices=False)
    singular_values[singular_values < 0.2 * np.sqrt(resolution)] = 0
    graphon = np.clip((u * singular_values) @ vh, 0, 1)
    node_template = feature_sum / np.maximum(feature_count, 1)
    edge_template = edge_sum / max(edge_count, 1)
    return graphon, node_template.astype(np.float32), edge_template.astype(np.float32)


def _sample_graph(graphon, node_features, edge_features, soft_label, rng):
    adjacency = (rng.random(graphon.shape) <= graphon).astype(np.int64)
    adjacency = np.triu(adjacency, k=1)
    adjacency = adjacency + adjacency.T
    active = np.flatnonzero(adjacency.sum(1) > 0)
    if active.size < 2:
        upper = np.triu(graphon, k=1)
        source, target = np.unravel_index(np.argmax(upper), upper.shape)
        if source == target:
            source, target = 0, min(1, graphon.shape[0] - 1)
        active = np.array(sorted({source, target}))
        adjacency[source, target] = adjacency[target, source] = 1
    adjacency = adjacency[np.ix_(active, active)]
    row, col = np.nonzero(adjacency)
    edge_index = torch.tensor(np.stack((row, col)), dtype=torch.long)
    edge_attr = torch.from_numpy(np.repeat(edge_features[None, :], len(row), axis=0)).float()
    return Data(
        x=torch.from_numpy(node_features[active]).float(),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(soft_label, dtype=torch.float32),
        num_nodes=int(active.size),
    )


def make_augmented_dataset(dataset, augmented_ratio, seed, aug_num=10, lam_range=(0.005, 0.01)):
    originals = [dataset[index] for index in range(len(dataset))]
    if augmented_ratio <= 0:
        return [_with_soft_label(graph) for graph in originals], 0
    classes = {label: [] for label in (0, 1)}
    for graph in originals:
        classes[int(graph.y.view(-1)[0])].append(graph)
    if any(not graphs for graphs in classes.values()):
        raise ValueError("G-Mixup requires both IC50 classes in the training split")
    node_counts = [int(graph.num_nodes) for graph in originals]
    resolution = max(2, int(np.median(node_counts)))
    node_dim = int(originals[0].x.shape[-1])
    edge_dim = int(originals[0].edge_attr.shape[-1])
    statistics = {
        label: _aligned_statistics(graphs, resolution, node_dim, edge_dim)
        for label, graphs in classes.items()
    }
    per_mix = int(len(originals) * augmented_ratio / aug_num)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    synthetic = []
    for lam in rng.uniform(lam_range[0], lam_range[1], size=aug_num):
        first, second = py_rng.sample((0, 1), 2)
        graphon = lam * statistics[first][0] + (1 - lam) * statistics[second][0]
        node_features = lam * statistics[first][1] + (1 - lam) * statistics[second][1]
        edge_features = lam * statistics[first][2] + (1 - lam) * statistics[second][2]
        soft_label = lam * F.one_hot(torch.tensor(first), 2).numpy() + (1 - lam) * F.one_hot(torch.tensor(second), 2).numpy()
        for _ in range(per_mix):
            synthetic.append(_sample_graph(graphon, node_features, edge_features, soft_label, rng))
    return [_with_soft_label(graph) for graph in originals] + synthetic, len(synthetic)


def _with_soft_label(graph):
    # DrugOOD caches may carry split-specific metadata such as ``group``.
    # Synthetic G-Mixup graphs do not have meaningful values for those fields,
    # and PyG requires every graph in a batch to expose the same key set.
    # Keep only the attributes consumed by the GINE classifier.
    return Data(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        y=F.one_hot(graph.y.view(-1)[0].long(), num_classes=2).float(),
        num_nodes=int(graph.num_nodes),
    )
