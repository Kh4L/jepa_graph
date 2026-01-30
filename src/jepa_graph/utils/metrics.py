"""
Evaluation metrics for Graph JEPA.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def compute_embedding_quality(
    embeddings: Tensor,
    labels: Optional[Tensor] = None,
) -> Dict[str, float]:
    """Compute metrics to assess embedding quality."""
    embeddings = embeddings.detach()
    N, D = embeddings.shape
    
    metrics = {}
    
    emb_norm = F.normalize(embeddings, dim=-1)
    
    sq_dist = torch.cdist(emb_norm, emb_norm, p=2).pow(2)
    uniformity = sq_dist.mul(-2).exp().mean().log().item()
    metrics["uniformity"] = uniformity
    
    if labels is not None:
        alignment = compute_alignment(emb_norm, labels)
        metrics["alignment"] = alignment
    
    variance = embeddings.var(dim=0)
    metrics["mean_variance"] = variance.mean().item()
    metrics["min_variance"] = variance.min().item()
    metrics["variance_std"] = variance.std().item()
    
    dead_dims = (variance < 1e-6).sum().item()
    metrics["dead_dimensions"] = dead_dims
    metrics["dead_ratio"] = dead_dims / D
    
    try:
        _, S, _ = torch.svd(embeddings - embeddings.mean(dim=0))
        S_norm = S / S.sum()
        entropy = -(S_norm * (S_norm + 1e-10).log()).sum()
        effective_rank = entropy.exp().item()
        metrics["effective_rank"] = effective_rank
        metrics["rank_ratio"] = effective_rank / D
    except Exception:
        metrics["effective_rank"] = -1
        metrics["rank_ratio"] = -1
    
    cov = (emb_norm.T @ emb_norm) / N
    eigenvalues = torch.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues.clamp(min=1e-10)
    isotropy = eigenvalues.min().item() / eigenvalues.max().item()
    metrics["isotropy"] = isotropy
    
    return metrics


def compute_alignment(
    embeddings: Tensor,
    labels: Tensor,
    alpha: float = 2.0,
) -> float:
    """Compute alignment: average distance between positive pairs."""
    unique_labels = labels.unique()
    
    total_alignment = 0.0
    count = 0
    
    for label in unique_labels:
        mask = labels == label
        if mask.sum() < 2:
            continue
        
        class_emb = embeddings[mask]
        distances = torch.cdist(class_emb, class_emb, p=2).pow(alpha)
        
        n = class_emb.size(0)
        distances = distances[~torch.eye(n, dtype=torch.bool, device=distances.device)]
        
        total_alignment += distances.mean().item()
        count += 1
    
    return total_alignment / max(count, 1)


def compute_graph_classification_accuracy(
    model,
    data_loader,
    classifier=None,
) -> Dict[str, float]:
    """Evaluate graph classification accuracy using learned embeddings."""
    device = next(model.parameters()).device
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            data = batch.to(device)
            
            if hasattr(model, "encode"):
                _, graph_emb = model.encode(data)
            else:
                _, graph_emb = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch,
                    return_node_embeddings=True,
                )
            
            all_embeddings.append(graph_emb)
            all_labels.append(data.y)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    if classifier is None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        X = embeddings.cpu().numpy()
        y = labels.cpu().numpy()
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5)
        
        return {
            "accuracy": scores.mean(),
            "accuracy_std": scores.std(),
        }
    else:
        classifier.eval()
        with torch.no_grad():
            logits = classifier(embeddings)
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()
        
        return {"accuracy": accuracy}


def compute_node_classification_accuracy(
    model,
    data,
    train_mask: Tensor,
    test_mask: Tensor,
) -> Dict[str, float]:
    """Evaluate node classification accuracy."""
    from sklearn.linear_model import LogisticRegression
    
    device = next(model.parameters()).device
    data = data.to(device)
    model.eval()
    
    with torch.no_grad():
        if hasattr(model, "encode"):
            node_emb, _ = model.encode(data)
        else:
            node_emb, _ = model(
                x=data.x,
                edge_index=data.edge_index,
                return_node_embeddings=True,
            )
    
    X_train = node_emb[train_mask].cpu().numpy()
    y_train = data.y[train_mask].cpu().numpy()
    X_test = node_emb[test_mask].cpu().numpy()
    y_test = data.y[test_mask].cpu().numpy()
    
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    
    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
    }
