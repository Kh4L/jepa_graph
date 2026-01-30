# Graph-JEPA Training: Complete Walkthrough

This document explains **what we're training**, **where training happens**, and **how all the pieces fit together**.

**Dataset:** STARK-Prime (Knowledge Graph QA Benchmark)
**Baseline:** [G-Retriever](https://github.com/puririshi98/gretriever-stark-prime)

---

## TL;DR - The Big Picture

We're training a **Graph Neural Network (GNN) encoder** to learn good graph representations **without labels** (self-supervised), then finetuning with an LLM for QA. The key insight from JEPA:

> **Don't predict the raw graph structure. Predict the *representation* of masked parts.**

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GRAPH JEPA ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐        │
│   │   CONTEXT   │          │  PREDICTOR  │          │   TARGET    │        │
│   │   ENCODER   │ ──────▶  │             │ ──────▶  │   ENCODER   │        │
│   │  (Online)   │          │ (Trainable) │          │    (EMA)    │        │
│   └─────────────┘          └─────────────┘          └─────────────┘        │
│         │                        │                        │                │
│         │                        │                        │                │
│    Gradients ✓               Gradients ✓            Gradients ✗            │
│    (trained)                 (trained)              (frozen, EMA)          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## The Three Main Components

### 1. Context Encoder (Trainable)
**Location:** `src/jepa_graph/models/encoders.py` → `GraphEncoder`

```python
# What it does:
# Takes the VISIBLE part of the graph (context region)
# Outputs node embeddings for these visible nodes

context_node_emb, _ = self.context_encoder(
    x=context_data.x,           # Node features of visible nodes
    edge_index=context_data.edge_index,  # Edges between visible nodes
)
# Output: [num_context_nodes, embed_dim] → e.g., [85, 256]
```

**This encoder gets trained via backpropagation.**

---

### 2. Target Encoder (EMA - Not Trained Directly)
**Location:** `src/jepa_graph/models/graph_jepa.py` → lines 45-48

```python
# Created as a COPY of context encoder
self.target_encoder = copy.deepcopy(encoder)

# Gradients are DISABLED
for param in self.target_encoder.parameters():
    param.requires_grad = False
```

```python
# What it does:
# Takes the MASKED part of the graph (target region)
# Outputs the "ground truth" embeddings we want to predict

with torch.no_grad():  # No gradients!
    target_node_emb, _ = self.target_encoder(
        x=target_data.x,           # Node features of masked nodes
        edge_index=target_data.edge_index,  # Edges in masked region
    )
# Output: [num_target_nodes, embed_dim] → e.g., [15, 256]
```

**Updated via Exponential Moving Average (EMA) after each step:**

```python
# After each training step:
target_params = 0.996 * target_params + 0.004 * context_params
```

---

### 3. Predictor (Trainable)
**Location:** `src/jepa_graph/models/predictor.py` → `JEPAPredictor`

```python
# What it does:
# Takes context embeddings
# Predicts what the target embeddings SHOULD be

predicted_embeddings = self.predictor(
    context_embeddings=context_node_emb,  # From context encoder
    num_target_nodes=15,                   # How many to predict
)
# Output: [num_target_nodes, embed_dim] → e.g., [15, 256]
```

**This predictor is trained via backpropagation.**

---

## Training Flow Step-by-Step

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP (ONE ITERATION)                      │
└────────────────────────────────────────────────────────────────────────────┘

STEP 1: DATA PREPARATION
════════════════════════

    Original Graph (e.g., 100 nodes)
    ┌────────────────────────────────┐
    │  ●──●──●──●──●──●──●──●──●──●  │
    │  │  │  │  │  │  │  │  │  │  │  │
    │  ●──●──●──●──●──●──●──●──●──●  │
    │     │     │     │     │     │  │
    │  ●──●──●──●──●──●──●──●──●──●  │
    └────────────────────────────────┘
                    │
                    ▼  GraphMasker (k-hop ball)
    
    ┌─────────────────┐   ┌─────────────────┐
    │  CONTEXT (85)   │   │  TARGET (15)    │
    │  ●──●──●──●──●  │   │     ●──●──●     │
    │  │  │     │  │  │   │     │  │  │     │
    │  ●──●     ●──●  │   │     ●──●──●     │
    │     │     │     │   │        │        │
    │  ●──●──●──●──●  │   │     ●──●──●     │
    │   (visible)     │   │   (masked)      │
    └─────────────────┘   └─────────────────┘
    
    Location: src/jepa_graph/data/masking.py


STEP 2: ENCODING
════════════════

    Context Data                          Target Data
         │                                     │
         ▼                                     ▼
    ┌─────────────┐                     ┌─────────────┐
    │   Context   │                     │   Target    │
    │   Encoder   │                     │   Encoder   │
    │   (GNN)     │                     │   (GNN)     │
    │             │                     │   [EMA]     │
    └─────────────┘                     └─────────────┘
         │                                     │
         ▼                                     ▼
    context_emb                          target_emb
    [85, 256]                            [15, 256]
    (gradients ✓)                        (stop_grad ✗)


STEP 3: PREDICTION
══════════════════

    context_emb [85, 256]
         │
         ▼
    ┌─────────────────────────────┐
    │        PREDICTOR            │
    │  ┌─────────────────────┐    │
    │  │ Cross-Attention     │    │
    │  │ (context → queries) │    │
    │  └─────────────────────┘    │
    │            │                │
    │  ┌─────────────────────┐    │
    │  │ Transformer Layers  │    │
    │  └─────────────────────┘    │
    │            │                │
    │  ┌─────────────────────┐    │
    │  │ Output Projection   │    │
    │  └─────────────────────┘    │
    └─────────────────────────────┘
         │
         ▼
    predicted_emb [15, 256]


STEP 4: LOSS COMPUTATION
════════════════════════

    predicted_emb [15, 256]        target_emb [15, 256]
         │                              │
         │      ┌────────────┐          │
         └─────▶│  MSE LOSS  │◀─────────┘
                │ (on L2-    │
                │ normalized │
                │ vectors)   │
                └────────────┘
                      │
                      ▼
                loss = 0.0234
    
    Location: src/jepa_graph/training/losses.py


STEP 5: BACKPROPAGATION & UPDATE
════════════════════════════════

                    loss
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │   Context   │         │  Predictor  │
    │   Encoder   │         │             │
    │  weights    │         │  weights    │
    └─────────────┘         └─────────────┘
          │                       │
          │   optimizer.step()    │
          ▼                       ▼
    [Updated via                [Updated via
     gradients]                  gradients]


STEP 6: EMA UPDATE (Target Encoder)
═══════════════════════════════════

    ┌─────────────┐              ┌─────────────┐
    │   Context   │              │   Target    │
    │   Encoder   │─────────────▶│   Encoder   │
    │  (updated)  │    EMA       │  (updated)  │
    └─────────────┘              └─────────────┘
    
    target_params = 0.996 * target_params + 0.004 * context_params
    
    Location: src/jepa_graph/models/graph_jepa.py → update_target_encoder()
```

---

## Where Each File Fits In

```
src/jepa_graph/
├── models/
│   ├── encoders.py         ← GNN/Transformer architectures (STEP 2)
│   ├── predictor.py        ← Context→Target prediction (STEP 3)
│   └── graph_jepa.py       ← Ties everything together (STEPS 2-6)
│
├── data/
│   ├── masking.py          ← Split graph into context/target (STEP 1)
│   ├── dataset.py          ← PyG dataset wrapper
│   └── structural_encoding.py  ← Position encodings
│
├── training/
│   ├── trainer.py          ← Training loop (ALL STEPS)
│   └── losses.py           ← JEPA loss functions (STEP 4)
│
└── graphrag/               ← Uses pretrained encoder (AFTER training)
    ├── retriever.py
    ├── fusion.py
    └── pipeline.py
```

---

## What Gets Trained vs. What Doesn't

| Component | Trained via Backprop? | Updated How? |
|-----------|----------------------|--------------|
| **Context Encoder** | ✅ Yes | Gradient descent |
| **Predictor** | ✅ Yes | Gradient descent |
| **Target Encoder** | ❌ No | EMA from context encoder |

---

## The Key Insight: Why This Works

### Traditional Autoencoder (Masked Autoencoding):
```
Masked Graph → Encoder → Decoder → Reconstruct masked nodes/edges
                                   (predict raw structure)
```
**Problem:** Forces encoder to memorize low-level details.

### JEPA Approach:
```
Context → Context Encoder → Predictor → Predict TARGET EMBEDDINGS
Target  → Target Encoder  →            (not raw structure)
```
**Benefit:** Encoder learns *semantic* features, not surface-level details.

---

## Training Configuration

**Location:** `src/jepa_graph/training/trainer.py` → `TrainingConfig`

```python
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-4      # AdamW learning rate
    weight_decay: float = 0.05       # L2 regularization
    max_epochs: int = 100            # Training epochs
    batch_size: int = 32             # Graphs per batch
    gradient_clip: float = 1.0       # Gradient clipping
    
    # EMA settings
    ema_decay: float = 0.996         # τ in: target = τ*target + (1-τ)*context
    
    # Loss
    loss_type: str = "mse"           # MSE on normalized embeddings
```

---

## After Training: Using the Encoder

Once training is complete, you extract the **target encoder** (more stable due to EMA):

```python
# Get the pretrained encoder
encoder = model.get_encoder_for_downstream(use_target=True)

# Save it
torch.save(encoder.state_dict(), "pretrained_encoder.pt")

# Use for GraphRAG
retriever = GraphRetriever(graph_encoder=encoder)
result = retriever.retrieve(query, knowledge_graph)
```

---

## Visual Summary: End-to-End

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRETRAINING PHASE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Graphs        Masking         Encoding        Prediction       Loss        │
│  ──────        ───────         ────────        ──────────       ────        │
│                                                                             │
│   ●●●●●   →   Context   →   Context Enc  →   Predictor   →   MSE Loss      │
│   ●●●●●       ●●●●●         [85, 256]        [15, 256]       ↓             │
│   ●●●●●       Target    →   Target Enc   ─────────────────▶ compare        │
│               ●●●●●         [15, 256]                                       │
│                             (stop-grad)                                     │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                     │                                       │
│                                     ▼                                       │
│                            Update Context Encoder (gradients)               │
│                            Update Predictor (gradients)                     │
│                            Update Target Encoder (EMA)                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ Save pretrained_encoder.pt
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DOWNSTREAM PHASE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Query   →   Retriever   →   Encode Subgraph   →   Fusion   →   LLM       │
│   "What        (uses            (pretrained           (graph     Answer     │
│    is..."    encoder)           encoder)              + text)              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Running Training

### On STARK-Prime (Recommended)

```bash
# Full pipeline: JEPA pretraining + G-Retriever finetuning
python examples/train_stark_prime.py \
    --dataset prime \
    --pretrain-epochs 50 \
    --finetune-epochs 3 \
    --hidden-dim 512 \
    --encoder-type gat

# Skip pretraining (use existing encoder)
python examples/train_stark_prime.py \
    --encoder-path outputs/stark_prime/jepa_pretrained_encoder.pt \
    --skip-pretrain

# Only pretraining (no LLM finetuning)
python examples/train_stark_prime.py \
    --skip-finetune
```

### On TU Datasets (Simple Testing)

```bash
python examples/pretrain_jepa.py --dataset PROTEINS --epochs 100
```

**Output:**
- `outputs/stark_prime/jepa_pretrained_encoder.pt` - JEPA pretrained encoder
- `outputs/stark_prime/gretriever_finetuned.pt` - Full G-Retriever model
- `outputs/stark_prime/encoder_config.pt` - Encoder configuration

---

## Key Equations

### EMA Update (Target Encoder)
$$\theta_{\text{target}} \leftarrow \tau \cdot \theta_{\text{target}} + (1 - \tau) \cdot \theta_{\text{context}}$$

where $\tau = 0.996$ (momentum coefficient)

### JEPA Loss
$$\mathcal{L} = \| \hat{z}_T - \text{sg}(z_T) \|_2^2$$

where:
- $\hat{z}_T$ = predicted target embeddings (from predictor)
- $z_T$ = actual target embeddings (from target encoder)
- $\text{sg}(\cdot)$ = stop gradient operator

---

---

## Two-Phase Training on STARK-Prime

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: JEPA PRETRAINING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STARK-Prime KG              Subgraph                JEPA                   │
│  (129K nodes)     →         Sampling       →       Pretraining              │
│                              (5000 samples)         (50 epochs)             │
│                                                          │                  │
│                                                          ▼                  │
│                                                   jepa_encoder.pt           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: G-RETRIEVER FINETUNING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  QA Pairs              Subgraph           JEPA Encoder                      │
│  (train/val/test)  →   Retrieval     →    + LLM        →    Answers         │
│                        (KNN + PCST)       (Llama-3.1)                       │
│                                                                             │
│  Loss: Cross-entropy on generated answer tokens                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### STARK-Prime Dataset

| Split | QA Pairs | Purpose |
|-------|----------|---------|
| Train | ~70% | G-Retriever finetuning |
| Val | ~20% | Hyperparameter tuning |
| Test | ~10% | Final evaluation |

**Knowledge Graph:**
- ~129K biomedical entities
- ~500K relations
- Node embeddings: 384-dim (sentence transformer)

---

## Alignment with G-Retriever Baseline

This implementation follows the [G-Retriever baseline](https://github.com/puririshi98/gretriever-stark-prime):

| Component | Baseline | Our Implementation |
|-----------|----------|-------------------|
| GNN Encoder | Random init GAT | **JEPA-pretrained GAT** |
| LLM | Llama-3.1-8B | Llama-3.1-8B |
| Retrieval | KNN + PCST | KNN + PCST |
| LR Schedule | Cosine + warmup | Cosine + warmup |
| Training | 3 epochs | 3 epochs |

**Key difference:** We pretrain the GNN encoder with JEPA first, rather than training from random initialization.

---

## Evaluation Metrics

```python
# STARK-Prime metrics (computed in train_stark_prime.py)
F1              # Harmonic mean of precision/recall
Precision       # |pred ∩ label| / |pred|
Recall          # |pred ∩ label| / |label|
Exact hit@1     # First prediction in label set
Exact hit@5     # Any of top-5 in label set
MRR             # Mean reciprocal rank
```

---

## Summary

| Question | Answer |
|----------|--------|
| **What are we training?** | Context Encoder + Predictor (GNN that learns graph representations) |
| **What's the objective?** | Predict embeddings of masked graph regions from visible regions |
| **Where does training happen?** | `trainer.py` orchestrates, `graph_jepa.py` defines forward pass |
| **What's special about JEPA?** | Predicts representations, not raw data → learns semantics |
| **Why EMA?** | Stable target prevents collapse (no negative samples needed) |
| **What dataset?** | STARK-Prime (biomedical KG QA benchmark) |
| **End result?** | JEPA-pretrained GNN for G-Retriever (GNN+LLM for QA) |
