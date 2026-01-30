# Graph-JEPA: Joint Embedding Predictive Architecture for Graph Neural Networks

A self-supervised learning framework for pretraining Graph Neural Networks using JEPA (Joint Embedding Predictive Architecture), designed for integration with GraphRAG and G-Retriever.

**Dataset:** [STARK-Prime](https://github.com/snap-stanford/stark) (Biomedical KG QA)  
**Baseline:** [G-Retriever](https://github.com/puririshi98/gretriever-stark-prime)

## Quick Start

```bash
# Install
pip install -e ".[stark]"

# Train on STARK-Prime (JEPA pretraining + G-Retriever finetuning)
python examples/train_stark_prime.py \
    --dataset prime \
    --pretrain-epochs 50 \
    --finetune-epochs 3
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: JEPA PRETRAINING                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐          ┌─────────────┐          ┌─────────────┐        │
│   │   CONTEXT   │          │  PREDICTOR  │          │   TARGET    │        │
│   │   ENCODER   │ ──────▶  │             │ ──────▶  │   ENCODER   │        │
│   │  (Online)   │          │ (Trainable) │          │    (EMA)    │        │
│   └─────────────┘          └─────────────┘          └─────────────┘        │
│         │                        │                        │                │
│    Gradients ✓               Gradients ✓            Gradients ✗            │
│    (trained)                 (trained)              (frozen, EMA)          │
│                                                                             │
│   Loss: MSE between predicted and actual target embeddings                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼  pretrained encoder
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: G-RETRIEVER FINETUNING                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Question  →  Subgraph Retrieval  →  JEPA Encoder  →  LLM  →  Answer      │
│               (KNN + PCST filter)      (GNN)          (Llama-3.1)          │
│                                                                             │
│   Loss: Cross-entropy on answer tokens                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Idea

**Traditional approach:** Train GNN from random initialization alongside LLM  
**Our approach:** Pretrain GNN with JEPA first, then finetune with LLM

JEPA pretraining teaches the GNN to predict *representations* of masked graph regions, learning semantic structure before seeing any QA labels.

## Installation

```bash
# Clone
git clone https://github.com/your-org/jepa-graph.git
cd jepa-graph

# Install with STARK-Prime support
pip install -e ".[stark]"

# Install PyG (if not already installed)
pip install torch-geometric torch-scatter torch-sparse
```

## Training Pipeline

### 1. JEPA Pretraining (Self-supervised)

```python
from jepa_graph.models.graph_jepa import GraphJEPAConfig
from jepa_graph.data.masking import GraphMasker, MaskingStrategy
from jepa_graph.training.trainer import JEPATrainer, TrainingConfig

# Build model
config = GraphJEPAConfig(
    encoder_type="gat",
    in_channels=384,  # STARK-Prime embedding dim
    hidden_channels=512,
    out_channels=512,
    num_heads=4,
    dropout=0.5,
)
model = config.build_model()

# Train with masking
masker = GraphMasker(strategy=MaskingStrategy.KHOP_BALL, k_hops=2)
trainer = JEPATrainer(model, config=TrainingConfig(max_epochs=50), train_loader=loader)
trainer.train()

# Export encoder
encoder = model.get_encoder_for_downstream(use_target=True)
torch.save(encoder.state_dict(), "jepa_encoder.pt")
```

### 2. G-Retriever Finetuning (Supervised QA)

```python
from jepa_graph.graphrag.gretriever import create_gretriever_from_jepa, GRetrieverTrainer

# Create G-Retriever with JEPA encoder
model = create_gretriever_from_jepa(
    jepa_encoder_path="jepa_encoder.pt",
    encoder_config={...},
    llm_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
)

# Finetune on QA
trainer = GRetrieverTrainer(model, train_loader, val_loader, test_loader, epochs=3)
trainer.train()
```

### Full Pipeline

```bash
# Everything in one command
python examples/train_stark_prime.py \
    --dataset prime \
    --pretrain-epochs 50 \
    --finetune-epochs 3 \
    --encoder-type gat \
    --hidden-dim 512
```

## Project Structure

```
jepa_graph/
├── src/jepa_graph/
│   ├── models/
│   │   ├── encoders.py        # GNN/Transformer encoders
│   │   ├── predictor.py       # JEPA predictor
│   │   └── graph_jepa.py      # Main JEPA model
│   ├── data/
│   │   ├── masking.py         # Graph masking strategies
│   │   ├── stark_prime.py     # STARK-Prime dataset
│   │   └── dataset.py         # Dataset utilities
│   ├── training/
│   │   ├── trainer.py         # JEPA training loop
│   │   └── losses.py          # Loss functions
│   └── graphrag/
│       ├── gretriever.py      # G-Retriever integration
│       ├── retriever.py       # Graph retrieval
│       └── pipeline.py        # End-to-end RAG
├── examples/
│   └── train_stark_prime.py   # Main training script
├── docs/
│   └── TRAINING_EXPLAINED.md  # Detailed walkthrough
└── configs/
    └── default.yaml           # Default config
```

## Masking Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `KHOP_BALL` | Mask k-hop neighborhood around anchor | Local structure |
| `RANDOM_NODES` | Random node sampling | Baseline |
| `RANDOM_SUBGRAPH` | Connected subgraph via random walk | Global context |
| `EDGE_PATH` | Mask shortest paths | Path reasoning |

## Evaluation Metrics

On STARK-Prime QA benchmark:

| Metric | Description |
|--------|-------------|
| F1 | Harmonic mean of precision/recall |
| Exact hit@1 | First prediction matches |
| Exact hit@5 | Any top-5 prediction matches |
| MRR | Mean reciprocal rank |

## Baseline Comparison

| Model | Encoder | Hit@1 |
|-------|---------|-------|
| LLM only | None | ~15% |
| G-Retriever | Random GAT | ~32% |
| **G-Retriever + JEPA** | JEPA-pretrained GAT | **TBD** |

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- PyTorch Geometric >= 2.4
- STARK-QA (for dataset)
- Transformers (for LLM)

## Citation

```bibtex
@software{jepa_graph,
  title = {Graph-JEPA: JEPA for Graph Neural Networks with G-Retriever},
  year = {2024},
  url = {https://github.com/your-org/jepa-graph}
}
```

## References

- [I-JEPA](https://arxiv.org/abs/2301.08243) - Image JEPA
- [G-Retriever](https://arxiv.org/abs/2402.07630) - GNN + LLM for GraphQA
- [STARK](https://github.com/snap-stanford/stark) - Knowledge Graph QA Benchmark

## License

MIT
