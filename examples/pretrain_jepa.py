#!/usr/bin/env python3
"""
Example: Pretrain a Graph JEPA model on a graph dataset.
"""

import argparse
from pathlib import Path

import torch
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.transforms import Compose

from jepa_graph.models.graph_jepa import GraphJEPAConfig
from jepa_graph.data.masking import GraphMasker, MaskingStrategy
from jepa_graph.data.dataset import create_jepa_dataloader
from jepa_graph.data.structural_encoding import StructuralEncodingTransform
from jepa_graph.training.trainer import JEPATrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Pretrain Graph JEPA")
    
    parser.add_argument("--dataset", type=str, default="PROTEINS")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--encoder-type", type=str, default="gat",
                       choices=["gat", "gcn", "gin", "transformer"])
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--masking", type=str, default="khop",
                       choices=["khop", "random", "subgraph", "path"])
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--k-hops", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.996)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--wandb", action="store_true")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    print(f"Loading dataset: {args.dataset}")
    transform = Compose([
        StructuralEncodingTransform(encoding_type="random_walk", encoding_dim=16),
    ])
    
    if args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        dataset = Planetoid(root=args.data_root, name=args.dataset, transform=transform)
    else:
        dataset = TUDataset(root=args.data_root, name=args.dataset, transform=transform)
    
    print(f"Dataset: {len(dataset)} graphs")
    
    sample = dataset[0]
    if sample.x is not None:
        in_channels = sample.x.size(1)
    else:
        in_channels = 1
        for data in dataset:
            data.x = torch.ones(data.num_nodes, 1)
    
    if hasattr(sample, "pe"):
        in_channels += sample.pe.size(1)
    
    print(f"Input channels: {in_channels}")
    
    strategy_map = {
        "khop": MaskingStrategy.KHOP_BALL,
        "random": MaskingStrategy.RANDOM_NODES,
        "subgraph": MaskingStrategy.RANDOM_SUBGRAPH,
        "path": MaskingStrategy.EDGE_PATH,
    }
    
    masker = GraphMasker(
        strategy=strategy_map[args.masking],
        mask_ratio=args.mask_ratio,
        k_hops=args.k_hops,
    )
    
    train_loader = create_jepa_dataloader(
        dataset=dataset,
        masker=masker,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    
    model_config = GraphJEPAConfig(
        encoder_type=args.encoder_type,
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        out_channels=args.hidden_dim,
        num_encoder_layers=args.num_layers,
        ema_decay=args.ema_decay,
    )
    
    model = model_config.build_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    training_config = TrainingConfig(
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        checkpoint_dir=args.output_dir,
        use_wandb=args.wandb,
        wandb_project="jepa-graph",
    )
    
    trainer = JEPATrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
    )
    
    print("Starting pretraining...")
    results = trainer.train()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    encoder = model.get_encoder_for_downstream(use_target=True)
    torch.save(encoder.state_dict(), output_path / "pretrained_encoder.pt")
    
    print(f"Pretraining complete! Encoder saved to {output_path / 'pretrained_encoder.pt'}")
    print(f"Final training loss: {results['history'][-1]['train']['loss']:.4f}")


if __name__ == "__main__":
    main()
