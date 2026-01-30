#!/usr/bin/env python3
"""
Train Graph JEPA on STARK-Prime dataset.

This script provides end-to-end training:
1. JEPA pretraining on the STARK-Prime knowledge graph
2. G-Retriever finetuning with JEPA encoder
3. Evaluation on QA benchmark

Aligned with baseline: https://github.com/puririshi98/gretriever-stark-prime
"""

import argparse
import os
import gc
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Local imports
from jepa_graph.models.graph_jepa import GraphJEPAConfig
from jepa_graph.data.masking import GraphMasker, MaskingStrategy
from jepa_graph.data.stark_prime import STARKPrimeDataset
from jepa_graph.training.trainer import JEPATrainer, TrainingConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Graph JEPA on STARK-Prime",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="prime",
                       choices=["prime", "mag", "amazon"],
                       help="STARK dataset name")
    parser.add_argument("--data-dir", type=str, default="./data/stark",
                       help="Data cache directory")
    
    # JEPA Pretraining
    parser.add_argument("--pretrain-epochs", type=int, default=50,
                       help="JEPA pretraining epochs")
    parser.add_argument("--pretrain-batch-size", type=int, default=32,
                       help="Pretraining batch size")
    parser.add_argument("--pretrain-lr", type=float, default=1e-4,
                       help="Pretraining learning rate")
    
    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=512,
                       help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=3,
                       help="Number of GNN layers")
    parser.add_argument("--num-heads", type=int, default=4,
                       help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.5,
                       help="Dropout rate")
    parser.add_argument("--encoder-type", type=str, default="gat",
                       choices=["gat", "gcn", "gin", "transformer"],
                       help="GNN encoder type")
    
    # Masking
    parser.add_argument("--masking", type=str, default="khop",
                       choices=["khop", "random", "subgraph", "path"],
                       help="Masking strategy")
    parser.add_argument("--mask-ratio", type=float, default=0.15,
                       help="Mask ratio")
    parser.add_argument("--k-hops", type=int, default=2,
                       help="K-hops for masking")
    
    # G-Retriever Finetuning
    parser.add_argument("--finetune-epochs", type=int, default=3,
                       help="Finetuning epochs")
    parser.add_argument("--finetune-lr", type=float, default=1e-5,
                       help="Finetuning learning rate")
    parser.add_argument("--finetune-batch-size", type=int, default=1,
                       help="Finetuning batch size (small due to LLM memory)")
    parser.add_argument("--llm-model", type=str,
                       default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                       help="LLM model for G-Retriever")
    parser.add_argument("--freeze-gnn", action="store_true",
                       help="Freeze GNN during finetuning")
    
    # Retrieval settings
    parser.add_argument("--num-hops", type=int, default=3,
                       help="Number of hops for neighbor sampling")
    parser.add_argument("--k-nodes", type=int, default=16,
                       help="K for KNN seed selection")
    parser.add_argument("--fanout", type=int, default=10,
                       help="Fanout for neighbor sampling")
    
    # Training
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    parser.add_argument("--output-dir", type=str, default="./outputs/stark_prime",
                       help="Output directory")
    parser.add_argument("--skip-pretrain", action="store_true",
                       help="Skip JEPA pretraining")
    parser.add_argument("--skip-finetune", action="store_true",
                       help="Skip G-Retriever finetuning")
    parser.add_argument("--encoder-path", type=str, default=None,
                       help="Path to pretrained encoder (skips pretraining)")
    
    return parser.parse_args()


def pretrain_jepa(args, dataset: STARKPrimeDataset) -> str:
    """
    JEPA pretraining on STARK-Prime KG.
    
    Returns path to saved encoder.
    """
    print("\n" + "="*60)
    print("PHASE 1: JEPA PRETRAINING")
    print("="*60)
    
    # Get KG data
    kg_data = dataset.get_jepa_pretraining_data()
    print(f"Knowledge Graph: {kg_data.num_nodes} nodes, {kg_data.edge_index.size(1)} edges")
    print(f"Node embedding dim: {kg_data.x.size(1)}")
    
    # Configure masking
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
    
    # For large KG, sample subgraphs
    from jepa_graph.data.dataset import SubgraphSampler, GraphJEPADataset, JEPACollater
    
    sampler = SubgraphSampler(
        num_hops=args.num_hops,
        max_nodes=256,
        sample_strategy="random_walk",
    )
    
    # Create training samples by sampling subgraphs
    print("Sampling subgraphs for pretraining...")
    train_graphs = []
    num_samples = min(5000, kg_data.num_nodes // 10)
    
    for i in tqdm(range(num_samples), desc="Sampling"):
        anchor = torch.randint(0, kg_data.num_nodes, (1,)).item()
        subgraph = sampler.sample(kg_data, anchor_node=anchor)
        if subgraph.num_nodes >= 10:  # Filter tiny subgraphs
            train_graphs.append(subgraph)
    
    print(f"Created {len(train_graphs)} training subgraphs")
    
    # Wrap in JEPA dataset
    class SimpleGraphList:
        def __init__(self, graphs):
            self.graphs = graphs
        def __len__(self):
            return len(self.graphs)
        def __getitem__(self, idx):
            return self.graphs[idx]
    
    jepa_dataset = GraphJEPADataset(
        base_dataset=SimpleGraphList(train_graphs),
        masker=masker,
        num_masks_per_graph=1,
    )
    
    train_loader = DataLoader(
        jepa_dataset,
        batch_size=args.pretrain_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        collate_fn=JEPACollater(),
    )
    
    # Build model
    in_channels = kg_data.x.size(1)
    
    config = GraphJEPAConfig(
        encoder_type=args.encoder_type,
        in_channels=in_channels,
        hidden_channels=args.hidden_dim,
        out_channels=args.hidden_dim,
        num_encoder_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    
    model = config.build_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training config
    training_config = TrainingConfig(
        max_epochs=args.pretrain_epochs,
        batch_size=args.pretrain_batch_size,
        learning_rate=args.pretrain_lr,
        device=args.device,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
    )
    
    # Train
    trainer = JEPATrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
    )
    
    results = trainer.train()
    
    # Save encoder
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    encoder = model.get_encoder_for_downstream(use_target=True)
    encoder_path = output_path / "jepa_pretrained_encoder.pt"
    torch.save(encoder.state_dict(), encoder_path)
    
    # Save config
    config_path = output_path / "encoder_config.pt"
    torch.save({
        'in_channels': in_channels,
        'hidden_channels': args.hidden_dim,
        'out_channels': args.hidden_dim,
        'num_layers': args.num_layers,
        'gnn_type': args.encoder_type,
        'heads': args.num_heads,
        'dropout': args.dropout,
    }, config_path)
    
    print(f"\nJEPA pretraining complete!")
    print(f"Encoder saved to: {encoder_path}")
    
    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return str(encoder_path)


def finetune_gretriever(args, dataset: STARKPrimeDataset, encoder_path: str):
    """
    Finetune G-Retriever with JEPA encoder on STARK-Prime QA.
    """
    print("\n" + "="*60)
    print("PHASE 2: G-RETRIEVER FINETUNING")
    print("="*60)
    
    from jepa_graph.graphrag.gretriever import (
        create_gretriever_from_jepa,
        GRetrieverTrainer,
    )
    
    # Load encoder config
    config_path = Path(args.output_dir) / "encoder_config.pt"
    if config_path.exists():
        encoder_config = torch.load(config_path)
    else:
        # Default config
        encoder_config = {
            'in_channels': dataset.kg.x.size(1),
            'hidden_channels': args.hidden_dim,
            'out_channels': args.hidden_dim,
            'num_layers': args.num_layers,
            'gnn_type': args.encoder_type,
            'heads': args.num_heads,
            'dropout': args.dropout,
        }
    
    # Get QA splits
    print("Loading QA dataset splits...")
    splits = dataset.get_splits(
        num_hops=args.num_hops,
        k_nodes=args.k_nodes,
        fanout=args.fanout,
    )
    
    # Create data loaders
    train_loader = DataLoader(
        splits["train"],
        batch_size=args.finetune_batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        splits["validation"],
        batch_size=args.finetune_batch_size * 2,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        splits["test"],
        batch_size=args.finetune_batch_size * 2,
        shuffle=False,
        pin_memory=True,
    )
    
    # Create model
    print(f"Creating G-Retriever with JEPA encoder from {encoder_path}")
    model = create_gretriever_from_jepa(
        jepa_encoder_path=encoder_path,
        encoder_config=encoder_config,
        llm_model_name=args.llm_model,
        freeze_gnn=args.freeze_gnn,
        device=args.device,
    )
    
    # Train
    trainer = GRetrieverTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=args.finetune_lr,
        epochs=args.finetune_epochs,
        device=args.device,
    )
    
    model = trainer.train()
    
    # Evaluate
    print("\nFinal Evaluation...")
    eval_output = trainer.evaluate()
    
    # Compute metrics
    compute_stark_metrics(eval_output)
    
    # Save model
    model_path = Path(args.output_dir) / "gretriever_finetuned.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


def compute_stark_metrics(eval_output, skip_invalid_hit=True):
    """
    Compute STARK-Prime evaluation metrics.
    
    Matches baseline metrics.
    """
    import re
    import pandas as pd
    
    df = pd.concat([pd.DataFrame(d) for d in eval_output])
    
    all_hit = []
    all_exact_hit_at_1 = []
    all_exact_hit_at_5 = []
    all_precision = []
    all_recall = []
    all_mrr = []
    all_f1 = []
    
    for pred, label in zip(df.pred.tolist(), df.label.tolist()):
        # Parse predictions
        if isinstance(pred, str):
            pred = pred.split('[/s]')[0].strip().split('|')
        elif isinstance(pred, list):
            pred = [str(p).split('[/s]')[0].strip() for p in pred]
        
        # Check substring hit
        try:
            hit = re.findall(pred[0], label) if pred else []
        except Exception:
            if skip_invalid_hit:
                continue
            hit = []
        
        all_hit.append(len(hit) > 0)
        
        # Parse label
        label_set = set(label.split('|'))
        pred_set = set(pred)
        
        # Exact hit metrics
        exact_hit_at_1 = 1 * (pred[0] in label_set) if pred else 0
        exact_hit_at_5 = 1 * (len(pred_set & label_set) > 0) if len(pred) <= 5 else \
                         1 * (len(set(pred[:5]) & label_set) > 0)
        
        # Precision/Recall
        matches = pred_set & label_set
        precision = len(matches) / len(pred_set) if pred_set else 0
        recall = len(matches) / len(label_set) if label_set else 0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # MRR
        mrr = 0
        for i, node in enumerate(pred):
            if node in label_set:
                mrr = 1 / (i + 1)
                break
        
        all_exact_hit_at_1.append(exact_hit_at_1)
        all_exact_hit_at_5.append(exact_hit_at_5)
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        all_mrr.append(mrr)
    
    n = len(df.label.tolist())
    
    print("\n" + "="*40)
    print("STARK-Prime Evaluation Results")
    print("="*40)
    print(f"F1:              {sum(all_f1)/n:.4f}")
    print(f"Precision:       {sum(all_precision)/n:.4f}")
    print(f"Recall:          {sum(all_recall)/n:.4f}")
    print(f"Substring hit@1: {sum(all_hit)/n:.4f}")
    print(f"Exact hit@1:     {sum(all_exact_hit_at_1)/n:.4f}")
    print(f"Exact hit@5:     {sum(all_exact_hit_at_5)/n:.4f}")
    print(f"MRR:             {sum(all_mrr)/n:.4f}")


def main():
    args = parse_args()
    
    print("="*60)
    print("GRAPH JEPA + G-RETRIEVER on STARK-PRIME")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    
    # Initialize dataset
    dataset = STARKPrimeDataset(
        dataset_name=args.dataset,
        cache_dir=args.data_dir,
        device=args.device,
    )
    
    # Phase 1: JEPA Pretraining
    if args.encoder_path:
        encoder_path = args.encoder_path
        print(f"\nUsing provided encoder: {encoder_path}")
    elif args.skip_pretrain:
        encoder_path = os.path.join(args.output_dir, "jepa_pretrained_encoder.pt")
        if not os.path.exists(encoder_path):
            raise ValueError(f"No encoder found at {encoder_path}. Run pretraining first.")
    else:
        encoder_path = pretrain_jepa(args, dataset)
    
    # Phase 2: G-Retriever Finetuning
    if not args.skip_finetune:
        finetune_gretriever(args, dataset, encoder_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
