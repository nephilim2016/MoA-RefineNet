#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 20:09:37 2025

@author: nephilim
"""


"""
visualize_router_weights.py

Usage (from shell):
  python visualize_router_weights.py --log_dir ./router_w_logs --epochs 016,017,018 --out ./router_figs

What it does:
- Loads batch-wise router weights files like router_w_e016_b0003.npy (shape [B,E]).
- For each epoch, stacks all batches along sample dimension (N = sum of batch sizes), producing W_epoch [N, E].
- Plots:
  1) Heatmap of W_epoch (optionally downsampled to top N_show samples).
  2) top1 curve (mean of max probability) across selected epochs.
  3) entropy curve across selected epochs.
"""

import os, glob, argparse
import numpy as np
import matplotlib.pyplot as plt

def load_epoch_matrix(log_dir: str, epoch: int):
    patt = os.path.join(log_dir, f"router_w_stage1_e{epoch:03d}_b*.npy")
    files = sorted(glob.glob(patt))
    if not files:
        return None
    mats = []
    for p in files:
        w = np.load(p)  # [B, E]
        # Filter any degenerate batches
        if w.ndim == 2 and w.shape[0] > 0 and w.shape[1] > 0:
            mats.append(w)
    if not mats:
        return None
    W = np.concatenate(mats, axis=0)  # [N, E]
    return W

def entropy_rows(W, eps=1e-8):
    Wc = np.clip(W, eps, 1.0)
    H = -np.sum(Wc * np.log(Wc), axis=1)   # [N]
    return H

def plot_heatmap(W, out_png, title="", max_rows=512):
    # Downsample rows if too many
    if W.shape[0] > max_rows:
        idx = np.linspace(0, W.shape[0]-1, max_rows).astype(int)
        W_show = W[idx]
    else:
        W_show = W
    plt.figure(figsize=(6, 6))
    plt.imshow(W_show, aspect='auto')
    plt.xlabel("Expert index")
    plt.ylabel("Sample index")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=str, required=True, help="Directory containing router_w_eXXX_bYYYY.npy files")
    ap.add_argument("--epochs", type=str, required=True, help="Comma-separated epoch numbers, e.g., 016,017,018")
    ap.add_argument("--out", type=str, default="./router_figs", help="Output directory for figures")
    ap.add_argument("--max_rows", type=int, default=512, help="Max rows to show in heatmap")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    epoch_list = [int(s) for s in args.epochs.split(",")]
    top1_curve, ent_curve = [], []
    valid_epochs = []

    for ep in epoch_list:
        W = load_epoch_matrix(args.log_dir, ep)
        if W is None:
            print(f"[warn] no data for epoch {ep}")
            continue
        valid_epochs.append(ep)

        # Heatmap per-epoch
        out_png = os.path.join(args.out, f"router_heat_e{ep:03d}.png")
        plot_heatmap(W, out_png, title=f"Router weights heatmap (epoch {ep})", max_rows=args.max_rows)

        # Scalar metrics
        top1 = np.mean(np.max(W, axis=1))
        H = np.mean(entropy_rows(W))
        top1_curve.append(top1)
        ent_curve.append(H)

        print(f"[epoch {ep}] N={W.shape[0]} E={W.shape[1]}  top1={top1:.3f}  entropy={H:.3f}  -> {out_png}")

    # Summary curves
    if valid_epochs:
        xs = np.arange(len(valid_epochs))
        # top1
        plt.figure(figsize=(5,3))
        plt.plot(xs, top1_curve, marker='o')
        plt.xticks(xs, [f"{e}" for e in valid_epochs], rotation=0)
        plt.xlabel("Epoch")
        plt.ylabel("top1 (mean max weight)")
        plt.title("Router top1 across epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "router_top1_curve.png"), dpi=150)
        plt.close()

        # entropy
        plt.figure(figsize=(5,3))
        plt.plot(xs, ent_curve, marker='o')
        plt.xticks(xs, [f"{e}" for e in valid_epochs], rotation=0)
        plt.xlabel("Epoch")
        plt.ylabel("Mean entropy")
        plt.title("Router entropy across epochs")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "router_entropy_curve.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    main()