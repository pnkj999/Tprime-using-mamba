# Tprime-using-mamba
T-PRIME Wireless Protocol Classification using Mamba2

This repository implements a Mamba2-based replacement for the Transformer model in the T-PRIME protocol classification framework.

The project explores whether State Space Models (SSMs)—specifically Mamba2—can effectively model raw RF I/Q time-series signals for wireless protocol classification, while avoiding the computational cost of attention.

🧠 Motivation

The original T-PRIME pipeline uses Transformer-style sequence modeling to classify wireless protocols from raw baseband signals.

However:

Attention has quadratic complexity

RF signals can be very long sequences

Temporal dependencies are often better modeled with state-space dynamics

This project replaces the Transformer entirely with Mamba2, a linear-time SSM, while keeping the rest of the T-PRIME data pipeline intact.

📊 Classification Task

The model performs multi-class classification of Wi-Fi protocols:

802_11ax

802_11b_upsampled

802_11n

802_11g

Input data consists of raw complex baseband signals.

🏗️ Model Overview

Input

Complex I/Q samples

Converted to interleaved real-valued sequences

Segmented into (sequence_length × slice_length) windows

Architecture

Dynamic input projection

Stack of Mamba2 layers

Global average pooling over time

MLP classifier with LogSoftmax

Important design choice

❌ No attention

❌ No CNN front-end

❌ No token pooling tricks

✅ Pure Mamba2 temporal modeling

📁 Repository Structure
.
├── mamba2_model.py                 # Core Mamba2 classifier
├── TPrime_mamba2_train.py          # Training script
├── test_mamba2_classification.py   # Dataset-level evaluation
├── test_mamba2_single.py           # Single-signal inference
└── preprocessing/
    └── TPrime_dataset_python_only.py

📄 File Descriptions
mamba2_model.py

Defines the Mamba2-based sequence classifier.

Key features:

Dynamically adapts to input feature dimension

Stack of Mamba2 SSM layers

Mean pooling over time

Lightweight MLP classification head

TPrime_mamba2_train.py

End-to-end training pipeline.

Supports:

Training from scratch

Resume from checkpoint

Learning rate scheduling

Gradient clipping

Confusion matrix logging

Optional Ray-based distributed training

test_mamba2_classification.py

Evaluates a trained model on a full test dataset.

Outputs:

Overall accuracy

Per-protocol accuracy

Confusion matrix

Classification report

Inference throughput

test_mamba2_single.py

Lightweight script for single-signal inference.

Useful for:

Debugging

Quick qualitative evaluation

Inspecting confidence scores
