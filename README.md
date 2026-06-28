# Unbalanced-Optimal-Transport-for-EVDR

Official implementation of Unbalanced Optimal Transport for Efficient Visual Document Retrieval.

This repository provides a training-free visual token compression framework for efficient OCR-free Visual Document Retrieval (VDR). Our method formulates document-side visual token compression as an Unbalanced Optimal Transport (UOT) problem, enabling scalable multi-vector retrieval while preserving fine-grained document semantics.

### Key Idea

Effective visual token compression for document retrieval requires balancing three objectives:

1. Semantic redundancy reduction
    Similar visual tokens should be merged to reduce unnecessary token duplication.
2. Token importance preservation
    Informative document regions should receive higher priority during compression.
3. Coverage capacity regulation
    Representative tokens should avoid collapsing too many distinct visual regions into a single token.

Our method unifies these objectives in a single Unbalanced Optimal Transport formulation. The transport cost captures semantic redundancy, the source marginal reflects token importance, and the target marginal regulates representative-token capacity.

### Installation
git clone https://github.com/your-username/uot-for-evdr.git
cd uot-for-evdr

conda create -n uot-evdr python=3.10
conda activate uot-evdr

pip install -r requirements.txt

### Dataset Preparation

We evaluate our method on ViDoRe benchmarks.

Expected dataset structure:
  data/
  ├── vidore_v1/
  ├── vidore_v3/
  └── vidore_v3_multi/

### Usage

1. Extract document embeddings
