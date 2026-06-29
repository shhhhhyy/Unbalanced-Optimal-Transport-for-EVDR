# Unbalanced-Optimal-Transport-for-EVDR

Official implementation of Unbalanced Optimal Transport for Efficient Visual Document Retrieval.

This repository provides a training-free visual token compression framework for efficient OCR-free Visual Document Retrieval (VDR). Our method formulates document-side visual token compression as an Unbalanced Optimal Transport (UOT) problem, enabling scalable multi-vector retrieval while preserving fine-grained document semantics.



### Dataset Preparation

We evaluate our method on ViDoRe benchmarks.

Expected dataset structure:
  data/
  
  ├── vidore_v1/
  
  ├── vidore_v3/
  
  └── vidore_v3_multi/

### Usage

1. Extract document embeddings
