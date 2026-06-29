# Unbalanced Optimal Transport for Efficient Visual Document Retrieval

## Associated Paper

Our paper has been accepted to **The 19th European Conference on Computer Vision (ECCV) 2026**.

## Introduction

This repository provides a training-free visual token compression framework for efficient OCR-free Visual Document Retrieval (VDR).

Our method formulates document-side visual token compression as an **Unbalanced Optimal Transport (UOT)** problem, enabling scalable multi-vector retrieval while preserving fine-grained document semantics.

The framework is designed to compress document visual tokens extracted from multi-vector VDR backbones such as **ColPali**, **ColQwen2**, and **ColQwen2.5**, while maintaining retrieval effectiveness under aggressive compression ratios.

## Dataset Preparation

We assume that each VDR benchmark follows a structure similar to:

```text
dataset/
├── test/
│   ├── images/
│   │   ├── sample_0001.jpg
│   │   ├── sample_0002.jpg
│   │   └── ...
│   └── test.json
```

Each benchmark should contain document images and query-document relevance annotations.

Before running UOT-based compression, document embeddings, query embeddings, image-token masks, and attention scores should be extracted from the selected VDR backbone.

The preprocessing script `preprocess_colqwen.py` converts the extracted `.npz` files into the format required by the UOT compression pipeline.

### Input `.npz` Format

The preprocessing script expects each input file to contain the following fields:

```text
documents      : document-side token embeddings
query          : query-side token embeddings
docid          : document IDs
qid            : query IDs
attention      : vision attention scores
doc_imgmask    : valid image-token mask for each document
relevant_docs  : query-to-relevant-document annotations
```

### Preprocessed Output Format

After running `preprocess_colqwen.py`, each dataset is saved as a structured `.npz` file containing:

```text
dataset
├── task
├── model
├── documents
│   ├── features      # (num_docs, max_doc_tokens, embed_dim)
│   ├── attention     # (num_docs, max_doc_tokens)
│   ├── pad_masks     # (num_docs, max_doc_tokens)
│   └── max_tokens
└── queries
    ├── features      # (num_queries, max_query_tokens, embed_dim)
    ├── pad_masks     # (num_queries, max_query_tokens)
    ├── max_tokens
    └── relevant_doc_indices
```

During preprocessing, the script performs the following operations:

1. Removes duplicated document entries based on document IDs.
2. Removes system tokens from document-side embeddings and attention scores.
3. Pads document and query token sequences to the maximum sequence length within each dataset.
4. Creates padding masks for valid document and query tokens.
5. Converts relevant document IDs into internal document indices.
6. Saves the processed dataset into a UOT-compatible `.npz` file.

## Usage

### 1. Prepare VDR Benchmark Data

Prepare document images and query annotations according to the benchmark format.

Example:

```text
vidore_v1/
└── test/
    ├── images/
    │   └── sample.jpg
    └── test.json
```

### 2. Extract Embeddings and Attention Scores

Extract document embeddings, query embeddings, and attention scores using a VDR backbone.

Supported backbones include:

- `ColPali`
- `ColQwen2`
- `ColQwen2.5`

The extracted files are expected to be saved as `.npz` files, for example:

```text
colqwen2_5/docvqa_test_subsampled_dump_all.npz
colqwen2_5/infovqa_test_subsampled_dump_all.npz
colqwen2_5/tabfquad_test_subsampled_dump_all.npz
colqwen2_5/tatdqa_test_dump_all.npz
```

### 3. Run Preprocessing

Run the preprocessing script to convert the extracted features into the UOT-compatible format.

```bash
python preprocess_colqwen.py
```

Example configuration inside `preprocess_colqwen.py`:

```python
MODEL = "colqwen2_5"
NUM_SYS_TOKENS = 4
OUTPUT_DIR = "data"

DATASETS = [
    "docvqa",
    "infovqa",
    "tabfquad",
    "tatdqa",
]

INPUT_PATHS = [
    f"{MODEL}/{DATASETS[0]}_test_subsampled_dump_all.npz",
    f"{MODEL}/{DATASETS[1]}_test_subsampled_dump_all.npz",
    f"{MODEL}/{DATASETS[2]}_test_subsampled_dump_all.npz",
    f"{MODEL}/{DATASETS[3]}_test_dump_all.npz",
]
```

The processed files will be saved as:

```text
data/colqwen2_5_docvqa.npz
data/colqwen2_5_infovqa.npz
data/colqwen2_5_tabfquad.npz
data/colqwen2_5_tatdqa.npz
```

### 4. Run UOT-based Compression

After preprocessing, run UOT compression with the desired compression ratio.

```bash
python run_abl_uot.py
```

The script compresses document-side visual tokens according to the specified compression setting and evaluates retrieval performance on the target VDR benchmark.

## Repository Pipeline

```text
Raw VDR Benchmark
        │
        ▼
Embedding & Attention Extraction
        │
        ▼
preprocess_colqwen.py
        │
        ▼
UOT-compatible .npz files
        │
        ▼
run_abl_uot.py
        │
        ▼
Compressed Document Tokens + Retrieval Evaluation
```
