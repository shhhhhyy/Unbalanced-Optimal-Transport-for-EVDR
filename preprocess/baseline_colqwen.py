import numpy as np
import torch
import os

from utils import ndcg_k, recall_at_k, maxsim_scores,ndcg_k_graded_matrix,recall_at_k_fraction,recall_at_k_multi


# 설정
MODEL = 'colqwen'

# DATASETS = ['arxivqa', 'docvqa', 'infovqa', 'tabfquad', 'tatdqa', 'shiftproject', 'artificial_intelligence','energy', 'government_reports', 'healthcare_industry']
#DATASETS = ['computer_science','finance_en','hr','industrial','pharmaceuticals']#,'industrial','pharmaceuticals']
#DATASETS = ['mathvista']#['mmlongbench_doc'] #slidevqa
DATASETS = ['biomedical_lectures', 'economics_reports','esg_reports','esg_reports_human_labeled']#,'multi_finance_en','multi_hr','multi_industrial']#,'multi_pharmaceuticals','multi_finance_fr','multi_physics','multi_energy']
OUTPUT_DIR = 'data'
DEVICE = 'cuda' #'cuda' if torch.cuda.is_available() else 'cpu'
NDCG_K = 5  # 평가할 k 값들
REACALL_K = 1

print(f"Using device: {DEVICE}")

for dataset_name in DATASETS:
    file_path = f'{OUTPUT_DIR}/{MODEL}_{dataset_name}.npz'
    
    if not os.path.exists(file_path):
        print(f"Skipping {dataset_name}: File not found at {file_path}")
        continue

    print(f"\n========================================")
    print(f"Evaluating Dataset: {dataset_name}")

    # 1. 데이터 로드
    # np.savez로 저장된 딕셔너리를 로드할 때는 allow_pickle=True가 필요하며,
    # 내부 딕셔너리는 .item()을 통해 꺼내야 합니다.
    data = np.load(file_path, allow_pickle=True)
    
    # documents와 queries는 0-d object array로 저장되므로 item()으로 추출
    docs_data = data['documents'].item()
    queries_data = data['queries'].item()
    
    # 2. Tensor 변환 및 Device 이동 (GPU 가속을 위해)
    # utils.maxsim_scores 함수는 PyTorch Tensor를 입력으로 받습니다.
    
    # Document Features
    d_embeds = torch.from_numpy(docs_data['features']).to(DEVICE)
    d_masks = torch.from_numpy(docs_data['pad_masks']).to(DEVICE)
    
    # Query Features
    q_embeds = torch.from_numpy(queries_data['features']).to(DEVICE)
    q_masks = torch.from_numpy(queries_data['pad_masks']).to(DEVICE)
    
    

    # 3. Score 계산 (ColBERT MaxSim)
    with torch.no_grad():
        q_embeds = torch.nn.functional.normalize(q_embeds, p=2, dim=-1)
        d_embeds = torch.nn.functional.normalize(d_embeds, p=2, dim=-1)
        scores_tensor = maxsim_scores(q_embeds, d_embeds, q_masks, d_masks)
        print(scores_tensor.shape)
    
    # 평가를 위해 CPU/Numpy로 변환
    scores_numpy = scores_tensor.cpu().numpy()

    # 4. 성능 평가 (nDCG@k, Recall@k)
    # not V3
    # relevant_doc_indices = queries_data['relevant_doc_indices']
    # ndcg_score_10 = ndcg_k(scores_numpy, relevant_doc_indices, 10)
    # ndcg_score = ndcg_k(scores_numpy, relevant_doc_indices, NDCG_K)
    # recall_score = recall_at_k(scores_numpy, relevant_doc_indices, REACALL_K)

    # for V3
    targets = list(queries_data['relevant_doc_rels'])  # object array -> list[dict]

    #for mmlongbench
    # relevant_doc_rels = queries_data['relevant_doc_rels']

    # targets = []
    # for doc_idxs, rels in zip(relevant_doc_indices, relevant_doc_rels):
    #     target_dict = {int(doc_idx): int(rel) for doc_idx, rel in zip(doc_idxs, rels)}
    #     targets.append(target_dict)

    # 점수 계산
    ndcg_score_10 = ndcg_k_graded_matrix(scores_numpy, targets, k=10)
    ndcg_score = ndcg_k_graded_matrix(scores_numpy, targets, k=5)
    
    recall_score = recall_at_k_multi(scores_numpy, targets, REACALL_K)
    
    print(f"nDCG@10: {ndcg_score_10:.4f} | nDCG@5: {ndcg_score:.4f} | Recall@1: {recall_score:.4f}")

print("\nEvaluation Finished.")