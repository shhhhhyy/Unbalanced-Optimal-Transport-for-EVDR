import numpy as np

MODEL = 'colqwen2_5'
NUM_SYS_TOKENS = 4
#DATASETS = ['arxivqa', 'docvqa', 'infovqa', 'tabfquad', 'tatdqa', 'shiftproject', 'artificial_intelligence','energy', 'government_reports', 'healthcare_industry']
OUTPUT_DIR = 'data'
# DATASETS = ['computer_science','finance_en','hr','industrial','pharmaceuticals','finance_fr','physics','energy']
# DATASETS = ['finance_fr','physics','energy']
DATASETS = ['docvqa', 'infovqa', 'tabfquad', 'tatdqa']


# INPUT_PATHES = [
#     f'{MODEL}/{DATASETS[0]}_dump_all.npz']


# INPUT_PATHES = [
#     f'{MODEL}/vidore_v3_{DATASETS[0]}_dump_all.npz',
#     f'{MODEL}/vidore_v3_{DATASETS[1]}_dump_all.npz',
#     f'{MODEL}/vidore_v3_{DATASETS[2]}_dump_all.npz',
#     f'{MODEL}/vidore_v3_{DATASETS[3]}_dump_all.npz']

INPUT_PATHES = [
    f'{MODEL}/{DATASETS[0]}_test_subsampled_dump_all.npz',
    f'{MODEL}/{DATASETS[1]}_test_subsampled_dump_all.npz',
    f'{MODEL}/{DATASETS[2]}_test_subsampled_dump_all.npz',
    f'{MODEL}/{DATASETS[3]}_test_dump_all.npz']
#     f'{MODEL}/{DATASETS[4]}_test_dump_all.npz',
#     f'{MODEL}/{DATASETS[5]}_test_dump_all.npz',
#     f'{MODEL}/syntheticDocQA_{DATASETS[6]}_test_dump_all.npz',
#     f'{MODEL}/syntheticDocQA_{DATASETS[7]}_test_dump_all.npz',
#     f'{MODEL}/syntheticDocQA_{DATASETS[8]}_test_dump_all.npz',
#     f'{MODEL}/syntheticDocQA_{DATASETS[9]}_test_dump_all.npz']



for n in range(len(INPUT_PATHES)):
    # ------------------------------------------------------------------
    # 데이터 로드
    # ------------------------------------------------------------------
    data = np.load(INPUT_PATHES[n], allow_pickle=True)

    # ------------------------------------------------------------------
    # 문서 전처리
    # ------------------------------------------------------------------
    raw_doc_embeds = data['documents']
    raw_doc_ids = data['docid']
    raw_attn_scores = data['attention']
    doc_imgmask = data['doc_imgmask']

    # 문서 중복 및 시스템 토큰 제거
    unique_doc_ids = []      # 유니크한 문서 ID 순서대로 저장
    unique_doc_embeds = []   # 유니크한 문서 임베딩 저장
    unique_attn_scores = []  # 유니크한 어텐션 스코어 저장
    docid_to_new_idx = {}    # Doc ID -> 새로운 유니크 인덱스 (0, 1, 2...) 매핑 생성
    num_raw_docs = raw_doc_embeds.shape[0]

    for i in range(num_raw_docs):
        d_id = raw_doc_ids[i]
        
        # 이미 처리한 문서 ID라면 스킵 (중복 제거)
        if d_id in docid_to_new_idx:
            continue

        # 처음 보는 문서라면: 새 인덱스 부여 & 임베딩 전처리
        new_idx = len(unique_doc_ids)
        docid_to_new_idx[d_id] = new_idx
        unique_doc_ids.append(d_id)

        num_image_tokens = doc_imgmask[i].sum()

        # 임베딩 시스템 토큰 제거
        processed_embed = raw_doc_embeds[i][NUM_SYS_TOKENS:NUM_SYS_TOKENS+num_image_tokens]
        unique_doc_embeds.append(processed_embed)

        # 어텐션 스코어 시스템 토큰 제거
        processed_attn = raw_attn_scores[i][NUM_SYS_TOKENS:NUM_SYS_TOKENS+num_image_tokens]
        unique_attn_scores.append(processed_attn)

    num_unique_docs = len(unique_doc_embeds)


    # 문서 토큰 수 계산 및 최대 토큰 수 업데이트
    embed_dim = unique_doc_embeds[0].shape[1]
    max_doc_tokens = max(len(doc) for doc in unique_doc_embeds)

    # Zero Padding 및 Padding Mask 생성
    padded_docs = np.zeros((num_unique_docs, max_doc_tokens, embed_dim), dtype=np.float32)
    padded_attn = np.zeros((num_unique_docs, max_doc_tokens), dtype=np.float32)
    doc_pad_mask = np.zeros((num_unique_docs, max_doc_tokens), dtype=np.float32)

    # 각 문서를 순회하며 값 채워넣기
    for idx, (doc, attn) in enumerate(zip(unique_doc_embeds, unique_attn_scores)):
        length = doc.shape[0]
        padded_docs[idx, :length, :] = doc
        padded_attn[idx, :length] = attn.flatten()
        doc_pad_mask[idx, :length] = 1.0
    
    # 문서 정보 재구조화
    documents = {
        'features': padded_docs,  # (num_docs, max_doc_tokens, embed_dim)
        'attention': padded_attn,  # (num_docs, max_doc_tokens)
        'pad_masks': doc_pad_mask,  # (num_docs, max_doc_tokens)
        'max_tokens': max_doc_tokens,  # 최대 문서 토큰 수
    }

    # ------------------------------------------------------------------
    # 질의 전처리
    # ------------------------------------------------------------------
    query_embed = data['query']
    query_id = data['qid']
    num_queries = query_id.shape[0]
    max_query_tokens = max(query_embed[i].shape[0] for i in range(num_queries)) 

    # Zero Padding 및 Padding Mask 생성
    padded_queries = np.zeros((num_queries, max_query_tokens, embed_dim), dtype=np.float32)
    query_pad_mask = np.zeros((num_queries, max_query_tokens), dtype=np.float32)
    for q_idx, q in enumerate(query_embed):
        length = q.shape[0]
        padded_queries[q_idx, :length, :] = q
        query_pad_mask[q_idx, :length] = 1.0 

    # ------------------------------------------------------------------
    # 정답 전처리
    # ------------------------------------------------------------------
    relevant_docs = data['relevant_docs'].item()
    relevant_doc_indices = np.zeros(num_queries, dtype=np.int32)
    for idx, qid in enumerate(query_id):
        # 해당 쿼리의 정답 문서 ID 가져오기
        q_rels = relevant_docs[qid]
        target_doc_id = list(q_rels.keys())[0]
        
        # 새로 만든 유니크 인덱스로 변환
        relevant_doc_indices[idx] = docid_to_new_idx[target_doc_id]

    # 질의 정보 재구조화    
    queries = {
        'features': padded_queries,  # (num_queries, max_query_tokens, embed_dim
        'pad_masks': query_pad_mask,  # (num_queries, max_query_tokens)
        'max_tokens': max_query_tokens,  # 최대 질의 토큰 수
        'relevant_doc_indices': relevant_doc_indices,  # (num_queries,) 각 질의에 대한 정답 문서 인덱스
    }

    # ------------------------------------------------------------------
    # 데이터세트 딕셔너리
    # -------------------------------------------------------------------
    dataset = {
        'task': DATASETS[n],
        'model': MODEL,
        'documents': documents,
        'queries': queries,
    }

    # ------------------------------------------------------------------
    # 데이터세트 저장
    # ------------------------------------------------------------------
    if 'multi' in INPUT_PATHES[n]:
        output_path = f'{OUTPUT_DIR}/{MODEL}_multi_{DATASETS[n]}.npz'
    else: output_path = f'{OUTPUT_DIR}/{MODEL}_{DATASETS[n]}.npz'
    np.savez(output_path, **dataset)
    print(output_path,' 저장 완료')
