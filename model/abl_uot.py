import torch
import torch.nn.functional as F

def init_standard_kmeans_pp(valid_x, num_clusters):
    """ [Ablation 4, 5용] 표준 K-Means++ 초기화 (기하학적 거리만 사용) """
    N, D = valid_x.shape
    device = valid_x.device
    centroids = torch.zeros((num_clusters, D), device=device, dtype=valid_x.dtype)
    
    idx = torch.randint(0, N, (1,), device=device)
    centroids[0] = valid_x[idx]
    min_dists_sq = torch.full((N,), float('inf'), device=device)
    
    for k in range(1, num_clusters):
        prev_centroid = centroids[k-1].unsqueeze(0)
        dist_sq = 2.0 * (1.0 - (valid_x @ prev_centroid.T).squeeze())
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        min_dists_sq = torch.min(min_dists_sq, dist_sq)
        probs = min_dists_sq.clone()
        if probs.sum() <= 1e-12: probs = torch.ones(N, device=device)
            
        next_idx = torch.multinomial(probs, 1)
        centroids[k] = valid_x[next_idx]
        
    return centroids

def init_mass_weighted_kmeans_pp(valid_x, r, num_clusters):
    """ [Ablation 6용] 제안하는 Mass-Weighted K-Means++ 초기화 """
    N, D = valid_x.shape
    device = valid_x.device
    centroids = torch.zeros((num_clusters, D), device=device, dtype=valid_x.dtype)
    
    idx = torch.multinomial(r, 1)
    centroids[0] = valid_x[idx]
    
    min_dists_sq = torch.full((N,), float('inf'), device=device)
    current_r = r.clone(); current_r[idx] = 0 
    
    for k in range(1, num_clusters):
        prev_centroid = centroids[k-1].unsqueeze(0)
        dist_sq = 2.0 * (1.0 - (valid_x @ prev_centroid.T).squeeze())
        dist_sq = torch.clamp(dist_sq, min=0.0)
        
        min_dists_sq = torch.min(min_dists_sq, dist_sq)
        probs = min_dists_sq * current_r # [핵심] 거리 * 어텐션 질량
        
        if probs.sum() <= 1e-12:
            probs = current_r.clone()
            probs[probs < 0] = 0
            
        next_idx = torch.multinomial(probs, 1)
        centroids[k] = valid_x[next_idx]
        current_r[next_idx] = 0
        
    return centroids

def sinkhorn_solver(dist_matrix, r, c, gamma, n_iters=3, rho=10.0, is_unbalanced=True):
    """
    통합 Sinkhorn Solver (Balanced / Unbalanced 전환 가능)
    """
    log_r = torch.log(r + 1e-9)
    log_c = torch.log(c + 1e-9)
    log_u = torch.zeros_like(log_r)
    log_v = torch.zeros_like(log_c)
    
    log_K_mat = -dist_matrix / gamma
    
    # [핵심] fi(Scaling Factor)가 1.0이면 Balanced OT (Hard Constraint)
    # fi < 1.0 이면 Unbalanced OT (Soft Constraint)
    fi = (rho / (rho + gamma)) if is_unbalanced else 1.0
    
    for _ in range(n_iters):
        t1 = log_K_mat + log_v.unsqueeze(0)
        log_u = fi * (log_r - torch.logsumexp(t1, dim=1))
        
        t2 = log_K_mat + log_u.unsqueeze(1)
        log_v = fi * (log_c - torch.logsumexp(t2, dim=0))
        
    log_P = log_u.unsqueeze(1) + log_K_mat + log_v.unsqueeze(0)
    return torch.exp(log_P)

def ot_ablation_baselines(x, attention, mask, num_clusters, 
                          ablation_type=5, gamma=0.04, rho=10.0, n_iter=20, sinkhorn_iter=3):
    """
    Ablation 4~6 수행을 위한 통합 OT/UOT 베이스라인
    
    Args:
        ablation_type:
            4 = Balanced OT + 표준 K-Means++ 초기화
            5 = Unbalanced OT + 표준 K-Means++ 초기화  (제안 방법론 최종본) -> 진짜임
            6 = Unbalanced OT + 가중 K-Means++ 초기화 X(제안 방법론 최종본)X
    """
    valid_mask = (mask == 1)
    valid_x = x[valid_mask]         
    valid_attn = attention[valid_mask] 
    
    N, D = valid_x.shape
    device = x.device
    
    if N <= num_clusters:
        return F.normalize(valid_x, p=2, dim=-1)

    valid_x = F.normalize(valid_x, p=2, dim=-1)
    r = F.normalize(valid_attn + 1e-9, p=1, dim=0)
    c = torch.ones(num_clusters, device=device, dtype=valid_x.dtype) / num_clusters

    # ==========================================
    # 1. Initialization (Ablation 4, 5 vs 6)
    # ==========================================
    if ablation_type in [4, 5]:
        # [Ablation 4, 5] 어텐션을 무시한 표준 초기화
        centroids = init_standard_kmeans_pp(valid_x, num_clusters)
    else:
        # [Ablation 6] 어텐션과 거리를 동시에 고려한 초기화
        centroids = init_mass_weighted_kmeans_pp(valid_x, r, num_clusters)
        
    centroids = F.normalize(centroids, p=2, dim=-1)

    # ==========================================
    # 2. EM Loop
    # ==========================================
    for i in range(n_iter):
        old_centroids = centroids.clone()
        
        # --- E-Step (OT Routing) ---
        dist_matrix = 1.0 - (valid_x @ centroids.T)
        
        is_unbalanced = True if ablation_type in [5, 6] else False
        
        # [Ablation 4 vs 5, 6] Balanced / Unbalanced 스위칭
        P = sinkhorn_solver(dist_matrix, r, c, gamma, sinkhorn_iter, rho, is_unbalanced)
        
        # --- M-Step (Centroid Update) ---
        # 전송 행렬 P의 행 합(Row Sum)은 이미 r(Attention Mass)과 같거나 비례하도록 맞춰짐.
        # 따라서 P.T @ valid_x 연산 자체에 이미 "어텐션 가중 평균"이 자연스럽게 내포되어 있음!
        sum_features = P.T @ valid_x
        centroids = F.normalize(sum_features, p=2, dim=-1)
        
        if torch.norm(centroids - old_centroids, dim=1).max() < 1e-4:
            break
            
    return centroids


def save_ot_ablation(
    x,
    attention,
    mask,
    num_clusters,
    ablation_type=6,
    gamma=0.05,
    rho=10.0,
    n_iter=20,
    sinkhorn_iter=3,
    return_details=False,
):
    """
    Ablation 4~6 수행을 위한 통합 OT/UOT 베이스라인

    Args:
        ablation_type:
            4 = Balanced OT + 표준 K-Means++ 초기화
            5 = Unbalanced OT + 표준 K-Means++ 초기화
            6 = Unbalanced OT + 가중 K-Means++ 초기화 (제안 방법론 최종본)

        return_details:
            True이면 theorem 분석용 중간 결과(P, C, X, Y)를 함께 반환
    """
    valid_mask = (mask == 1)
    valid_x = x[valid_mask]
    valid_attn = attention[valid_mask]

    N, D = valid_x.shape
    device = x.device

    if N <= num_clusters:
        centroids = F.normalize(valid_x, p=2, dim=-1)
        if return_details:
            dist_matrix = 1.0 - (centroids @ centroids.T)
            dummy_P = torch.eye(centroids.shape[0], device=device, dtype=centroids.dtype)
            details = {
                "X": centroids.detach().cpu(),
                "Y": centroids.detach().cpu(),
                "C": dist_matrix.detach().cpu(),
                "P": dummy_P.detach().cpu(),
                "r": None,
                "c": None,
            }
            return centroids, details
        return centroids

    valid_x = F.normalize(valid_x, p=2, dim=-1)
    r = F.normalize(valid_attn + 1e-9, p=1, dim=0)
    c = torch.ones(num_clusters, device=device, dtype=valid_x.dtype) / num_clusters

    # 1. initialization
    if ablation_type in [4, 5]:
        centroids = init_standard_kmeans_pp(valid_x, num_clusters)
    else:
        centroids = init_mass_weighted_kmeans_pp(valid_x, r, num_clusters)

    centroids = F.normalize(centroids, p=2, dim=-1)

    last_P = None
    last_C = None

    # 2. EM loop
    for _ in range(n_iter):
        old_centroids = centroids.clone()

        dist_matrix = 1.0 - (valid_x @ centroids.T)
        is_unbalanced = ablation_type in [5, 6]

        P = sinkhorn_solver(
            dist_matrix=dist_matrix,
            r=r,
            c=c,
            gamma=gamma,
            n_iters=sinkhorn_iter,
            rho=rho,
            is_unbalanced=is_unbalanced,
        )

        sum_features = P.T @ valid_x
        centroids = F.normalize(sum_features, p=2, dim=-1)

        last_P = P
        last_C = dist_matrix

        if torch.norm(centroids - old_centroids, dim=1).max() < 1e-4:
            break

    if return_details:
        details = {
            "X": valid_x.detach().cpu(),
            "Y": centroids.detach().cpu(),
            "C": last_C.detach().cpu(),
            "P": last_P.detach().cpu(),
            "r": r.detach().cpu(),
            "c": c.detach().cpu(),
        }
        return centroids, details

    return centroids