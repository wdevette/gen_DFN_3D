# modules/orientation_clustering.py
"""
Módulo para clusterização de orientações de fraturas usando Fisher distribution
Adaptado do FRAMFRAT-DFN para integração com o sistema DFN 3D
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.cluster import KMeans

@dataclass
class FisherParams:
    """Parâmetros da distribuição de Fisher para uma família de fraturas"""
    mu: np.ndarray  # Vetor direção média
    kappa: float    # Parâmetro de concentração
    weight: int     # Número de fraturas nesta família

def normals_from_azimuth(orient_deg: np.ndarray) -> np.ndarray:
    """
    Converte orientações em graus para vetores normais 2D
    
    Args:
        orient_deg: Array de orientações em graus (0-360)
    
    Returns:
        Array de vetores normais normalizados (n x 3)
    """
    theta = np.deg2rad(orient_deg % 360)
    nx = np.cos(theta)
    ny = np.sin(theta)
    nz = np.zeros_like(nx)
    v = np.vstack([nx, ny, nz]).T
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v

def normals_from_dip_dipdir(dips: np.ndarray, dip_directions: np.ndarray) -> np.ndarray:
    """
    Converte dip e dip direction para vetores normais 3D
    
    Args:
        dips: Ângulos de mergulho em graus (0-90)
        dip_directions: Direções de mergulho em graus (0-360)
    
    Returns:
        Array de vetores normais normalizados (n x 3)
    """
    dip_rad = np.deg2rad(dips)
    dd_rad = np.deg2rad(dip_directions)
    
    nx = np.sin(dip_rad) * np.sin(dd_rad)
    ny = np.sin(dip_rad) * np.cos(dd_rad)
    nz = np.cos(dip_rad)
    
    v = np.vstack([nx, ny, nz]).T
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    return v

def fit_fisher_params(vectors: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Ajusta parâmetros da distribuição de Fisher para um conjunto de vetores
    
    Args:
        vectors: Array de vetores normalizados (n x 3)
    
    Returns:
        mu: Vetor direção média
        kappa: Parâmetro de concentração
    """
    R = np.linalg.norm(np.sum(vectors, axis=0))
    n = len(vectors)
    Rbar = R / max(n, 1)
    
    # Calcular kappa usando aproximações de Fisher
    if Rbar < 1e-8:
        kappa = 0.0
    elif Rbar < 0.53:
        kappa = 2*Rbar + Rbar**3 + 5*Rbar**5/6
    elif Rbar < 0.85:
        kappa = -0.4 + 1.39*Rbar + 0.43/(1-Rbar)
    else:
        kappa = 1/(Rbar**3 - 4*Rbar**2 + 3*Rbar)
    
    # Calcular direção média
    mu = np.sum(vectors, axis=0) / max(R, 1e-12)
    mu = mu / np.linalg.norm(mu)
    
    return mu, float(kappa)

def cluster_orientations_2d(orientations: np.ndarray, 
                           n_sets: int = 2) -> List[FisherParams]:
    """
    Clusteriza orientações 2D em famílias usando K-means
    
    Args:
        orientations: Array de orientações em graus
        n_sets: Número de famílias/sets a identificar
    
    Returns:
        Lista de FisherParams para cada família
    """
    # Converter para vetores normais
    V = normals_from_azimuth(orientations)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_sets, n_init=20, random_state=0).fit(V)
    
    params = []
    for s in range(n_sets):
        v = V[kmeans.labels_ == s]
        if len(v) == 0:
            continue
        
        mu, kappa = fit_fisher_params(v)
        params.append(FisherParams(mu=mu, kappa=kappa, weight=len(v)))
    
    return params

def cluster_orientations_3d(dips: np.ndarray, 
                           dip_directions: np.ndarray,
                           n_sets: int = 2) -> List[FisherParams]:
    """
    Clusteriza orientações 3D em famílias usando K-means
    
    Args:
        dips: Array de ângulos de mergulho em graus
        dip_directions: Array de direções de mergulho em graus
        n_sets: Número de famílias/sets a identificar
    
    Returns:
        Lista de FisherParams para cada família
    """
    # Converter para vetores normais
    V = normals_from_dip_dipdir(dips, dip_directions)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_sets, n_init=20, random_state=0).fit(V)
    
    params = []
    for s in range(n_sets):
        v = V[kmeans.labels_ == s]
        if len(v) == 0:
            continue
        
        mu, kappa = fit_fisher_params(v)
        params.append(FisherParams(mu=mu, kappa=kappa, weight=len(v)))
    
    return params

def extract_orientation_stats(fisher_params: List[FisherParams], 
                             dimension: str = '2d') -> List[dict]:
    """
    Extrai estatísticas de orientação das famílias identificadas
    
    Args:
        fisher_params: Lista de parâmetros Fisher
        dimension: '2d' ou '3d'
    
    Returns:
        Lista de dicionários com estatísticas de cada família
    """
    stats = []
    
    for i, fp in enumerate(fisher_params):
        if dimension == '2d':
            # Converter vetor normal de volta para azimute
            orientation = np.rad2deg(np.arctan2(fp.mu[1], fp.mu[0])) % 360
            
            # Estimar desvio padrão a partir de kappa
            # Para Fisher 2D: std ≈ sqrt(2/kappa) em radianos
            std_rad = np.sqrt(2 / max(fp.kappa, 0.01))
            std_deg = np.rad2deg(std_rad)
            
            stats.append({
                'family_id': i,
                'orientation_mean': orientation,
                'orientation_std': std_deg,
                'kappa': fp.kappa,
                'n_fractures': fp.weight,
                'percentage': 0  # Será calculado depois
            })
        else:  # 3d
            # Converter vetor normal de volta para dip e dip direction
            dip = np.rad2deg(np.arccos(fp.mu[2]))
            dip_dir = np.rad2deg(np.arctan2(fp.mu[0], fp.mu[1])) % 360
            
            # Estimar desvios padrão
            std_rad = np.sqrt(2 / max(fp.kappa, 0.01))
            std_deg = np.rad2deg(std_rad)
            
            stats.append({
                'family_id': i,
                'dip_mean': dip,
                'dip_std': std_deg,
                'dip_dir_mean': dip_dir,
                'dip_dir_std': std_deg,
                'kappa': fp.kappa,
                'n_fractures': fp.weight,
                'percentage': 0
            })
    
    # Calcular percentagens
    total = sum(s['n_fractures'] for s in stats)
    for s in stats:
        s['percentage'] = (s['n_fractures'] / total * 100) if total > 0 else 0
    
    return stats

def auto_determine_n_sets(orientations: np.ndarray, 
                         max_sets: int = 4,
                         dimension: str = '2d',
                         dip_directions: np.ndarray = None) -> int:
    """
    Determina automaticamente o número ótimo de famílias usando método do cotovelo
    
    Args:
        orientations: Array de orientações (azimutes para 2D, dips para 3D)
        max_sets: Número máximo de sets a testar
        dimension: '2d' ou '3d'
        dip_directions: Necessário se dimension='3d'
    
    Returns:
        Número ótimo de sets
    """
    if dimension == '2d':
        V = normals_from_azimuth(orientations)
    else:
        if dip_directions is None:
            raise ValueError("dip_directions necessário para análise 3D")
        V = normals_from_dip_dipdir(orientations, dip_directions)
    
    inertias = []
    K_range = range(1, min(max_sets + 1, len(V)))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=0).fit(V)
        inertias.append(kmeans.inertia_)
    
    # Método do cotovelo simplificado
    if len(inertias) < 2:
        return 2  # Default
    
    # Calcular diferenças
    diffs = np.diff(inertias)
    
    # Encontrar o "cotovelo" onde a melhoria diminui significativamente
    if len(diffs) > 1:
        diffs_2nd = np.diff(diffs)
        optimal_k = np.argmax(diffs_2nd) + 2  # +2 porque começamos em k=1 e diff perde índice
    else:
        optimal_k = 2
    
    return min(optimal_k, max_sets)
