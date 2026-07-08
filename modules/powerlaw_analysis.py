# modules/powerlaw_analysis.py
"""
Análise automática de power-law com validação robusta
VERSÃO CORRIGIDA - Janeiro 2026

Melhorias:
- Calcula OLS e MLE simultaneamente
- Permite escolha do método (OLS, MLE, média ponderada)
- Sugere automaticamente a melhor opção
- Parâmetros globais para passar entre etapas
- Remoção da análise de abertura (foco em comprimento)

Referências:
- Clauset, A., Shalizi, C. R., & Newman, M. E. (2009). SIAM review, 51(4), 661-703.
- Bonnet, E., et al. (2001). Reviews of geophysics, 39(3), 347-383.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
import warnings
from scipy import stats


@dataclass
class PowerLawParams:
    """
    Parâmetros globais de power-law para passar entre etapas
    """
    # Parâmetros principais
    alpha: float                    # Expoente usado (pode ser OLS, MLE ou ponderado)
    l_min: float                    # Comprimento mínimo (m)
    l_max: float                    # Comprimento máximo (m)
    
    # Método selecionado
    method_used: str = 'weighted'   # 'OLS', 'MLE', ou 'weighted'
    
    # Resultados individuais
    alpha_ols: float = 0.0
    alpha_mle: float = 0.0
    r_squared_ols: float = 0.0
    r_squared_mle: float = 0.0
    
    # Qualidade
    is_valid: bool = True
    validation_status: str = 'OK'
    warnings: List[str] = field(default_factory=list)
    
    # Estatísticas
    n_data: int = 0
    n_total: int = 0
    
    def summary(self) -> str:
        """Retorna resumo formatado"""
        return (
            f"=== Parâmetros Power-Law ===\n"
            f"α (usado): {self.alpha:.3f} ({self.method_used})\n"
            f"α OLS: {self.alpha_ols:.3f} (R²={self.r_squared_ols:.4f})\n"
            f"α MLE: {self.alpha_mle:.3f} (R²={self.r_squared_mle:.4f})\n"
            f"l_min: {self.l_min:.4f} m\n"
            f"l_max: {self.l_max:.4f} m\n"
            f"Dados: {self.n_data}/{self.n_total}\n"
            f"Status: {self.validation_status}"
        )


class PowerLawAnalyzer:
    """
    Analisador de distribuições power-law com múltiplos métodos
    """
    
    # Limites aceitáveis para α (geológico)
    ALPHA_MIN = 1.2
    ALPHA_MAX = 2.8
    ALPHA_FALLBACK = 1.8  # Valor de fallback se α inválido
    
    def __init__(self):
        self.results = None
        self.params: Optional[PowerLawParams] = None
        
    def analyze_both_methods(self,
                            lengths: np.ndarray,
                            percentile_min: float = 5.0,
                            percentile_max: float = 95.0,
                            validate: bool = True) -> Dict:
        """
        Análise completa com OLS e MLE simultaneamente
        
        Args:
            lengths: Array de comprimentos (em METROS)
            percentile_min: Percentil para l_min (default 5%)
            percentile_max: Percentil para l_max (default 95%)
            validate: Se True, valida α e aplica fallback se necessário
        
        Returns:
            Dicionário com resultados de ambos os métodos
        """
        # Remover valores inválidos
        lengths_clean = lengths[lengths > 0]
        n_total = len(lengths)
        n_valid = len(lengths_clean)
        
        if n_valid < 10:
            raise ValueError(f"Poucos dados válidos: {n_valid} < 10")
        
        # Calcular l_min e l_max por percentis (detecção automática)
        l_min = np.percentile(lengths_clean, percentile_min)
        l_max = np.percentile(lengths_clean, percentile_max)
        
        # Aplicar limites absolutos de segurança
        l_min = max(l_min, 0.01)   # Mínimo 1 cm
        l_max = min(l_max, 100.0)  # Máximo 100 m
        
        # Filtrar dados para o range [l_min, l_max]
        lengths_range = lengths_clean[(lengths_clean >= l_min) & (lengths_clean <= l_max)]
        n_range = len(lengths_range)
        
        if n_range < 5:
            warnings.warn(f"Poucos dados no range [{l_min:.3f}, {l_max:.3f}]: {n_range}")
        
        # Ajustar OLS
        ols_result = self._fit_powerlaw_ols(lengths_range)
        
        # Ajustar MLE
        mle_result = self._fit_powerlaw_mle(lengths_range, l_min)
        
        # Validar ambos
        ols_validation = self._validate_alpha(ols_result['exponent'], validate)
        mle_validation = self._validate_alpha(mle_result['exponent'], validate)
        
        # Calcular média ponderada (baseada em R²)
        r2_ols = ols_result.get('r_squared', 0.5)
        r2_mle = mle_result.get('r_squared', 0.5)
        
        # Pesos baseados em R²
        total_r2 = r2_ols + r2_mle
        if total_r2 > 0:
            w_ols = r2_ols / total_r2
            w_mle = r2_mle / total_r2
        else:
            w_ols = w_mle = 0.5
        
        alpha_weighted = w_ols * ols_result['exponent'] + w_mle * mle_result['exponent']
        
        # Determinar melhor método automaticamente
        recommended_method, recommendation_reason = self._recommend_method(
            ols_result, mle_result, ols_validation, mle_validation
        )
        
        # Montar resultado completo
        result = {
            # Parâmetros de range (GLOBAIS - passam para outras etapas)
            'l_min': l_min,
            'l_max': l_max,
            'l_min_percentile': percentile_min,
            'l_max_percentile': percentile_max,
            
            # Resultados OLS
            'ols': {
                'alpha': ols_result['exponent'],
                'coefficient': ols_result.get('coefficient', 0),
                'r_squared': r2_ols,
                'p_value': ols_result.get('p_value', 1.0),
                'validation': ols_validation
            },
            
            # Resultados MLE
            'mle': {
                'alpha': mle_result['exponent'],
                'sigma': mle_result.get('sigma', 0),
                'r_squared': r2_mle,
                'ks_statistic': mle_result.get('ks_statistic', 1.0),
                'validation': mle_validation
            },
            
            # Média ponderada
            'weighted': {
                'alpha': alpha_weighted,
                'weight_ols': w_ols,
                'weight_mle': w_mle
            },
            
            # Recomendação
            'recommendation': {
                'method': recommended_method,
                'reason': recommendation_reason,
                'alpha_recommended': self._get_alpha_by_method(
                    recommended_method, ols_result, mle_result, alpha_weighted
                )
            },
            
            # Estatísticas dos dados
            'data_stats': {
                'n_total': n_total,
                'n_valid': n_valid,
                'n_range': n_range,
                'length_min': float(np.min(lengths_clean)),
                'length_max': float(np.max(lengths_clean)),
                'length_mean': float(np.mean(lengths_clean)),
                'length_median': float(np.median(lengths_clean))
            }
        }
        
        self.results = result
        return result
    
    def get_params(self, method: str = 'auto') -> PowerLawParams:
        """
        Retorna parâmetros power-law para usar em outras etapas
        
        Args:
            method: 'OLS', 'MLE', 'weighted', ou 'auto' (usa recomendação)
        
        Returns:
            PowerLawParams com valores selecionados
        """
        if self.results is None:
            raise ValueError("Execute analyze_both_methods() primeiro")
        
        r = self.results
        
        # Determinar método
        if method.lower() == 'auto':
            method_used = r['recommendation']['method']
        else:
            method_used = method.upper() if method.upper() in ['OLS', 'MLE'] else 'weighted'
        
        # Obter alpha pelo método
        if method_used == 'OLS':
            alpha = r['ols']['alpha']
            validation = r['ols']['validation']
        elif method_used == 'MLE':
            alpha = r['mle']['alpha']
            validation = r['mle']['validation']
        else:  # weighted
            alpha = r['weighted']['alpha']
            # Validar média ponderada
            validation = self._validate_alpha(alpha, True)
        
        # Criar objeto de parâmetros
        params = PowerLawParams(
            alpha=alpha,
            l_min=r['l_min'],
            l_max=r['l_max'],
            method_used=method_used,
            alpha_ols=r['ols']['alpha'],
            alpha_mle=r['mle']['alpha'],
            r_squared_ols=r['ols']['r_squared'],
            r_squared_mle=r['mle']['r_squared'],
            is_valid=validation['is_valid'],
            validation_status=validation['status'],
            warnings=validation.get('warnings', []),
            n_data=r['data_stats']['n_range'],
            n_total=r['data_stats']['n_total']
        )
        
        self.params = params
        return params
    
    def _fit_powerlaw_ols(self, data: np.ndarray) -> Dict:
        """Ajuste OLS em espaço log-log"""
        # Distribuição cumulativa complementar
        sorted_data = np.sort(data)[::-1]
        n = len(sorted_data)
        cumulative = np.arange(1, n + 1)
        
        # Log-log
        log_x = np.log10(sorted_data)
        log_y = np.log10(cumulative)
        
        # Regressão linear
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        
        # Parâmetros
        exponent = -slope  # α = -slope
        coefficient = 10**intercept
        
        return {
            'exponent': exponent,
            'coefficient': coefficient,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            'n_data': n
        }
    
    def _fit_powerlaw_mle(self, data: np.ndarray, x_min: float) -> Dict:
        """Ajuste MLE (Clauset et al. 2009)"""
        data_filtered = data[data >= x_min]
        n = len(data_filtered)
        
        if n == 0:
            return {'exponent': 2.0, 'sigma': 1.0, 'r_squared': 0.0}
        
        # Estimador MLE para α
        sum_log = np.sum(np.log(data_filtered / x_min))
        if sum_log <= 0:
            alpha = 2.0
        else:
            alpha = 1 + n / sum_log
        
        # Erro padrão
        sigma = (alpha - 1) / np.sqrt(n) if n > 0 else 1.0
        
        # Pseudo R² (baseado em KS)
        ks_stat = self._ks_statistic(data_filtered, x_min, alpha)
        pseudo_r2 = max(0, 1 - ks_stat)
        
        return {
            'exponent': alpha,
            'coefficient': (alpha - 1) * x_min**(alpha - 1) * n if alpha > 1 else n,
            'r_squared': pseudo_r2,
            'ks_statistic': ks_stat,
            'sigma': sigma,
            'n_data': n
        }
    
    def _ks_statistic(self, data: np.ndarray, x_min: float, alpha: float) -> float:
        """Estatística Kolmogorov-Smirnov"""
        data_sorted = np.sort(data)
        n = len(data_sorted)
        
        if n == 0:
            return 1.0
        
        # CDF empírica
        cdf_empirical = np.arange(1, n + 1) / n
        
        # CDF teórica (power-law)
        if alpha > 1:
            cdf_theoretical = 1 - (x_min / data_sorted)**(alpha - 1)
        else:
            cdf_theoretical = np.linspace(0, 1, n)
        
        # Estatística KS
        ks = np.max(np.abs(cdf_empirical - cdf_theoretical))
        
        return ks
    
    def _validate_alpha(self, alpha: float, apply_fallback: bool = True) -> Dict:
        """Valida α e aplica fallback se necessário"""
        validation = {
            'alpha_raw': alpha,
            'is_valid': False,
            'status': '',
            'alpha_used': alpha,
            'fallback_applied': False,
            'warnings': []
        }
        
        # Verificar se está no range aceitável
        if self.ALPHA_MIN <= alpha <= self.ALPHA_MAX:
            validation['is_valid'] = True
            validation['status'] = 'OK'
            validation['alpha_used'] = alpha
        else:
            validation['is_valid'] = False
            
            if alpha < self.ALPHA_MIN:
                validation['status'] = f'TOO_LOW (α < {self.ALPHA_MIN})'
                validation['warnings'].append(
                    f"α = {alpha:.3f} muito baixo! "
                    f"Possível problema: excesso de fraturas pequenas"
                )
            else:
                validation['status'] = f'TOO_HIGH (α > {self.ALPHA_MAX})'
                validation['warnings'].append(
                    f"α = {alpha:.3f} muito alto! "
                    f"Possível problema: falta de fraturas pequenas"
                )
            
            # Aplicar fallback
            if apply_fallback:
                validation['alpha_used'] = self.ALPHA_FALLBACK
                validation['fallback_applied'] = True
                validation['warnings'].append(
                    f"Usando α = {self.ALPHA_FALLBACK} (fallback)"
                )
            else:
                validation['alpha_used'] = alpha
        
        return validation
    
    def _recommend_method(self, ols_result: Dict, mle_result: Dict,
                         ols_validation: Dict, mle_validation: Dict) -> Tuple[str, str]:
        """
        Recomenda o melhor método baseado nos resultados
        
        Returns:
            Tuple (método_recomendado, razão)
        """
        alpha_ols = ols_result['exponent']
        alpha_mle = mle_result['exponent']
        r2_ols = ols_result.get('r_squared', 0)
        r2_mle = mle_result.get('r_squared', 0)
        
        ols_valid = ols_validation['is_valid']
        mle_valid = mle_validation['is_valid']
        
        diff = abs(alpha_ols - alpha_mle)
        
        # Regra 1: Se diferença é pequena, usar OLS (mais interpretável)
        if diff < 0.2:
            return 'OLS', f'Métodos concordam (Δα = {diff:.3f})'
        
        # Regra 2: Se apenas um é válido, usar esse
        if ols_valid and not mle_valid:
            return 'OLS', 'MLE fora do range válido'
        if mle_valid and not ols_valid:
            return 'MLE', 'OLS fora do range válido'
        
        # Regra 3: Se ambos válidos mas diferem, usar média ponderada
        if ols_valid and mle_valid and diff >= 0.2:
            return 'weighted', f'Diferença significativa (Δα = {diff:.3f}), usando média ponderada'
        
        # Regra 4: Se nenhum válido, usar R² para decidir
        if r2_ols > r2_mle:
            return 'OLS', f'Melhor ajuste (R² OLS={r2_ols:.3f} > MLE={r2_mle:.3f})'
        else:
            return 'MLE', f'Melhor ajuste (R² MLE={r2_mle:.3f} > OLS={r2_ols:.3f})'
    
    def _get_alpha_by_method(self, method: str, ols_result: Dict, 
                            mle_result: Dict, alpha_weighted: float) -> float:
        """Retorna alpha pelo método especificado"""
        if method == 'OLS':
            return ols_result['exponent']
        elif method == 'MLE':
            return mle_result['exponent']
        else:
            return alpha_weighted
    
    def get_plot_data(self) -> Dict:
        """
        Retorna dados para visualização
        """
        if self.results is None:
            return {}
        
        r = self.results
        return {
            'l_min': r['l_min'],
            'l_max': r['l_max'],
            'alpha_ols': r['ols']['alpha'],
            'alpha_mle': r['mle']['alpha'],
            'alpha_weighted': r['weighted']['alpha'],
            'r2_ols': r['ols']['r_squared'],
            'r2_mle': r['mle']['r_squared'],
            'recommendation': r['recommendation']['method']
        }


def calculate_powerlaw_dual(data: pd.DataFrame,
                           length_col: str = 'length',
                           percentile_min: float = 5.0,
                           percentile_max: float = 95.0) -> Tuple[Dict, PowerLawParams]:
    """
    Função auxiliar para cálculo completo de parâmetros power-law
    
    Args:
        data: DataFrame com dados
        length_col: Nome da coluna de comprimento (em metros)
        percentile_min: Percentil para l_min
        percentile_max: Percentil para l_max
    
    Returns:
        Tupla (resultados_completos, parâmetros_globais)
    """
    analyzer = PowerLawAnalyzer()
    
    lengths = data[length_col].values
    
    results = analyzer.analyze_both_methods(
        lengths=lengths,
        percentile_min=percentile_min,
        percentile_max=percentile_max,
        validate=True
    )
    
    # Usar método recomendado por padrão
    params = analyzer.get_params(method='auto')
    
    return results, params
