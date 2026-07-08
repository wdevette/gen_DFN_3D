# modules/powerlaw_fits.py
"""
Ajustador de leis de potência para distribuições de fraturas
VERSÃO MELHORADA com detecção automática de x_min e x_max

Melhorias implementadas:
- find_optimal_xmin() agora retorna informações detalhadas
- find_optimal_xmax() NOVO - detecta limite superior
- bootstrap_confidence_intervals() NOVO - IC mais robustos
- bootstrap_pvalue() NOVO - teste de goodness-of-fit
- diagnose_fit_quality() NOVO - diagnósticos automáticos
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple, Optional, List
import warnings

class PowerLawFitter:
    """Ajustador de leis de potência para distribuições de fraturas"""
    
    # Limites geológicos aceitáveis para o expoente
    ALPHA_MIN_GEOLOGICAL = 1.0
    ALPHA_MAX_GEOLOGICAL = 3.0
    
    def __init__(self):
        self._xmin_search_results = None
        self._bootstrap_results = None
    
    def fit_power_law(self, data: np.ndarray, x_min: float, 
                      method: str = "OLS",
                      x_max: float = None) -> Dict:
        """
        Ajusta lei de potência N = a * x^(-c)
        
        Args:
            data: Valores de tamanho (comprimento ou abertura)
            x_min: Valor mínimo para ajuste
            method: "OLS" ou "MLE"
            x_max: Valor máximo para ajuste (opcional) - NOVO
        
        Returns:
            Dicionário com parâmetros ajustados
        """
        # Filtrar dados
        data = data[data >= x_min]
        
        # NOVO: Aplicar x_max se fornecido
        if x_max is not None:
            data = data[data <= x_max]
        
        if len(data) < 10:
            warnings.warn("Poucos dados para ajuste confiável")
        
        if method == "OLS":
            result = self._fit_ols(data, x_min)
        elif method == "MLE":
            result = self._fit_mle(data, x_min)
        else:
            raise ValueError(f"Método desconhecido: {method}")
        
        # NOVO: Adicionar x_max ao resultado
        result['x_max'] = x_max if x_max else float(np.max(data))
        
        return result
    
    def _fit_ols(self, data: np.ndarray, x_min: float) -> Dict:
        """Ajuste por mínimos quadrados em log-log"""
        # Calcular distribuição cumulativa
        sorted_data = np.sort(data)[::-1]
        n = len(sorted_data)
        cumulative = np.arange(1, n + 1)
        
        # Transformação log
        log_x = np.log10(sorted_data)
        log_y = np.log10(cumulative)
        
        # Regressão linear
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)
        
        # Parâmetros da lei de potência
        exponent = -slope
        coefficient = 10**intercept
        
        # Intervalos de confiança (95%)
        t_stat = stats.t.ppf(0.975, n - 2)
        ci_slope = t_stat * std_err
        
        return {
            'exponent': exponent,
            'coefficient': coefficient,
            'x_min': x_min,
            'r_squared': r_value**2,
            'p_value': p_value,
            'ci_exponent': [exponent - ci_slope, exponent + ci_slope],
            'method': 'OLS',
            'n_data': n
        }
    
    def _fit_mle(self, data: np.ndarray, x_min: float) -> Dict:
        """Ajuste por máxima verossimilhança (Clauset et al. 2009)"""
        # Estimar expoente
        data_filtered = data[data >= x_min]
        n = len(data_filtered)
        
        # Estimador de MLE para expoente
        alpha = 1 + n / np.sum(np.log(data_filtered / x_min))
        
        # Erro padrão
        sigma = (alpha - 1) / np.sqrt(n)
        
        # Teste KS para qualidade do ajuste
        ks_stat = self._calculate_ks_statistic(data_filtered, x_min, alpha)
        
        # Coeficiente
        coefficient = (alpha - 1) * x_min**(alpha - 1) * n
        
        # Pseudo-R² baseado na estatística KS
        pseudo_r_squared = max(0, 1 - ks_stat)

        return {
            'exponent': alpha,
            'coefficient': coefficient,
            'x_min': x_min,
            'ks_statistic': ks_stat,
            'r_squared': pseudo_r_squared,
            'sigma': sigma,
            'ci_exponent': [alpha - 1.96*sigma, alpha + 1.96*sigma],
            'method': 'MLE',
            'n_data': n
        }
    
    def _calculate_ks_statistic(self, data: np.ndarray, x_min: float, 
                                alpha: float) -> float:
        """Calcula estatística de Kolmogorov-Smirnov"""
        data_sorted = np.sort(data)
        n = len(data_sorted)
        
        # CDF empírica
        cdf_empirical = np.arange(1, n + 1) / n
        
        # CDF teórica
        cdf_theoretical = 1 - (x_min / data_sorted)**(alpha - 1)
        
        # Estatística KS
        ks_stat = np.max(np.abs(cdf_empirical - cdf_theoretical))
        
        return ks_stat
    
    def fit_aperture_length_relation(self, apertures: np.ndarray, 
                                    lengths: np.ndarray) -> Dict:
        """
        Ajusta relação b = g * l^m
        
        Args:
            apertures: Valores de abertura
            lengths: Valores de comprimento
        
        Returns:
            Parâmetros da relação
        """
        # Remover zeros e valores inválidos
        mask = (apertures > 0) & (lengths > 0)
        b = apertures[mask]
        l = lengths[mask]
        
        # Transformação log
        log_b = np.log10(b)
        log_l = np.log10(l)
        
        # Regressão
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_l, log_b)
        
        # Parâmetros
        m = slope
        g = 10**intercept
        
        # Regressão robusta (RANSAC) como alternativa
        try:
            from sklearn.linear_model import RANSACRegressor
            ransac = RANSACRegressor()
            X = log_l.reshape(-1, 1)
            ransac.fit(X, log_b)
            m_robust = ransac.estimator_.coef_[0]
            g_robust = 10**ransac.estimator_.intercept_
            
            robust_params = {
                'm_robust': m_robust,
                'g_robust': g_robust
            }
        except:
            robust_params = {}
        
        return {
            'm': m,
            'g': g,
            'r_squared': r_value**2,
            'p_value': p_value,
            'std_err': std_err,
            **robust_params
        }
    
    # =========================================================================
    # MÉTODO MELHORADO: find_optimal_xmin
    # =========================================================================
    def find_optimal_xmin(self, data: np.ndarray,
                          min_tail_fraction: float = 0.1,
                          return_details: bool = False) -> float:
        """
        Encontra x_min ótimo por minimização de KS (Clauset et al. 2009)
        
        MELHORADO: Agora retorna informações detalhadas e armazena resultados
        
        Args:
            data: Array de dados
            min_tail_fraction: Fração mínima de dados na cauda (default 10%)
            return_details: Se True, retorna dicionário com detalhes
        
        Returns:
            x_min ótimo (ou dicionário se return_details=True)
        """
        data = data[data > 0]
        unique_values = np.unique(data)
        n_total = len(data)
        
        if len(unique_values) < 5:
            result = unique_values[0]
            if return_details:
                return {'x_min': result, 'alpha': 2.0, 'ks': 1.0, 'n_tail': n_total}
            return result
        
        # Número mínimo de pontos na cauda
        min_tail_size = max(10, int(n_total * min_tail_fraction))
        
        # Testar diferentes x_min (até 50% dos valores únicos)
        max_idx = len(unique_values) - min_tail_size
        x_min_candidates = unique_values[:max(1, int(len(unique_values) * 0.5))]
        
        results_list = []
        best_xmin = x_min_candidates[0]
        best_alpha = 2.0
        best_ks = np.inf
        best_n_tail = n_total
        
        for xm in x_min_candidates:
            data_tail = data[data >= xm]
            n_tail = len(data_tail)
            
            if n_tail < min_tail_size:
                continue
            
            # Estimar alpha
            alpha = 1 + n_tail / np.sum(np.log(data_tail / xm))
            
            # Calcular KS
            ks = self._calculate_ks_statistic(data_tail, xm, alpha)
            
            results_list.append({
                'x_min': xm,
                'alpha': alpha,
                'ks': ks,
                'n_tail': n_tail
            })
            
            if ks < best_ks:
                best_ks = ks
                best_xmin = xm
                best_alpha = alpha
                best_n_tail = n_tail
        
        # Armazenar resultados para análise posterior
        self._xmin_search_results = results_list
        
        if return_details:
            return {
                'x_min': best_xmin,
                'alpha': best_alpha,
                'ks': best_ks,
                'n_tail': best_n_tail,
                'n_total': n_total,
                'fraction_used': best_n_tail / n_total,
                'all_candidates': results_list
            }
        
        return best_xmin
    
    # =========================================================================
    # NOVO: find_optimal_xmax
    # =========================================================================
    def find_optimal_xmax(self, data: np.ndarray, x_min: float,
                          method: str = 'percentile') -> float:
        """
        Encontra x_max ótimo detectando desvios da power-law
        
        NOVO MÉTODO
        
        Args:
            data: Array de dados
            x_min: Limite inferior já determinado
            method: 'percentile' (95%) ou 'residuals' (análise de resíduos)
        
        Returns:
            x_max ótimo
        """
        tail = data[data >= x_min]
        
        if method == 'percentile':
            return float(np.percentile(tail, 95))
        
        elif method == 'residuals':
            tail_sorted = np.sort(tail)[::-1]
            n = len(tail_sorted)
            
            if n < 20:
                return float(np.percentile(tail, 95))
            
            # Ajustar power-law
            alpha = 1 + n / np.sum(np.log(tail / x_min))
            
            # Calcular resíduos
            cumulative = np.arange(1, n + 1)
            log_x = np.log10(tail_sorted)
            log_y_obs = np.log10(cumulative)
            log_y_pred = np.log10(n) - alpha * (log_x - np.log10(x_min))
            
            residuals = log_y_obs - log_y_pred
            
            # Detectar desvio sistemático (média móvel de resíduos < -0.1)
            window_size = max(5, n // 20)
            
            for i in range(n - window_size):
                window_residuals = residuals[i:i+window_size]
                if np.mean(window_residuals) < -0.15:
                    return float(tail_sorted[i])
            
            return float(np.percentile(tail, 99))
        
        else:
            return float(np.percentile(tail, 95))
    
    # =========================================================================
    # NOVO: bootstrap_confidence_intervals
    # =========================================================================
    def bootstrap_confidence_intervals(self, data: np.ndarray, x_min: float,
                                       n_bootstrap: int = 1000,
                                       confidence: float = 0.95) -> Dict:
        """
        Calcula intervalos de confiança via bootstrap
        
        NOVO MÉTODO - IC mais robustos que fórmula assintótica
        
        Args:
            data: Array de dados
            x_min: Limite inferior
            n_bootstrap: Número de amostras bootstrap
            confidence: Nível de confiança (default 95%)
        
        Returns:
            Dicionário com IC
        """
        tail = data[data >= x_min]
        n = len(tail)
        
        alphas_bootstrap = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(tail, size=n, replace=True)
            alpha_boot = 1 + n / np.sum(np.log(sample / x_min))
            alphas_bootstrap.append(alpha_boot)
        
        alphas_bootstrap = np.array(alphas_bootstrap)
        
        alpha_low = (1 - confidence) / 2 * 100
        alpha_high = (1 + confidence) / 2 * 100
        
        self._bootstrap_results = alphas_bootstrap
        
        return {
            'alpha_mean': float(np.mean(alphas_bootstrap)),
            'alpha_std': float(np.std(alphas_bootstrap)),
            'ci_lower': float(np.percentile(alphas_bootstrap, alpha_low)),
            'ci_upper': float(np.percentile(alphas_bootstrap, alpha_high)),
            'confidence': confidence,
            'n_bootstrap': n_bootstrap
        }
    
    # =========================================================================
    # NOVO: bootstrap_pvalue
    # =========================================================================
    def bootstrap_pvalue(self, data: np.ndarray, x_min: float, alpha: float,
                         n_bootstrap: int = 500) -> float:
        """
        Calcula p-valor do ajuste via bootstrap (Clauset et al. 2009)
        
        NOVO MÉTODO - Teste de goodness-of-fit
        
        Args:
            data: Array de dados
            x_min: Limite inferior
            alpha: Expoente ajustado
            n_bootstrap: Número de amostras bootstrap
        
        Returns:
            p-valor
        """
        tail = data[data >= x_min]
        n = len(tail)
        
        # KS observado
        ks_observed = self._calculate_ks_statistic(tail, x_min, alpha)
        
        # Gerar amostras sintéticas
        ks_synthetic = []
        
        for _ in range(n_bootstrap):
            # Gerar dados sintéticos da power-law
            u = np.random.uniform(0, 1, n)
            synthetic_tail = x_min * (1 - u) ** (-1 / (alpha - 1))
            
            # Reajustar alpha
            alpha_synth = 1 + n / np.sum(np.log(synthetic_tail / x_min))
            
            # Calcular KS
            ks_synth = self._calculate_ks_statistic(synthetic_tail, x_min, alpha_synth)
            ks_synthetic.append(ks_synth)
        
        # p-valor
        p_value = np.mean(np.array(ks_synthetic) >= ks_observed)
        
        return float(p_value)
    
    # =========================================================================
    # NOVO: diagnose_fit_quality
    # =========================================================================
    def diagnose_fit_quality(self, data: np.ndarray, x_min: float, 
                             alpha: float) -> Dict:
        """
        Diagnostica qualidade do ajuste
        
        NOVO MÉTODO - Detecta rollover, cutoff, etc.
        
        Args:
            data: Array de dados
            x_min: Limite inferior
            alpha: Expoente ajustado
        
        Returns:
            Dicionário com diagnósticos
        """
        tail = data[data >= x_min]
        tail_sorted = np.sort(tail)[::-1]
        n = len(tail_sorted)
        
        # Calcular resíduos
        cumulative = np.arange(1, n + 1)
        log_x = np.log10(tail_sorted)
        log_y_obs = np.log10(cumulative)
        log_y_pred = np.log10(n) - alpha * (log_x - np.log10(x_min))
        residuals = log_y_obs - log_y_pred
        
        # Dividir em regiões
        n_third = max(1, n // 3)
        res_start = residuals[:n_third]
        res_middle = residuals[n_third:2*n_third] if 2*n_third <= n else residuals[n_third:]
        res_end = residuals[2*n_third:] if 2*n_third < n else np.array([0])
        
        diagnostics = {
            'residuals_mean': float(np.mean(residuals)),
            'residuals_std': float(np.std(residuals)),
            'start_mean': float(np.mean(res_start)),
            'end_mean': float(np.mean(res_end)),
            'issues': [],
            'is_good_fit': True,
            'alpha_in_range': self.ALPHA_MIN_GEOLOGICAL <= alpha <= self.ALPHA_MAX_GEOLOGICAL
        }
        
        # Detectar problemas
        if np.mean(res_start) > 0.1:
            diagnostics['issues'].append({
                'type': 'ROLLOVER',
                'description': 'Excesso de fraturas pequenas detectado',
                'suggestion': 'Aumentar x_min ou verificar artefatos'
            })
            diagnostics['is_good_fit'] = False
        
        if len(res_end) > 0 and np.mean(res_end) < -0.1:
            diagnostics['issues'].append({
                'type': 'EXPONENTIAL_CUTOFF', 
                'description': 'Falta de fraturas grandes detectada',
                'suggestion': 'Diminuir x_max ou usar distribuição truncada'
            })
            diagnostics['is_good_fit'] = False
        
        if np.std(residuals) > 0.2:
            diagnostics['issues'].append({
                'type': 'HIGH_VARIANCE',
                'description': 'Alta variância nos resíduos',
                'suggestion': 'Verificar se dados seguem power-law'
            })
            diagnostics['is_good_fit'] = False
        
        if not diagnostics['alpha_in_range']:
            diagnostics['issues'].append({
                'type': 'ALPHA_OUT_OF_RANGE',
                'description': f'α={alpha:.2f} fora do range geológico (1.0-3.0)',
                'suggestion': 'Verificar qualidade dos dados ou ajustar x_min'
            })
        
        return diagnostics
    
    # =========================================================================
    # NOVO: fit_power_law_auto - MÉTODO PRINCIPAL COM TUDO AUTOMÁTICO
    # =========================================================================
    def fit_power_law_auto(self, data: np.ndarray,
                           method: str = "MLE",
                           auto_xmin: bool = True,
                           auto_xmax: bool = True,
                           compute_pvalue: bool = False,
                           n_bootstrap: int = 200) -> Dict:
        """
        Ajuste automático completo de power-law
        
        NOVO MÉTODO - Combina todas as melhorias em um único método
        
        Args:
            data: Array de dados
            method: "OLS" ou "MLE"
            auto_xmin: Detectar x_min automaticamente
            auto_xmax: Detectar x_max automaticamente
            compute_pvalue: Calcular p-valor via bootstrap (mais lento)
            n_bootstrap: Número de amostras bootstrap
        
        Returns:
            Dicionário completo com resultados e diagnósticos
        """
        data = data[data > 0]
        n_total = len(data)
        
        # 1. Encontrar x_min
        if auto_xmin:
            xmin_result = self.find_optimal_xmin(data, return_details=True)
            x_min = xmin_result['x_min']
            xmin_info = xmin_result
        else:
            x_min = float(np.percentile(data, 5))
            xmin_info = {'x_min': x_min, 'method': 'percentile_5'}
        
        # 2. Encontrar x_max
        if auto_xmax:
            x_max = self.find_optimal_xmax(data, x_min, method='residuals')
        else:
            x_max = float(np.percentile(data, 99))
        
        # 3. Ajustar power-law
        fit_result = self.fit_power_law(data, x_min, method=method, x_max=x_max)
        
        # 4. Diagnósticos
        diagnostics = self.diagnose_fit_quality(data, x_min, fit_result['exponent'])
        
        # 5. Bootstrap p-valor (opcional - mais lento)
        if compute_pvalue and method == "MLE":
            p_value = self.bootstrap_pvalue(data, x_min, fit_result['exponent'], n_bootstrap)
            fit_result['bootstrap_pvalue'] = p_value
            fit_result['is_significant'] = p_value > 0.1
        
        # 6. Montar resultado completo
        result = {
            **fit_result,
            'x_min_auto': auto_xmin,
            'x_max_auto': auto_xmax,
            'xmin_details': xmin_info,
            'diagnostics': diagnostics,
            'n_total': n_total,
            'fraction_used': fit_result['n_data'] / n_total
        }
        
        return result



