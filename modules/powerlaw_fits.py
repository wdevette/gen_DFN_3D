import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple, Optional
import warnings

class PowerLawFitter:
    """Ajustador de leis de potência para distribuições de fraturas"""
    
    def fit_power_law(self, data: np.ndarray, x_min: float, 
                      method: str = "OLS") -> Dict:
        """
        Ajusta lei de potência N = a * x^(-c)
        
        Args:
            data: Valores de tamanho (comprimento ou abertura)
            x_min: Valor mínimo para ajuste
            method: "OLS" ou "MLE"
        
        Returns:
            Dicionário com parâmetros ajustados
        """
        # Filtrar dados
        data = data[data >= x_min]
        
        if len(data) < 10:
            warnings.warn("Poucos dados para ajuste confiável")
        
        if method == "OLS":
            return self._fit_ols(data, x_min)
        elif method == "MLE":
            return self._fit_mle(data, x_min)
        else:
            raise ValueError(f"Método desconhecido: {method}")
    
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
        
        # Adicionar 'r_squared' para compatibilidade
        # Para MLE, usamos pseudo-R² baseado na estatística KS
        # Quanto menor o KS, melhor o ajuste (inverso para R²)
        pseudo_r_squared = max(0, 1 - ks_stat)  # Simplificação

        return {
            'exponent': alpha,
            'coefficient': coefficient,
            'x_min': x_min,
            'ks_statistic': ks_stat,
            'r_squared': pseudo_r_squared, # ADICIONADO para compatibilidade
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
    
    def find_optimal_xmin(self, data: np.ndarray) -> float:
        """
        Encontra x_min ótimo por minimização de KS
        (Clauset et al. 2009)
        """
        unique_values = np.unique(data)
        if len(unique_values) < 5:
            return unique_values[0]
        
        # Testar diferentes x_min
        x_min_candidates = unique_values[:int(len(unique_values) * 0.5)]
        ks_values = []
        
        for xm in x_min_candidates:
            data_tail = data[data >= xm]
            if len(data_tail) < 10:
                continue
            
            # Estimar alpha
            n = len(data_tail)
            alpha = 1 + n / np.sum(np.log(data_tail / xm))
            
            # Calcular KS
            ks = self._calculate_ks_statistic(data_tail, xm, alpha)
            ks_values.append((xm, ks))
        
        if ks_values:
            # Retornar x_min com menor KS
            best_xmin = min(ks_values, key=lambda x: x[1])[0]
            return best_xmin
        else:
            return unique_values[0]