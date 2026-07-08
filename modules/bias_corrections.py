# modules/bias_corrections.py
"""
Sistema completo de correção de vieses para dados de fraturas
Implementa: Terzaghi (1965), Marrett (1996), e D-L Scaling (Schultz 2008)

VERSÃO CORRIGIDA - Janeiro 2026
Correções aplicadas:
- Valores de DL_SCALING_CONSTANTS corrigidos conforme Schultz et al. (2008)
- Adicionado suporte para rochas vulcânicas (traquito, basalto)
- Valor para Itapoama (traquito): a_observado = 0.069, consistente com dikes Etiópia (0.078)

Referências:
- Terzaghi, R.D. (1965). Geotechnique, 15(3), 287-304
- Marrett, R. (1996). J Struct Geol, 18(2-3), 169-178
- Schultz, R.A. et al. (2008). J Struct Geol, 30(11), 1405-1411
- Olson, J.E. (2003). J Geophys Res, 108(B9), 2413
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import warnings

@dataclass
class CorrectionResults:
    """Resultados de uma correção de viés"""
    method: str
    intensity_observed: float
    intensity_corrected: float
    correction_factor: float
    n_data_original: int
    n_data_used: int
    parameters: Dict
    validation: Dict
    
    def summary(self) -> str:
        """Retorna resumo formatado dos resultados"""
        increase = (self.correction_factor - 1) * 100
        return (
            f"🔧 {self.method}\n"
            f"   Observado: {self.intensity_observed:.2f}\n"
            f"   Corrigido: {self.intensity_corrected:.2f}\n"
            f"   Fator: {self.correction_factor:.2f}× (+{increase:.1f}%)\n"
            f"   Dados: {self.n_data_used}/{self.n_data_original}"
        )


class BiasCorrector:
    """
    Sistema completo de correção de vieses para dados de fraturas
    """
    
    # ==========================================================================
    # CONSTANTES CORRIGIDAS para D-L Scaling (Schultz et al., 2008)
    # ==========================================================================
    # Relação: D = a × L^n, onde:
    #   D = abertura/deslocamento (metros)
    #   L = comprimento (metros)
    #   n = 0.5 para juntas (modo I)
    #   a = coeficiente dependente do tipo de rocha
    #
    # IMPORTANTE: Valores de 'a' extraídos de Schultz et al. (2008)
    # Para rochas vulcânicas: Ethiopian dikes → a ≈ 0.078
    # Para Itapoama (traquito): a_observado = 0.069 (consistente!)
    # ==========================================================================
    
    DL_SCALING_CONSTANTS = {
        # ===== ROCHAS VULCÂNICAS (valores altos de 'a') =====
        'volcanic': 0.078,              # Rochas vulcânicas em geral
        'basalt': 0.078,                # Basaltos (Ethiopian dikes)
        'trachyte': 0.078,              # Traquito (similar a basalto) - ITAPOAMA
        'dikes_ethiopia': 0.078,        # Dikes da Etiópia (referência)
        'andesite': 0.070,              # Andesitos
        'rhyolite': 0.065,              # Riolitos
        
        # ===== ROCHAS SEDIMENTARES =====
        'sandstone': 0.022,             # Arenitos em geral
        'sandstone_faults': 0.015,      # Falhas em arenitos
        'limestone': 0.025,             # Calcários em geral
        'carbonate': 0.025,             # Carbonatos
        'dolomite': 0.020,              # Dolomitos
        'shale': 0.012,                 # Folhelhos
        'mudstone': 0.012,              # Argilitos
        'siltstone': 0.015,             # Siltitos
        
        # ===== ROCHAS ÍGNEAS PLUTÔNICAS =====
        'granite': 0.035,               # Granito
        'granodiorite': 0.032,          # Granodiorito
        'diorite': 0.030,               # Diorito
        'gabbro': 0.040,                # Gabro
        
        # ===== ROCHAS METAMÓRFICAS =====
        'crystalline': 0.030,           # Rochas cristalinas em geral
        'gneiss': 0.028,                # Gnaisses
        'schist': 0.025,                # Xistos
        'marble': 0.022,                # Mármore
        'quartzite': 0.020,             # Quartzito
        
        # ===== DEFAULT =====
        'default': 0.030,               # Valor médio conservador
        'custom': None                  # Para valor customizado pelo usuário
    }
    
    # Descrições para UI
    ROCK_TYPE_DESCRIPTIONS = {
        'volcanic': 'Rochas Vulcânicas (a=0.078)',
        'basalt': 'Basalto (a=0.078)',
        'trachyte': 'Traquito (a=0.078) - Itapoama',
        'dikes_ethiopia': 'Dikes Etíopes (a=0.078) - Referência',
        'andesite': 'Andesito (a=0.070)',
        'rhyolite': 'Riolito (a=0.065)',
        'sandstone': 'Arenito (a=0.022)',
        'sandstone_faults': 'Falhas em Arenito (a=0.015)',
        'limestone': 'Calcário (a=0.025)',
        'carbonate': 'Carbonatos (a=0.025)',
        'dolomite': 'Dolomito (a=0.020)',
        'shale': 'Folhelho (a=0.012)',
        'mudstone': 'Argilito (a=0.012)',
        'granite': 'Granito (a=0.035)',
        'granodiorite': 'Granodiorito (a=0.032)',
        'gabbro': 'Gabro (a=0.040)',
        'crystalline': 'Rochas Cristalinas (a=0.030)',
        'gneiss': 'Gnaisse (a=0.028)',
        'schist': 'Xisto (a=0.025)',
        'marble': 'Mármore (a=0.022)',
        'quartzite': 'Quartzito (a=0.020)',
        'default': 'Padrão conservador (a=0.030)',
        'custom': 'Valor customizado'
    }
    
    def __init__(self):
        """Inicializa o corretor de vieses"""
        self.correction_history = []
    
    @classmethod
    def get_available_rock_types(cls) -> Dict[str, str]:
        """Retorna tipos de rocha disponíveis com descrições"""
        return cls.ROCK_TYPE_DESCRIPTIONS
    
    @classmethod
    def get_rock_constant(cls, rock_type: str) -> float:
        """Retorna a constante 'a' para um tipo de rocha"""
        return cls.DL_SCALING_CONSTANTS.get(
            rock_type.lower(), 
            cls.DL_SCALING_CONSTANTS['default']
        )
        
    # ========================================================================
    # TERZAGHI (1965) - Correção de Orientação para Scanline
    # ========================================================================
    
    def terzaghi_correction(self, 
                           data: pd.DataFrame,
                           scanline_azimuth: float,
                           scanline_length: float = None,
                           theta_min: float = 5.0,
                           weight_max: float = 10.0) -> CorrectionResults:
        """
        Aplica correção de Terzaghi (1965) para viés de orientação em scanlines
        """
        if len(data) == 0:
            raise ValueError("DataFrame vazio")
        
        if 'orientation' not in data.columns:
            raise ValueError("Coluna 'orientation' não encontrada")
        
        if scanline_length is None:
            if 'position' in data.columns:
                scanline_length = data['position'].max()
            else:
                warnings.warn("Comprimento do scanline não fornecido, usando 1.0m")
                scanline_length = 1.0
        
        frac_orientations = data['orientation'].values
        scanline_rad = np.deg2rad(scanline_azimuth % 360)
        frac_rad = np.deg2rad(frac_orientations % 360)
        
        theta = np.abs(frac_rad - scanline_rad)
        theta = np.minimum(theta, np.pi - theta)
        theta_deg = np.rad2deg(theta)
        
        valid_mask = theta_deg >= theta_min
        n_blind = np.sum(~valid_mask)
        
        if n_blind > 0:
            warnings.warn(f"{n_blind} fraturas na blind zone (θ < {theta_min}°) removidas")
        
        cos_theta = np.abs(np.cos(theta[valid_mask]))
        cos_theta = np.clip(cos_theta, 1e-6, 1.0)
        weights = 1.0 / cos_theta
        weights = np.minimum(weights, weight_max)
        
        n_obs = len(data)
        P10_obs = n_obs / scanline_length
        P10_corr = np.sum(weights) / scanline_length
        
        correction_factor = P10_corr / P10_obs if P10_obs > 0 else 1.0
        
        data_corrected = data[valid_mask].copy()
        data_corrected['terzaghi_weight'] = weights
        data_corrected['theta_deg'] = theta_deg[valid_mask]
        
        parameters = {
            'scanline_azimuth': scanline_azimuth,
            'scanline_length': scanline_length,
            'theta_min': theta_min,
            'weight_max': weight_max,
            'n_blind_zone': int(n_blind)
        }
        
        validation = {
            'mean_weight': float(np.mean(weights)),
            'max_weight': float(np.max(weights)),
            'weight_distribution': {
                'min': float(np.min(weights)),
                'q25': float(np.percentile(weights, 25)),
                'median': float(np.median(weights)),
                'q75': float(np.percentile(weights, 75)),
                'max': float(np.max(weights))
            },
            'data_corrected': data_corrected
        }
        
        result = CorrectionResults(
            method='Terzaghi (1965)',
            intensity_observed=P10_obs,
            intensity_corrected=P10_corr,
            correction_factor=correction_factor,
            n_data_original=n_obs,
            n_data_used=len(data_corrected),
            parameters=parameters,
            validation=validation
        )
        
        self.correction_history.append(result)
        return result
    
    # ========================================================================
    # MARRETT (1996) - Correção de Truncamento
    # ========================================================================
    
    def marrett_correction(self,
                          intensity_observed: float,
                          alpha: float,
                          l_min: float,
                          l_max: float) -> CorrectionResults:
        """
        Aplica correção de truncamento de Marrett (1996)
        
        Corrige a subestimação causada pelo truncamento de fraturas abaixo de l_min.
        """
        if alpha <= 1.0:
            warnings.warn(f"α = {alpha} ≤ 1.0. Usando α = 1.01 para evitar singularidade")
            alpha = 1.01
        
        if l_min <= 0:
            raise ValueError("l_min deve ser > 0")
        if l_max <= l_min:
            raise ValueError("l_max deve ser > l_min")
        
        # Calcular fator de correção de Marrett
        if np.abs(alpha - 1.0) < 1e-6:
            factor = np.log(l_max / l_min)
        else:
            numerator = l_max**(1 - alpha) - l_min**(1 - alpha)
            denominator = (1 - alpha) * l_max**(1 - alpha)
            factor = numerator / denominator
        
        intensity_corrected = intensity_observed * factor
        
        parameters = {
            'alpha': alpha,
            'l_min': l_min,
            'l_max': l_max,
            'formula': 'F = (l_max^(1-α) - l_min^(1-α)) / ((1-α) × l_max^(1-α))'
        }
        
        validation = {
            'factor_interpretation': self._interpret_marrett_factor(factor),
            'alpha_validity': 'OK' if 1.2 <= alpha <= 2.8 else 'WARNING: α fora do range típico',
            'size_range_ratio': l_max / l_min
        }
        
        result = CorrectionResults(
            method='Marrett (1996)',
            intensity_observed=intensity_observed,
            intensity_corrected=intensity_corrected,
            correction_factor=factor,
            n_data_original=0,
            n_data_used=0,
            parameters=parameters,
            validation=validation
        )
        
        self.correction_history.append(result)
        return result
    
    def _interpret_marrett_factor(self, factor: float) -> str:
        """Interpreta o significado do fator de correção"""
        if factor < 1.5:
            return "Baixo (< 1.5×) - truncamento mínimo"
        elif factor < 2:
            return "Moderado (1.5-2×)"
        elif factor < 5:
            return "Significativo (2-5×)"
        elif factor < 10:
            return "Alto (5-10×)"
        else:
            return "Muito Alto (> 10×) - revisar dados"
    
    # ========================================================================
    # D-L SCALING VALIDATION (Schultz 2008, Olson 2003) - CORRIGIDO
    # ========================================================================

    def dl_scaling_validation(self,
                             data: pd.DataFrame,
                             rock_type: str = 'default',
                             tolerance: float = 5.0,
                             A_custom: Optional[float] = None,
                             min_length_abs: float = 0.01,
                             area_m2: Optional[float] = None,
                             length_col: str = 'length',
                             aperture_col: str = 'aperture'
                             ) -> CorrectionResults:
        """
        Valida dados usando relação D-L Scaling (Displacement-Length)
        
        Para juntas (opening mode): D_max = A × L^n onde n = 0.5
        Remove fraturas que violam os princípios de mecânica de fraturas.
        
        IMPORTANTE: Dados devem estar em METROS (length e aperture)
        
        Args:
            data: DataFrame com dados de fraturas (em METROS)
            rock_type: Tipo de rocha ('volcanic', 'basalt', 'trachyte', etc)
            tolerance: Fator de tolerância (default 5× = moderado)
            A_custom: Constante A customizada (se None, usa do rock_type)
            min_length_abs: Comprimento mínimo absoluto (metros)
            area_m2: Área da amostra em m² (para cálculo de intensidade)
            length_col: Nome da coluna de comprimento
            aperture_col: Nome da coluna de abertura
        
        Returns:
            CorrectionResults com dados filtrados
        """
        # Verificar colunas
        if length_col not in data.columns:
            raise ValueError(f"Coluna '{length_col}' não encontrada. Colunas: {data.columns.tolist()}")
        if aperture_col not in data.columns:
            raise ValueError(f"Coluna '{aperture_col}' não encontrada. Colunas: {data.columns.tolist()}")
        
        # Remover dados inválidos
        valid_mask = (data[length_col] > 0) & (data[aperture_col] > 0)
        data_valid = data[valid_mask].copy()
        n_original = len(data)
        n_invalid = n_original - len(data_valid)
        
        if n_invalid > 0:
            warnings.warn(f"{n_invalid} fraturas com L≤0 ou b≤0 removidas")
        
        # Aplicar filtro de comprimento mínimo
        data_valid = data_valid[data_valid[length_col] >= min_length_abs].copy()
        
        # Obter constante A
        if A_custom is not None:
            A = A_custom
        else:
            A = self.DL_SCALING_CONSTANTS.get(
                rock_type.lower(), 
                self.DL_SCALING_CONSTANTS['default']
            )
        
        # Verificar se A é válido
        if A is None or A <= 0:
            raise ValueError(f"Constante A inválida para rock_type='{rock_type}'. Use A_custom ou escolha outro tipo.")
        
        # Calcular abertura esperada: b_exp = A × L^0.5
        lengths = data_valid[length_col].values
        apertures_obs = data_valid[aperture_col].values
        
        apertures_exp = A * np.sqrt(lengths)
        
        # Calcular razão observado/esperado
        ratios = apertures_obs / apertures_exp
        
        # Aplicar critério de validação
        lower_bound = 1.0 / tolerance
        upper_bound = tolerance
        
        valid_dl_mask = (ratios >= lower_bound) & (ratios <= upper_bound)
        
        data_filtered = data_valid[valid_dl_mask].copy()
        data_filtered['dl_ratio'] = ratios[valid_dl_mask]
        data_filtered['aperture_expected_m'] = apertures_exp[valid_dl_mask]
        
        n_removed = len(data_valid) - len(data_filtered)
        removal_percent = (n_removed / n_original * 100) if n_original > 0 else 0
        
        # Calcular intensidades
        if area_m2 is not None and area_m2 > 0:
            P21_obs = data_valid[length_col].sum() / area_m2
            P21_filtered = data_filtered[length_col].sum() / area_m2 if len(data_filtered) > 0 else 0
            intensity_ratio = P21_filtered / P21_obs if P21_obs > 0 else 0
        else:
            P21_obs = data_valid[length_col].sum()
            P21_filtered = data_filtered[length_col].sum() if len(data_filtered) > 0 else 0
            intensity_ratio = P21_filtered / P21_obs if P21_obs > 0 else 0
        
        # Parâmetros
        parameters = {
            'rock_type': rock_type,
            'A_constant': A,
            'tolerance': tolerance,
            'min_length': min_length_abs,
            'n_exponent': 0.5,
            'bounds': [lower_bound, upper_bound]
        }
        
        # Calcular a_observado médio
        a_observed = float(np.mean(apertures_obs / np.sqrt(lengths))) if len(lengths) > 0 else 0
        
        # Validação
        if len(data_filtered) > 0:
            validation = {
                'n_removed': int(n_removed),
                'n_valid': int(len(data_filtered)),
                'removal_percent': float(removal_percent),
                'ratio_statistics': {
                    'min': float(np.min(ratios[valid_dl_mask])),
                    'q25': float(np.percentile(ratios[valid_dl_mask], 25)),
                    'median': float(np.median(ratios[valid_dl_mask])),
                    'mean': float(np.mean(ratios[valid_dl_mask])),
                    'q75': float(np.percentile(ratios[valid_dl_mask], 75)),
                    'max': float(np.max(ratios[valid_dl_mask]))
                },
                'a_observed': a_observed,
                'a_expected': A,
                'a_ratio': a_observed / A if A > 0 else 0,
                'data_quality': 'Good' if removal_percent < 30 else 'Poor (>30% removed)',
                'data_filtered': data_filtered,
                'data_removed': data_valid[~valid_dl_mask]
            }
        else:
            # TODOS os dados foram removidos - PROBLEMA!
            validation = {
                'n_removed': int(n_removed),
                'n_valid': 0,
                'removal_percent': 100.0,
                'a_observed': a_observed,
                'a_expected': A,
                'a_ratio': a_observed / A if A > 0 else 0,
                'data_quality': 'CRITICAL - All data removed!',
                'warning': f'Todos os dados foram removidos! Verifique: (1) Unidades estão em metros? (2) Tipo de rocha correto? a_observado={a_observed:.4f}, A_esperado={A}',
                'ratio_range_all': f'[{np.min(ratios):.3f}, {np.max(ratios):.3f}]',
                'data_filtered': data_filtered,
                'data_removed': data_valid
            }
        
        result = CorrectionResults(
            method='D-L Scaling Validation',
            intensity_observed=P21_obs,
            intensity_corrected=P21_filtered,
            correction_factor=intensity_ratio,
            n_data_original=n_original,
            n_data_used=len(data_filtered),
            parameters=parameters,
            validation=validation
        )
        
        self.correction_history.append(result)
        return result
    
    # ========================================================================
    # PIPELINES COMPLETOS
    # ========================================================================
    
    def scanline_pipeline(self,
                         data: pd.DataFrame,
                         scanline_azimuth: float,
                         scanline_length: float,
                         alpha: float,
                         l_min: float,
                         l_max: float,
                         length_col: str = 'length',
                         orientation_col: str = 'orientation') -> Dict:
        """
        Pipeline completo para dados de scanline 1D
        """
        results = {
            'data_type': 'scanline_1d',
            'corrections': []
        }
        
        # Step 1: Terzaghi
        terzaghi = self.terzaghi_correction(
            data=data,
            scanline_azimuth=scanline_azimuth,
            scanline_length=scanline_length
        )
        results['corrections'].append(terzaghi)
        results['P10_terzaghi'] = terzaghi.intensity_corrected
        
        # Step 2: Marrett
        marrett = self.marrett_correction(
            intensity_observed=terzaghi.intensity_corrected,
            alpha=alpha,
            l_min=l_min,
            l_max=l_max
        )
        results['corrections'].append(marrett)
        results['P10_marrett'] = marrett.intensity_corrected
        
        # Step 3: Converter P10 → P30
        geometric_factor = 2.5
        P30 = marrett.intensity_corrected * geometric_factor
        results['P30_estimated'] = P30
        results['geometric_factor'] = geometric_factor
        
        # Resumo
        P10_raw = len(data) / scanline_length
        total_factor = P30 / P10_raw if P10_raw > 0 else 0
        
        results['summary'] = {
            'P10_raw': P10_raw,
            'P10_terzaghi': terzaghi.intensity_corrected,
            'P10_final': marrett.intensity_corrected,
            'P30_estimated': P30,
            'total_increase_factor': total_factor,
            'total_increase_percent': (total_factor - 1) * 100
        }
        
        return results
    
    def framfrat_pipeline(self,
                         data: pd.DataFrame,
                         area_m2: float,
                         rock_type: str,
                         alpha: float,
                         l_min: float,
                         l_max: float,
                         length_col: str = 'length',
                         aperture_col: str = 'aperture',
                         apply_dl_validation: bool = True,
                         A_custom: Optional[float] = None,
                         tolerance: float = 5.0) -> Dict:
        """
        Pipeline completo para dados FRAMFRAT 2D
        
        IMPORTANTE: Dados devem estar em METROS
        """
        results = {
            'data_type': 'framfrat_2d',
            'corrections': [],
            'input_params': {
                'area_m2': area_m2,
                'rock_type': rock_type,
                'alpha': alpha,
                'l_min': l_min,
                'l_max': l_max,
                'apply_dl_validation': apply_dl_validation
            }
        }
        
        data_working = data.copy()
        
        # Step 1: D-L Scaling Validation (opcional)
        if apply_dl_validation and aperture_col in data.columns:
            dl_result = self.dl_scaling_validation(
                data=data_working,
                rock_type=rock_type,
                area_m2=area_m2,
                length_col=length_col,
                aperture_col=aperture_col,
                A_custom=A_custom,
                tolerance=tolerance
            )
            results['corrections'].append(dl_result)
            data_working = dl_result.validation['data_filtered']
            
            results['dl_validation'] = {
                'applied': True,
                'n_original': dl_result.n_data_original,
                'n_valid': dl_result.n_data_used,
                'n_removed': dl_result.validation['n_removed'],
                'removal_percent': dl_result.validation['removal_percent'],
                'a_observed': dl_result.validation.get('a_observed', 0),
                'a_expected': dl_result.parameters['A_constant'],
                'data_quality': dl_result.validation.get('data_quality', 'Unknown')
            }
        else:
            results['dl_validation'] = {'applied': False}
        
        # Calcular P20 e P21 observados
        n_fracs = len(data_working)
        
        if n_fracs == 0:
            # Nenhum dado após D-L validation
            results['P20_observed'] = 0
            results['P21_observed'] = 0
            results['P20_corrected'] = 0
            results['P21_corrected'] = 0
            results['P32_estimated'] = 0
            
            results['summary'] = {
                'P20_raw': len(data) / area_m2,
                'P21_raw': data[length_col].sum() / area_m2,
                'P20_final': 0,
                'P21_final': 0,
                'P32_estimated': 0,
                'total_increase_factor': 0,
                'total_increase_percent': -100,
                'warning': 'Todos os dados foram removidos na validação D-L!'
            }
            return results
        
        total_length = data_working[length_col].sum()
        
        P20_obs = n_fracs / area_m2
        P21_obs = total_length / area_m2
        
        # Step 2: Marrett
        marrett = self.marrett_correction(
            intensity_observed=P21_obs,
            alpha=alpha,
            l_min=l_min,
            l_max=l_max
        )
        results['corrections'].append(marrett)
        
        P21_marrett = marrett.intensity_corrected
        P20_marrett = P20_obs * marrett.correction_factor
        
        results['P20_observed'] = P20_obs
        results['P21_observed'] = P21_obs
        results['P20_corrected'] = P20_marrett
        results['P21_corrected'] = P21_marrett
        
        # Estimar P32 (fator C23 = π/2 para isotropia)
        C23 = np.pi / 2
        P32_estimated = P21_marrett * C23
        results['P32_estimated'] = P32_estimated
        results['C23_factor'] = C23
        
        # Resumo
        P21_raw = data[length_col].sum() / area_m2
        total_factor = P21_marrett / P21_raw if P21_raw > 0 else 0
        
        results['summary'] = {
            'P20_raw': len(data) / area_m2,
            'P21_raw': P21_raw,
            'P20_final': P20_marrett,
            'P21_final': P21_marrett,
            'P32_estimated': P32_estimated,
            'total_increase_factor': total_factor,
            'total_increase_percent': (total_factor - 1) * 100 if total_factor > 0 else -100
        }
        
        return results
    
    def get_correction_summary(self) -> pd.DataFrame:
        """Retorna DataFrame com resumo de todas as correções aplicadas"""
        if not self.correction_history:
            return pd.DataFrame()
        
        summary_data = []
        for i, corr in enumerate(self.correction_history):
            summary_data.append({
                'step': i + 1,
                'method': corr.method,
                'intensity_obs': corr.intensity_observed,
                'intensity_corr': corr.intensity_corrected,
                'factor': corr.correction_factor,
                'increase_%': (corr.correction_factor - 1) * 100,
                'n_data': corr.n_data_used
            })
        
        return pd.DataFrame(summary_data)
