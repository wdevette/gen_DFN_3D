"""
intensity_spacing.py - VERSÃO CORRIGIDA

CORREÇÕES IMPLEMENTADAS:
1. Conversão 2D→3D com limites razoáveis de P30
2. Validação de valores para evitar números irreais
3. Warnings informativos quando limites são aplicados
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class Intensidade_Fraturas:
    """Classe para armazenar todos os índices refentes as intensidades das fraturas calculados"""
    # Índices 1D (scanline)
    P10: Optional[float] = None  # fraturas/m
    P11: Optional[float] = None  # m/m (adimensional)
    
    # Índices 2D (área/imagem) 
    P20: Optional[float] = None  # fraturas/m²
    P21: Optional[float] = None  # m/m²
    P22: Optional[float] = None  # m²/m² (adimensional)
    
    # Índices 3D (volume)
    P30: Optional[float] = None  # fraturas/m³
    P31: Optional[float] = None  # m/m³
    P32: Optional[float] = None  # m²/m³
    P33: Optional[float] = None  # m³/m³ (porosidade)
    
    # Metadados
    source: str = ""  # "scanline", "framfrat", "manual"
    mean_length_m: Optional[float] = None  # comprimento médio em metros
    mean_aperture_m: Optional[float] = None  # abertura média em metros
    powerlaw_exponent: Optional[float] = None  # expoente da lei de potência

    area_m2: Optional[float] = None  # área em m²
    
    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class IntensitySpacingAnalyzer:
    """
    Análise de intensidade e espaçamento
        
    Índices de intensidades das fraturas (Dershowitz & Herda 1992):
    - P10: fraturas/m (1D - linear)
    - P20: fraturas/m² (2D - areal)
    - P21: m/m² (2D - comprimento de traço por área)
    - P30: fraturas/m³ (3D - volumétrica)
    - P32: m²/m³ (3D - área de fratura por volume)
    - P33: m³/m³ (3D - volume de fratura por volume total)
    
    Fórmulas (com dados em METROS):
    - P10 = N / L  [fraturas/m]
    - P21 = Σ(comprimentos) / Área  [m/m²]
    - P32 = Σ(áreas de fraturas) / Volume  [m²/m³]
    
    LIMITES APLICADOS (NOVO):
    - P30 máximo: 500 fraturas/m³ (valores maiores indicam erro)
    - P32 máximo: 10 m²/m³ (valores maiores indicam erro)
    """
    
    # Limites máximos razoáveis
    MAX_P30 = 500.0  # fraturas/m³
    MAX_P32 = 10.0   # m²/m³
    
    def __init__(self):
        self.indices = Intensidade_Fraturas()
    
    def calculate_from_framfrat(self, data: pd.DataFrame, area_m2: float) -> Dict:
        """
        Calcula índices P2x a partir de dados FRAMFRAT
        
        Args:
            data: DataFrame com coluna 'length' em METROS (abertura é OPCIONAL)
            area_m2: Área da imagem em METROS QUADRADOS
        
        Returns:
            Dicionário com P20, P21, P22 calculados
        
        Fórmulas:
        P20 = N / A (número de fraturas por área)
        P21 = ΣL / A (comprimento total por área)  
        P22 = ΣAf / A (área total de fraturas por área) - REQUER ABERTURA
        """
        # Dados já estão em metros
        lengths_m = data['length'].values
        
        # Verificar se abertura está disponível
        has_aperture = 'aperture' in data.columns
        
        if has_aperture:
            apertures_m = data['aperture'].values
        
        # Número de fraturas
        n_fractures = len(data)
        
        # P20: número por área
        self.indices.P20 = n_fractures / area_m2
        
        # P21: comprimento total por área
        total_length_m = np.sum(lengths_m)
        self.indices.P21 = total_length_m / area_m2
        
        # P22: área total de fraturas por área (APENAS SE ABERTURA DISPONÍVEL)
        if has_aperture:
            fracture_areas_m2 = lengths_m * apertures_m
            total_fracture_area_m2 = np.sum(fracture_areas_m2)
            self.indices.P22 = total_fracture_area_m2 / area_m2
            self.indices.mean_aperture_m = np.mean(apertures_m)
        else:
            self.indices.P22 = None  # Não disponível
            self.indices.mean_aperture_m = None
        
        # Metadados
        self.indices.source = "framfrat"
        self.indices.mean_length_m = np.mean(lengths_m)
        self.indices.area_m2 = area_m2
        self.indices.has_aperture = has_aperture
        
        return self.indices.to_dict()
    
    def calculate_from_scanline(self, P10: Optional[float] = None,
                               P11: Optional[float] = None,
                               n_fractures: Optional[int] = None,
                               scanline_length_m: Optional[float] = None,
                               mean_length_m: Optional[float] = None,
                               mean_aperture_m: Optional[float] = None) -> 'Intensidade_Fraturas':
        """
        Calcula índices P1x a partir de dados de scanline
        
        Args:
            P10: Intensidade P10 direta (fraturas/m)
            P11: Intensidade P11 direta (adimensional)
            n_fractures: Número de fraturas observadas
            scanline_length_m: Comprimento da scanline em metros
            mean_length_m: Comprimento médio das fraturas em metros
            mean_aperture_m: Abertura média em metros
        
        Returns:
            Intensidade_Fraturas() com P10 e P11 calculados
        """
        # Calcular P10
        if P10 is not None:
            self.indices.P10 = P10
        elif n_fractures is not None and scanline_length_m is not None:
            self.indices.P10 = n_fractures / scanline_length_m
        
        # Calcular P11
        if P11 is not None:
            self.indices.P11 = P11
        elif self.indices.P10 is not None and mean_length_m is not None:
            self.indices.P11 = self.indices.P10 * mean_length_m
        
        self.indices.source = "scanline"
        self.indices.mean_length_m = mean_length_m
        self.indices.mean_aperture_m = mean_aperture_m
        
        return self.indices
    
    def convert_1d_to_3d(self, orientation_correction: float = 1.0,
                        shape_factor: float = np.pi/4) -> 'Intensidade_Fraturas':
        """
        Converte índices 1D (P10, P11) para 3D (P30, P31, P32, P33)
        Baseado em Dershowitz & Herda (1992) e correções de Ortega et al. (2006)
        
        Args:
            orientation_correction: Fator de correção para orientação C13 (default=1.0)
            shape_factor: Fator de forma das fraturas (π/4 para circular)
        
        Returns:
            Intensidade_Fraturas() com valores 3D calculados
        """
        if self.indices.P10 is None:
            raise ValueError("P10 necessário para conversão 1D→3D")
        
        # P32: área por volume (relação de Wang 2005)
        P32_calc = 2 * self.indices.P10 * orientation_correction
        
        # APLICAR LIMITE
        if P32_calc > self.MAX_P32:
            warnings.warn(f"P32 calculado ({P32_calc:.2f}) excede limite ({self.MAX_P32}). Limitando.")
            P32_calc = self.MAX_P32
        
        self.indices.P32 = P32_calc
        
        # Estimar área média das fraturas
        if self.indices.mean_length_m is not None:
            mean_length_m = self.indices.mean_length_m
            mean_area_m2 = shape_factor * mean_length_m**2
        else:
            warnings.warn("Usando área média padrão de 0.01 m²")
            mean_area_m2 = 0.01
        
        # P30: número por volume
        P30_calc = self.indices.P32 / mean_area_m2
        
        # APLICAR LIMITE
        if P30_calc > self.MAX_P30:
            warnings.warn(f"P30 calculado ({P30_calc:.1f}) excede limite ({self.MAX_P30}). Limitando.")
            P30_calc = self.MAX_P30
        
        self.indices.P30 = P30_calc
        
        # P31: comprimento por volume
        if self.indices.mean_length_m is not None:
            mean_length_m = self.indices.mean_length_m
            self.indices.P31 = self.indices.P30 * mean_length_m
        
        # P33: volume por volume (porosidade)
        if self.indices.mean_aperture_m is not None:
            mean_aperture_m = self.indices.mean_aperture_m
            self.indices.P33 = self.indices.P32 * mean_aperture_m
        
        return self.indices
    
    def convert_2d_to_3d(self, thickness_correction: float = 1.0,
                        stereology_factor: float = 1.5) -> 'Intensidade_Fraturas':
        """
        Converte índices 2D (P20, P21, P22) para 3D (P30, P31, P32, P33)
        Baseado em relações estereológicas de Mauldon et al. (2001)
        
        CORRIGIDO: Aplica limites razoáveis para evitar valores irreais
        
        Args:
            thickness_correction: Correção para espessura da amostra (default=1.0)
            stereology_factor: Fator estereológico (1.5 para distribuição isotrópica)
        
        Returns:
            Intensidade_Fraturas() com valores 3D calculados
        """
        if self.indices.P21 is None or self.indices.P20 is None:
            raise ValueError("P20 e P21 necessários para conversão 2D→3D")
        
        # P32: área por volume (Mauldon 2001)
        P32_calc = self.indices.P21 * np.pi / 2 * thickness_correction
        
        # APLICAR LIMITE
        if P32_calc > self.MAX_P32:
            original_P32 = P32_calc
            P32_calc = self.MAX_P32
            warnings.warn(f"P32 calculado ({original_P32:.2f} m²/m³) excede limite razoável. "
                         f"Limitado para {P32_calc} m²/m³. "
                         f"Verifique a área da amostra (muito pequena pode gerar valores irreais).")
        
        self.indices.P32 = P32_calc
        
        # P30: número por volume
        P30_calc = self.indices.P20 * stereology_factor * thickness_correction
        
        # APLICAR LIMITE
        if P30_calc > self.MAX_P30:
            original_P30 = P30_calc
            P30_calc = self.MAX_P30
            warnings.warn(f"P30 calculado ({original_P30:.1f} fraturas/m³) excede limite razoável. "
                         f"Limitado para {P30_calc} fraturas/m³. "
                         f"Verifique a área da amostra (muito pequena pode gerar valores irreais).")
        
        self.indices.P30 = P30_calc
        
        # P31: comprimento por volume
        if self.indices.mean_length_m is not None:
            mean_length_m = self.indices.mean_length_m
            self.indices.P31 = self.indices.P30 * mean_length_m
        else:
            # Estimar do P21/P20
            mean_length_est = self.indices.P21 / max(self.indices.P20, 1e-10)
            self.indices.P31 = self.indices.P30 * mean_length_est
        
        # P33: volume por volume (porosidade)
        if self.indices.mean_aperture_m is not None:
            mean_aperture_m = self.indices.mean_aperture_m
            self.indices.P33 = self.indices.P32 * mean_aperture_m
        
        print(f"Conversão 2D→3D concluída:")
        print(f"  P20={self.indices.P20:.1f} fraturas/m² → P30={self.indices.P30:.1f} fraturas/m³")
        print(f"  P21={self.indices.P21:.2f} m/m² → P32={self.indices.P32:.3f} m²/m³")
        
        return self.indices
    
    def calculate_n_fractures_for_dfn(self, volume_m3: float, 
                                      powerlaw_params: Dict = None) -> int:
        """
        Calcula número de fraturas necessárias para gerar um DFN
        que corresponda às intensidades calculadas
        
        CORRIGIDO: Aplica limites razoáveis
        
        Args:
            volume_m3: Volume do domínio em m³
            powerlaw_params: Parâmetros opcionais da power-law (exponent, x_min, x_max)
        
        Returns:
            Número de fraturas a gerar
        """
        if self.indices.P30 is not None:
            n_fractures = int(self.indices.P30 * volume_m3)
        elif self.indices.P32 is not None and self.indices.mean_length_m is not None:
            # Estimar P30 a partir de P32 e tamanho médio
            mean_area = np.pi / 4 * self.indices.mean_length_m**2
            p30_est = self.indices.P32 / mean_area
            n_fractures = int(p30_est * volume_m3)
        else:
            # Fallback: usar P20 se disponível
            if self.indices.P20 is not None:
                # Conversão aproximada
                thickness = volume_m3 ** (1/3)
                p30_approx = self.indices.P20 / thickness
                n_fractures = int(p30_approx * volume_m3)
            else:
                raise ValueError("Necessário P30 ou P32+mean_length para calcular número de fraturas")
        
        # Aplicar correção de Marrett se parâmetros disponíveis
        if powerlaw_params and 'exponent' in powerlaw_params:
            n_fractures = self._apply_marrett_correction(
                n_fractures, powerlaw_params, volume_m3
            )
        
        # LIMITE ABSOLUTO
        MAX_FRACTURES = 100000
        if n_fractures > MAX_FRACTURES:
            warnings.warn(f"Número de fraturas ({f'{n_fractures:,}'.replace(",",".")}) excede limite. Limitando para {MAX_FRACTURES:,}.")
            n_fractures = MAX_FRACTURES
        
        return max(1, n_fractures)
    
    def _apply_marrett_correction(self, n_base: int, params: Dict, volume: float) -> int:
        """
        Aplica correção de Marrett (1996) para distribuição power-law truncada
        """
        alpha = params['exponent']
        x_min = params.get('x_min', 0.01)
        x_max = params.get('x_max', volume**(1/3) / 2)
        
        if alpha != 1:
            f_trunc = (x_max**(1-alpha) - x_min**(1-alpha)) / ((1-alpha) * x_max**(1-alpha))
        else:
            f_trunc = np.log(x_max/x_min) / x_max
        
        # Limitar fator de correção
        f_trunc = min(f_trunc, 10.0)
        
        return int(n_base * f_trunc)
    
    # MÉTODOS ORIGINAIS MANTIDOS PARA COMPATIBILIDADE
    def calculate_p10(self, data: pd.DataFrame, threshold: float, 
                     area: float) -> float:
        """
        Calcula intensidade P10 (fraturas/m) para um limiar de tamanho
        MANTIDO PARA COMPATIBILIDADE - área agora em m²
        """
        # Filtrar por tamanho
        filtered = data[data['length'] >= threshold]
        n_fractures = len(filtered)
        
        # Comprimento de amostragem equivalente
        sample_length = 4 * np.sqrt(area)
        
        # P10
        p10 = n_fractures / sample_length
        
        return p10
    
    def calculate_p10_scanline(self, data: pd.DataFrame, threshold: float,
                               scanline_length: float) -> float:
        """
        Calcula P10 diretamente de dados de scanline
        MANTIDO PARA COMPATIBILIDADE
        """
        # Filtrar por tamanho
        if 'length' in data.columns:
            filtered = data[data['length'] >= threshold]
        elif 'aperture' in data.columns:
            filtered = data[data['aperture'] >= threshold/100]
        else:
            # Sem coluna de tamanho, retorna todos os dados
            filtered = data
        
        n_fractures = len(filtered)
        
        # P10
        p10 = n_fractures / scanline_length
        
        return p10
    
    def calculate_p21(self, data: pd.DataFrame, area: float) -> float:
        """
        Calcula intensidade P21 (m/m²) - comprimento total por área
        MANTIDO PARA COMPATIBILIDADE
        """
        total_length = data['length'].sum()
        area_m2 = area
        p21 = total_length / area_m2
        
        return p21
    
    def calculate_p32(self, data: pd.DataFrame, volume: float) -> float:
        """
        Calcula intensidade P32 (m²/m³) - área total por volume
        MANTIDO PARA COMPATIBILIDADE
        """
        if 'area' not in data.columns:
            data = data.copy()
            data['area'] = data['length']**2 * np.pi / 4
        
        total_area = data['area'].sum()
        volume_m3 = volume
        p32 = total_area / volume_m3
        
        return p32
    
    def calculate_average_spacing(self, p10: float) -> float:
        """
        Calcula espaçamento médio a partir de P10
        MANTIDO PARA COMPATIBILIDADE
        """
        if p10 > 0:
            return 1.0 / p10
        else:
            return np.inf
    
    def normalized_comparison(self, data1: pd.DataFrame, data2: pd.DataFrame,
                            threshold: float, area1: float, 
                            length2: float) -> Dict:
        """
        Compara intensidades normalizadas entre duas fontes
        MANTIDO PARA COMPATIBILIDADE
        """
        p10_1 = self.calculate_p10(data1, threshold, area1)
        p10_2 = self.calculate_p10_scanline(data2, threshold, length2)
        
        spacing_1 = self.calculate_average_spacing(p10_1)
        spacing_2 = self.calculate_average_spacing(p10_2)
        
        return {
            'threshold': threshold,
            'p10_framfrat': p10_1,
            'p10_scanline': p10_2,
            'spacing_framfrat': spacing_1,
            'spacing_scanline': spacing_2,
            'ratio_p10': p10_1 / p10_2 if p10_2 > 0 else np.inf,
            'ratio_spacing': spacing_1 / spacing_2 if spacing_2 > 0 else np.inf
        }
    
    def estimate_representative_volume(self, data: pd.DataFrame, 
                                      target_cv: float = 0.1) -> float:
        """
        Estima volume representativo elementar (REV)
        MANTIDO PARA COMPATIBILIDADE
        """
        lengths = data['length'].values
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        cv = std_length / mean_length if mean_length > 0 else 1
        
        n_required = (cv / target_cv)**2
        current_n = len(lengths)
        
        scale_factor = n_required / current_n
        rev = scale_factor * mean_length**3
        
        return rev


class MarrettCorrection:
    """
    Implementa correções de Marrett (1996) para distribuições power-law truncadas
    """
    
    @staticmethod
    def truncation_correction(alpha: float, x_min: float, x_max: float) -> float:
        """
        Fator de correção para distribuição power-law truncada
        
        Args:
            alpha: Expoente da lei de potência
            x_min: Tamanho mínimo (m)
            x_max: Tamanho máximo (m)
        
        Returns:
            Fator de correção multiplicativo
        
        Ref: Marrett (1996) Eq. 4
        """
        if alpha == 1:
            return np.log(x_max/x_min)
        else:
            num = x_max**(1-alpha) - x_min**(1-alpha)
            den = (1-alpha) * x_max**(1-alpha)
            return num / den
    
    @staticmethod
    def mean_size_truncated(alpha: float, x_min: float, x_max: float) -> float:
        """
        Tamanho médio para distribuição power-law truncada
        
        Ref: Marrett (1996) Eq. 5
        """
        if alpha == 2:
            return x_min * x_max / (x_min + x_max) * np.log(x_max/x_min)
        else:
            num = (2-alpha) * (x_max**(2-alpha) - x_min**(2-alpha))
            den = (1-alpha) * (x_max**(1-alpha) - x_min**(1-alpha))
            return num / den


class OrtegaCorrection:
    """
    Implementa correções size-cognizant de Ortega et al. (2006)
    """
    
    @staticmethod
    def size_dependent_p10(fractures: pd.DataFrame, 
                          threshold: float,
                          sample_length: float) -> float:
        """
        P10 size-cognizant considerando apenas fraturas acima do threshold
        
        Ref: Ortega et al. (2006) Eq. 2
        """
        if 'length' in fractures.columns:
            size_col = 'length'
        elif 'aperture' in fractures.columns:
            size_col = 'aperture'
        else:
            raise ValueError("DataFrame deve ter coluna 'length' ou 'aperture'")
        
        # Filtrar por tamanho
        large_fractures = fractures[fractures[size_col] >= threshold]
        n_large = len(large_fractures)
        
        # P10 size-cognizant
        p10_sc = n_large / sample_length
        
        return p10_sc
    
    @staticmethod
    def scale_correction_factor(scale1: float, scale2: float, 
                              alpha: float) -> float:
        """
        Fator de correção entre diferentes escalas de observação
        
        Ref: Ortega et al. (2006) Fig. 5
        """
        scale_ratio = scale2 / scale1
        
        # Correção empírica de Ortega
        if alpha < 1:
            return scale_ratio ** (1 - alpha/2)
        else:
            return scale_ratio ** (2 - alpha)

