"""
DFN Generator - Versão Corrigida v2
Compatível com app.py original + novas funcionalidades de intensidade

CORREÇÕES v2:
1. Retorna n_fractures_calculated para estatísticas teóricas corretas
2. Limite máximo de fraturas para evitar crash de WebSocket/memória
3. Distribuição correta entre famílias com validação
4. Suporte a exportação completa de atributos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
#import warnings

#Tentar importar streamlit (opcional para uso standalone)
from func_tools import show_error, show_info, show_success
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# # Função auxiliar para alertas
# def show_alert(msg):
#     if HAS_STREAMLIT:
#         st.warning(msg)
        
#     else:
#         warnings.warn(msg)


# Importar classes do módulo intensity_spacing para integração
try:
    from .intensity_spacing import Intensidade_Fraturas, IntensitySpacingAnalyzer, MarrettCorrection
except ImportError:
    try:
        from modules.intensity_spacing import Intensidade_Fraturas, IntensitySpacingAnalyzer, MarrettCorrection
    except ImportError:
        try:
            from intensity_spacing import Intensidade_Fraturas, IntensitySpacingAnalyzer, MarrettCorrection
        except ImportError:
            Intensidade_Fraturas = None
            IntensitySpacingAnalyzer = None
            MarrettCorrection = None


@dataclass
class Fracture2D:
    """Representa uma fratura 2D - dimensões em METROS"""
    x1: float  # m
    y1: float  # m
    x2: float  # m
    y2: float  # m
    length: float  # m
    aperture: float  # m
    orientation: float  # graus
    family: int = 0  # ID da família/set
    
    def to_dict(self):
        return {
            'x1': self.x1, 'y1': self.y1,
            'x2': self.x2, 'y2': self.y2,
            'length': self.length,
            'aperture': self.aperture,
            'orientation': self.orientation,
            'family': self.family
        }


@dataclass
class Fracture3D:
    """Representa uma fratura 3D como um disco - dimensões em METROS"""
    center: np.ndarray  # [x, y, z] em metros
    normal: np.ndarray  # vetor normal
    radius: float  # m
    aperture: float  # m
    dip: float  # graus
    dip_direction: float  # graus
    family: int = 0  # ID da família/set
    
    def to_dict(self):
        """Converte para dicionário com TODOS os atributos para exportação"""
        area = np.pi * self.radius**2
        azimuth = (self.dip_direction + 90) % 360  # Azimute da fratura (strike)
        
        return {
            'center': self.center.tolist(),
            'center_x': float(self.center[0]),
            'center_y': float(self.center[1]),
            'center_z': float(self.center[2]),
            'normal': self.normal.tolist(),
            'normal_x': float(self.normal[0]),
            'normal_y': float(self.normal[1]),
            'normal_z': float(self.normal[2]),
            'radius': float(self.radius),
            'diameter': float(self.radius * 2),  # Comprimento/diâmetro
            'aperture': float(self.aperture),
            'aperture_mm': float(self.aperture * 1000),  # Abertura em mm
            'dip': float(self.dip),
            'dip_direction': float(self.dip_direction),
            'azimuth': float(azimuth),  # Strike
            'area': float(area),  # Área do disco em m²
            'family': int(self.family)
        }


@dataclass
class FractureFamily:
    """Representa uma família/set de fraturas com orientação preferencial"""
    orientation_mean: float  # graus (para 2D) ou dip para 3D
    orientation_std: float   # desvio padrão
    dip_dir_mean: float = 0.0  # apenas para 3D
    dip_dir_std: float = 20.0  # apenas para 3D
    weight: float = 1.0  # peso relativo da família


@dataclass
class DFNGenerationResult:
    """Resultado da geração de DFN com metadados completos"""
    fractures: List  # Lista de Fracture2D ou Fracture3D
    n_fractures_calculated: int  # Número teórico calculado (antes do limite)
    n_fractures_generated: int  # Número efetivamente gerado (após limite)
    n_fractures_limited: bool  # Se houve limitação
    scale_factor: float  # Fator de escala (calculated / generated)
    params: Dict  # Parâmetros usados
    domain_size: Tuple  # Tamanho do domínio
    families: List[FractureFamily]  # Famílias utilizadas
    
    def get_dataframe(self) -> pd.DataFrame:
        """Retorna DataFrame com todas as fraturas"""
        return pd.DataFrame([f.to_dict() for f in self.fractures])
    
    def get_theoretical_stats(self, for_n_fractures: Optional[int] = None) -> Dict:
        """
        Calcula estatísticas teóricas escalonadas para o número correto de fraturas
        
        Args:
            for_n_fractures: Se None, usa n_fractures_calculated
        
        Returns:
            Dicionário com estatísticas escalonadas
        """
        if for_n_fractures is None:
            for_n_fractures = self.n_fractures_calculated
        
        scale = for_n_fractures / max(self.n_fractures_generated, 1)
        df = self.get_dataframe()
        
        # Estatísticas médias (não escalam)
        mean_radius = df['radius'].mean()
        mean_aperture = df['aperture'].mean()
        mean_area = df['area'].mean() if 'area' in df.columns else np.pi * mean_radius**2
        
        # Estatísticas que escalam com o número de fraturas
        total_area = mean_area * for_n_fractures
        
        # Volume do domínio
        if len(self.domain_size) == 3:
            volume = self.domain_size[0] * self.domain_size[1] * self.domain_size[2]
        else:
            volume = self.domain_size[0] * self.domain_size[1]  # 2D
        
        # Índices de intensidade teóricos
        P30_theoretical = for_n_fractures / volume
        P32_theoretical = total_area / volume
        
        # Porosidade teórica
        porosity_theoretical = (mean_aperture * total_area) / volume
        
        # Permeabilidade (lei cúbica - média não escala)
        k_theoretical = (mean_aperture**3) / 12
        
        return {
            'n_fractures_theoretical': for_n_fractures,
            'n_fractures_generated': self.n_fractures_generated,
            'scale_factor': scale,
            'mean_radius_m': mean_radius,
            'mean_aperture_m': mean_aperture,
            'mean_aperture_mm': mean_aperture * 1000,
            'mean_area_m2': mean_area,
            'total_area_m2': total_area,
            'volume_m3': volume,
            'P30_theoretical': P30_theoretical,
            'P32_theoretical': P32_theoretical,
            'porosity_theoretical': porosity_theoretical,
            'porosity_percent': porosity_theoretical * 100,
            'permeability_m2': k_theoretical,
            'permeability_mD': k_theoretical * 1e12,
            'P10_equiv': P30_theoretical * (2 * mean_radius)
        }


class DFNGenerator:
    """
    Gerador de Discrete Fracture Networks
    
    VERSÃO CORRIGIDA v2 com:
    - Retorna n_fractures_calculated para estatísticas corretas
    - Limite máximo de fraturas (MAX_FRACTURES)
    - Compatibilidade retroativa com app.py original
    - Suporte a famílias múltiplas com distribuição garantida
    - Exportação completa de atributos
    """
    
    # Valores default para parâmetros power-law (geológicos típicos)
    DEFAULT_EXPONENT = 2.0
    DEFAULT_X_MIN = 0.01  # 1 cm em metros
    DEFAULT_X_MAX = 10.0  # 10 m
    DEFAULT_COEFFICIENT = 100
    
    # LIMITE MÁXIMO DE FRATURAS para evitar crash
    MAX_FRACTURES_2D = 50000
    MAX_FRACTURES_3D = 5000  # 5k para geração (mesmo que visualização)
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        self.intensity_analyzer = IntensitySpacingAnalyzer() if IntensitySpacingAnalyzer else None
        
        # Armazenar metadados da última geração
        self._last_generation_metadata = {}
    
    def generate_3d_dfn(self, 
                        params: Dict, 
                        domain_size: Tuple[float, float, float],
                        n_fractures: Optional[int] = None,
                        families: Optional[List[FractureFamily]] = None,
                        intensi_fract: Optional[object] = None,
                        return_result: bool = False) -> Union[List[Fracture3D], DFNGenerationResult]:
        """
        Gera DFN 3D baseado em parâmetros estatísticos
        
        Args:
            params: Parâmetros da distribuição (exponent, x_min, coefficient, etc)
            domain_size: (largura, altura, profundidade) do domínio em METROS
            n_fractures: Número de fraturas (opcional - pode ser calculado automaticamente)
            families: Lista de famílias de fraturas (opcional)
            intensi_fract: Índices de intensidade para cálculo automático (opcional)
            return_result: Se True, retorna DFNGenerationResult com metadados
        
        Returns:
            Lista de Fracture3D ou DFNGenerationResult
        """
        width, height, depth = domain_size
        volume = width * height * depth
        
        # Garantir parâmetros mínimos
        params = self._ensure_default_params(params)
        
        # Configurar famílias
        if families is None:
            families = self._create_default_families_3d(params)
        
        # Debug: mostrar famílias configuradas
        print(f"Famílias configuradas: {len(families)}")
        for i, fam in enumerate(families):
            print(f"  Família {i}: dip={fam.orientation_mean:.1f}°, dip_dir={fam.dip_dir_mean:.1f}°, peso={fam.weight:.2%}")
        
        # Determinar número de fraturas CALCULADO (teórico)
        n_fractures_calculated = n_fractures
        if n_fractures_calculated is None:
            n_fractures_calculated = self._calculate_n_fractures_3d(params, volume, intensi_fract)
        
        if n_fractures_calculated is None or n_fractures_calculated <= 0:
            n_fractures_calculated = 100
            st.markdown("")
            show_error("⚠️ Usando número padrão de 100 fraturas. Configure índices de intensidade para cálculo automático.")
            st.markdown("")

        # Armazenar número calculado ANTES do limite
        n_fractures_original = n_fractures_calculated
        n_fractures_limited = False
        
        # APLICAR LIMITE MÁXIMO para evitar crash de memória/WebSocket
        if n_fractures_calculated > self.MAX_FRACTURES_3D:
            n_fractures_limited = True
            n_fractures_to_generate = self.MAX_FRACTURES_3D
            print(f"⚠️ LIMITE APLICADO: {n_fractures_original} → {n_fractures_to_generate}")
            # Não mostrar alerta - estatísticas serão calculadas com n_fractures_calculated
        else:
            n_fractures_to_generate = n_fractures_calculated
        
        # Armazenar metadados para acesso posterior
        self._last_generation_metadata = {
            'n_fractures_calculated': n_fractures_original,
            'n_fractures_generated': n_fractures_to_generate,
            'n_fractures_limited': n_fractures_limited,
            'scale_factor': n_fractures_original / max(n_fractures_to_generate, 1)
        }
        
        # Salvar no session_state do Streamlit se disponível
        if HAS_STREAMLIT:
            st.session_state.dfn_3d_metadata = self._last_generation_metadata
        
        print(f"Gerando {n_fractures_to_generate} fraturas 3D (calculado: {n_fractures_original})")
        
        # Distribuir entre famílias
        total_weight = sum(f.weight for f in families)
        if total_weight <= 0:
            total_weight = len(families)
            for f in families:
                f.weight = 1.0
        
        fractures_per_family = [
            max(1, int(n_fractures_to_generate * (f.weight / total_weight)))
            for f in families
        ]
        
        # Debug: mostrar distribuição planejada
        print(f"Distribuição planejada por família:")
        for i, n_fam in enumerate(fractures_per_family):
            print(f"  Família {i}: {n_fam} fraturas")
        
        fractures = []
        
        for family_id, (family, n_fam) in enumerate(zip(families, fractures_per_family)):
            print(f"  Gerando {n_fam} fraturas para família {family_id}...")
            
            for _ in range(n_fam):
                # Gerar raio (metade do comprimento)
                length = self._sample_power_law(
                    params['exponent'],
                    params['x_min'],
                    params.get('x_max', min(domain_size)/4)
                )
                radius = length / 2
                
                # Gerar abertura
                if 'g' in params and 'm' in params:
                    aperture = params['g'] * length ** params['m']
                else:
                    aperture = radius * 0.002
                
                # Gerar orientação com base nos parâmetros da FAMÍLIA específica
                dip = np.random.normal(family.orientation_mean, family.orientation_std)
                dip = np.clip(dip, 0, 90)
                dip_direction = np.random.normal(family.dip_dir_mean, family.dip_dir_std)
                
                # Calcular vetor normal
                dip_rad = np.radians(dip)
                dd_rad = np.radians(dip_direction)
                normal = np.array([
                    np.sin(dip_rad) * np.sin(dd_rad),
                    np.sin(dip_rad) * np.cos(dd_rad),
                    np.cos(dip_rad)
                ])
                
                # Centro aleatório
                center = np.array([
                    np.random.uniform(0, width),
                    np.random.uniform(0, height),
                    np.random.uniform(0, depth)
                ])
                
                fracture = Fracture3D(
                    center=center,
                    normal=normal,
                    radius=radius,
                    aperture=aperture,
                    dip=dip,
                    dip_direction=dip_direction,
                    family=family_id
                )
                fractures.append(fracture)
        
        # Debug: verificar distribuição final
        family_counts = {}
        for f in fractures:
            family_counts[f.family] = family_counts.get(f.family, 0) + 1
        print(f"Distribuição final de fraturas por família: {family_counts}")
        
        # Retornar resultado
        if return_result:
            return DFNGenerationResult(
                fractures=fractures,
                n_fractures_calculated=n_fractures_original,
                n_fractures_generated=len(fractures),
                n_fractures_limited=n_fractures_limited,
                scale_factor=n_fractures_original / max(len(fractures), 1),
                params=params,
                domain_size=domain_size,
                families=families
            )
        
        return fractures
    
    def get_last_generation_metadata(self) -> Dict:
        """Retorna metadados da última geração"""
        return self._last_generation_metadata
    
    def generate_2d_dfn(self, 
                        params: Dict, 
                        domain_size: Tuple[float, float],
                        n_fractures: Optional[int] = None,
                        families: Optional[List[FractureFamily]] = None,
                        intensi_fract: Optional[object] = None) -> List[Fracture2D]:
        """
        Gera DFN 2D baseado em parâmetros estatísticos
        """
        width, height = domain_size
        area = width * height
        
        params = self._ensure_default_params(params)
        
        if families is None:
            families = self._create_default_families_2d(params)
        
        if n_fractures is None:
            n_fractures = self._calculate_n_fractures_2d(params, area, intensi_fract)
        
        if n_fractures is None or n_fractures <= 0:
            n_fractures = 100
            st.markdown("")
            show_error("⚠️ Usando número padrão de 100 fraturas.")
            st.markdown("")
        
        # Armazenar calculado
        n_fractures_original = n_fractures
        
        if n_fractures > self.MAX_FRACTURES_2D:
            print(f"⚠️ Número de fraturas ({n_fractures}) excede limite ({self.MAX_FRACTURES_2D}). Limitando.")
            st.markdown("")
            show_error(f"⚠️ Número de fraturas limitado de {n_fractures} para {self.MAX_FRACTURES_2D}")
            st.markdown("")
            n_fractures = self.MAX_FRACTURES_2D
        
        self._last_generation_metadata = {
            'n_fractures_calculated': n_fractures_original,
            'n_fractures_generated': n_fractures,
            'n_fractures_limited': n_fractures_original > n_fractures,
            'scale_factor': n_fractures_original / max(n_fractures, 1)
        }
        
        if HAS_STREAMLIT:
            st.session_state.dfn_2d_metadata = self._last_generation_metadata
        
        print(f"Gerando {n_fractures} fraturas 2D")
        
        total_weight = sum(f.weight for f in families)
        if total_weight <= 0:
            total_weight = len(families)
            for f in families:
                f.weight = 1.0
        
        fractures_per_family = [
            max(1, int(n_fractures * (f.weight / total_weight)))
            for f in families
        ]
        
        for i, (fam, n_fam) in enumerate(zip(families, fractures_per_family)):
            print(f"  Família {i}: {n_fam} fraturas (peso: {fam.weight:.2%})")
        
        fractures = []
        
        for family_id, (family, n_fam) in enumerate(zip(families, fractures_per_family)):
            for _ in range(n_fam):
                length = self._sample_power_law(
                    params['exponent'],
                    params['x_min'],
                    params.get('x_max', width/2)
                )
                
                if 'g' in params and 'm' in params:
                    aperture = params['g'] * length ** params['m']
                else:
                    aperture = length * 0.001
                
                orientation = np.random.normal(family.orientation_mean, family.orientation_std)
                
                x1 = np.random.uniform(0, width)
                y1 = np.random.uniform(0, height)
                angle_rad = np.radians(orientation)
                x2 = x1 + length * np.cos(angle_rad)
                y2 = y1 + length * np.sin(angle_rad)
                
                fracture = Fracture2D(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    length=length, aperture=aperture,
                    orientation=orientation,
                    family=family_id
                )
                fractures.append(fracture)
        
        return fractures
    
    def _ensure_default_params(self, params: Dict) -> Dict:
        """Garante que parâmetros essenciais estejam definidos"""
        params = params.copy() if params else {}
        
        if 'exponent' not in params:
            params['exponent'] = self.DEFAULT_EXPONENT
        if 'x_min' not in params:
            params['x_min'] = self.DEFAULT_X_MIN
        if 'coefficient' not in params:
            params['coefficient'] = self.DEFAULT_COEFFICIENT
        
        return params
    
    def _create_default_families_2d(self, params: Dict) -> List[FractureFamily]:
        """Cria família default para DFN 2D"""
        orientation = params.get('orientation_mean', 45.0)
        orientation_std = params.get('orientation_std', 15.0)
        
        return [FractureFamily(
            orientation_mean=orientation,
            orientation_std=orientation_std,
            weight=1.0
        )]
    
    def _create_default_families_3d(self, params: Dict) -> List[FractureFamily]:
        """Cria família default para DFN 3D"""
        dip = params.get('dip_mean', 45.0)
        dip_std = params.get('dip_std', 10.0)
        dip_dir = params.get('dip_dir_mean', 90.0)
        dip_dir_std = params.get('dip_dir_std', 20.0)
        
        return [FractureFamily(
            orientation_mean=dip,
            orientation_std=dip_std,
            dip_dir_mean=dip_dir,
            dip_dir_std=dip_dir_std,
            weight=1.0
        )]
    
    def _calculate_n_fractures_2d(self, params: Dict, area_m2: float, intensi_fract) -> Optional[int]:
        """Calcula número de fraturas para DFN 2D"""
        if intensi_fract is None:
            return None
        
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        
        P20 = _get(intensi_fract, 'P20', None)
        
        if P20 is not None:
            n_fractures = int(P20 * area_m2)
            print(f"Calculando de P20={P20:.1f} fraturas/m², área={area_m2:.1f} m²")
            return max(1, n_fractures)
        
        return None
    
    def _calculate_n_fractures_3d(self, params: Dict, volume_m3: float, intensi_fract) -> Optional[int]:
        """
        Calcula número de fraturas para DFN 3D baseado em índices de intensidade
        
        NÃO APLICA LIMITE AQUI - limite é aplicado depois para preservar o valor calculado
        """
        if intensi_fract is None:
            return None
        
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)
        
        P30 = _get(intensi_fract, 'P30', None)
        P32 = _get(intensi_fract, 'P32', None)
        P20 = _get(intensi_fract, 'P20', None)
        mean_length_m = _get(intensi_fract, 'mean_length_m', None) or params.get('mean_length', 1.0)
        
        n_fractures = None
        
        if P30 is not None:
            # NÃO limitar P30 aqui - queremos o valor real calculado
            n_fractures = int(P30 * volume_m3)
            print(f"Calculando de P30={P30:.1f} fraturas/m³, volume={volume_m3:.1f} m³")
            print(f"  → n_fractures calculado: {n_fractures}")
            
        elif P32 is not None:
            mean_area_m2 = np.pi/4 * mean_length_m**2
            p30_est = P32 / mean_area_m2
            n_fractures = int(p30_est * volume_m3)
            print(f"Calculando de P32={P32:.3f} m²/m³, P30_est={p30_est:.1f}")
            
        elif P20 is not None:
            thickness_est = volume_m3 ** (1/3) / 10
            stereology_factor = 1.5
            p30_est = P20 * stereology_factor / max(thickness_est, 1.0)
            n_fractures = int(p30_est * volume_m3)
            print(f"Estimando de P20={P20:.1f} fraturas/m², P30_est={p30_est:.1f}")
        
        if n_fractures is not None:
            return max(1, n_fractures)
        return None
    
    def _sample_power_law(self, exponent: float, x_min: float, x_max: float) -> float:
        """Amostra de uma distribuição power-law truncada"""
        x_min = max(x_min, 1e-6)
        x_max = max(x_max, x_min * 2)
        
        if abs(exponent - 1.0) < 1e-6:
            return x_min * np.exp(np.random.random() * np.log(x_max / x_min))
        
        u = np.random.random()
        exp_factor = 1.0 - exponent
        
        x_min_exp = x_min ** exp_factor
        x_max_exp = x_max ** exp_factor
        
        value = (x_min_exp + u * (x_max_exp - x_min_exp)) ** (1.0 / exp_factor)
        
        return np.clip(value, x_min, x_max)



