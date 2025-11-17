import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class IntensitySpacingAnalyzer:
    """Analisador de intensidade e espaçamento size-cognizant (Ortega et al. 2006)"""
    
    def calculate_p10(self, data: pd.DataFrame, threshold: float, 
                     area: float) -> float:
        """
        Calcula intensidade P10 (fraturas/m) para um limiar de tamanho
        
        Args:
            data: DataFrame com fraturas
            threshold: Limiar de comprimento
            area: Área de amostragem (m²)
        
        Returns:
            P10 em fraturas/m
        """
        # Filtrar por tamanho
        filtered = data[data['length'] >= threshold]
        n_fractures = len(filtered)
        
        # Comprimento de amostragem equivalente
        # Para área 2D, usar perímetro aproximado
        sample_length = 4 * np.sqrt(area)
        
        # P10
        p10 = n_fractures / sample_length
        
        return p10
    
    def calculate_p10_scanline(self, data: pd.DataFrame, threshold: float,
                               scanline_length: float) -> float:
        """
        Calcula P10 diretamente de dados de scanline
        
        Args:
            data: DataFrame com dados de scanline
            threshold: Limiar de abertura/tamanho
            scanline_length: Comprimento da scanline
        
        Returns:
            P10 em fraturas/m
        """
        # Filtrar por tamanho
        if 'length' in data.columns:
            filtered = data[data['length'] >= threshold]
        else:
            # Usar abertura como proxy
            filtered = data[data['aperture'] >= threshold/100]  # Relação empírica
        
        n_fractures = len(filtered)
        
        # P10
        p10 = n_fractures / scanline_length
        
        return p10
    
    def calculate_p21(self, data: pd.DataFrame, area: float) -> float:
        """
        Calcula intensidade P21 (m/m²) - comprimento total por área
        
        Args:
            data: DataFrame com fraturas
            area: Área de amostragem
        
        Returns:
            P21 em m/m²
        """
        total_length = data['length'].sum()
        p21 = total_length / area
        
        return p21
    
    def calculate_p32(self, data: pd.DataFrame, volume: float) -> float:
        """
        Calcula intensidade P32 (m²/m³) - área total por volume
        
        Args:
            data: DataFrame com fraturas 3D
            volume: Volume de amostragem
        
        Returns:
            P32 em m²/m³
        """
        if 'area' not in data.columns:
            # Estimar área como comprimento² * π/4 (aproximação circular)
            data['area'] = data['length']**2 * np.pi / 4
        
        total_area = data['area'].sum()
        p32 = total_area / volume
        
        return p32
    
    def calculate_average_spacing(self, p10: float) -> float:
        """
        Calcula espaçamento médio a partir de P10
        
        Args:
            p10: Intensidade de fraturas (fraturas/m)
        
        Returns:
            Espaçamento médio em metros
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
        
        Args:
            data1: Dados FRAMFRAT
            data2: Dados scanline
            threshold: Limiar comum
            area1: Área FRAMFRAT
            length2: Comprimento scanline
        
        Returns:
            Dicionário com comparações
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
        
        Args:
            data: DataFrame com fraturas
            target_cv: Coeficiente de variação alvo
        
        Returns:
            Volume estimado em m³
        """
        # Baseado na variabilidade dos comprimentos
        lengths = data['length'].values
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        cv = std_length / mean_length
        
        # Estimar tamanho necessário
        n_required = (cv / target_cv)**2
        current_n = len(lengths)
        
        # Escalar volume
        scale_factor = n_required / current_n
        
        # Volume aproximado (assumindo densidade constante)
        rev = scale_factor * mean_length**3
        
        return rev







# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple

# class IntensitySpacingAnalyzer:
#     """Analisador de intensidade e espaçamento size-cognizant (Ortega et al. 2006)"""
    
#     def calculate_p10(self, data: pd.DataFrame, threshold: float, 
#                      area: float) -> float:
#         """
#         Calcula intensidade P10 (fraturas/m) para um limiar de tamanho
        
#         Args:
#             data: DataFrame com fraturas
#             threshold: Limiar de comprimento
#             area: Área de amostragem (m²)
        
#         Returns:
#             P10 em fraturas/m
#         """
#         # Filtrar por tamanho
#         filtered = data[data['length'] >= threshold]
#         n_fractures = len(filtered)
        
#         # Comprimento de amostragem equivalente
#         # Para área 2D, usar perímetro aproximado
#         sample_length = 4 * np.sqrt(area)
        
#         # P10
#         p10 = n_fractures / sample_length
        
#         return p10
    
#     def calculate_p10_scanline(self, data: pd.DataFrame, threshold: float,
#                                scanline_length: float) -> float:
#         """
#         Calcula P10 diretamente de dados de scanline
        
#         Args:
#             data: DataFrame com dados de scanline
#             threshold: Limiar de abertura/tamanho
#             scanline_length: Comprimento da scanline
        
#         Returns:
#             P10 em fraturas/m
#         """
#         # Filtrar por tamanho
#         if 'length' in data.columns:
#             filtered = data[data['length'] >= threshold]
#         else:
#             # Usar abertura como proxy
#             filtered = data[data['aperture'] >= threshold/100]  # Relação empírica
        
#         n_fractures = len(filtered)
        
#         # P10
#         p10 = n_fractures / scanline_length
        
#         return p10
    
#     def calculate_p21(self, data: pd.DataFrame, area: float) -> float:
#         """
#         Calcula intensidade P21 (m/m²) - comprimento total por área
        
#         Args:
#             data: DataFrame com fraturas
#             area: Área de amostragem
        
#         Returns:
#             P21 em m/m²
#         """
#         total_length = data['length'].sum()
#         p21 = total_length / area
        
#         return p21
    
#     def calculate_p32(self, data: pd.DataFrame, volume: float) -> float:
#         """
#         Calcula intensidade P32 (m²/m³) - área total por volume
        
#         Args:
#             data: DataFrame com fraturas 3D
#             volume: Volume de amostragem
        
#         Returns:
#             P32 em m²/m³
#         """
#         if 'area' not in data.columns:
#             # Estimar área como comprimento² * π/4 (aproximação circular)
#             data['area'] = data['length']**2 * np.pi / 4
        
#         total_area = data['area'].sum()
#         p32 = total_area / volume
        
#         return p32
    
#     def calculate_average_spacing(self, p10: float) -> float:
#         """
#         Calcula espaçamento médio a partir de P10
        
#         Args:
#             p10: Intensidade de fraturas (fraturas/m)
        
#         Returns:
#             Espaçamento médio em metros
#         """
#         if p10 > 0:
#             return 1.0 / p10
#         else:
#             return np.inf
    
#     def normalized_comparison(self, data1: pd.DataFrame, data2: pd.DataFrame,
#                             threshold: float, area1: float, 
#                             length2: float) -> Dict:
#         """
#         Compara intensidades normalizadas entre duas fontes
        
#         Args:
#             data1: Dados FRAMFRAT
#             data2: Dados scanline
#             threshold: Limiar comum
#             area1: Área FRAMFRAT
#             length2: Comprimento scanline
        
#         Returns:
#             Dicionário com comparações
#         """
#         p10_1 = self.calculate_p10(data1, threshold, area1)
#         p10_2 = self.calculate_p10_scanline(data2, threshold, length2)
        
#         spacing_1 = self.calculate_average_spacing(p10_1)
#         spacing_2 = self.calculate_average_spacing(p10_2)
        
#         return {
#             'threshold': threshold,
#             'p10_framfrat': p10_1,
#             'p10_scanline': p10_2,
#             'spacing_framfrat': spacing_1,
#             'spacing_scanline': spacing_2,
#             'ratio_p10': p10_1 / p10_2 if p10_2 > 0 else np.inf,
#             'ratio_spacing': spacing_1 / spacing_2 if spacing_2 > 0 else np.inf
#         }
    
#     def estimate_representative_volume(self, data: pd.DataFrame, 
#                                       target_cv: float = 0.1) -> float:
#         """
#         Estima volume representativo elementar (REV)
        
#         Args:
#             data: DataFrame com fraturas
#             target_cv: Coeficiente de variação alvo
        
#         Returns:
#             Volume estimado em m³
#         """
#         # Baseado na variabilidade dos comprimentos
#         lengths = data['length'].values
#         mean_length = np.mean(lengths)
#         std_length = np.std(lengths)
#         cv = std_length / mean_length
        
#         # Estimar tamanho necessário
#         n_required = (cv / target_cv)**2
#         current_n = len(lengths)
        
#         # Escalar volume
#         scale_factor = n_required / current_n
        
#         # Volume aproximado (assumindo densidade constante)
#         rev = scale_factor * mean_length**3
        
#         return rev