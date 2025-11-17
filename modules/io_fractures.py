import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any
import warnings

class FractureDataLoader:
    """Carregador e validador de dados de fraturas"""
    
    def __init__(self):
        self.column_mapping = {
            'length': ['length', 'comprimento', 'Comprimento (mm)', 'l', 'L', 'trace_length'],
            'aperture': ['aperture', 'Abertura Média (mm)', 'Abertura Mínima (mm)', 'abertura', 'b', 'B', 'width', 'opening'],
            'orientation': ['orientation', 'Orientação (graus)', 'orientacao', 'azimuth', 'strike', 'direction'],
            'x': ['x', 'centroid_x', 'center_x', 'pos_x'],
            'y': ['y', 'centroid_y', 'center_y', 'pos_y']
        }
    
    def map_columns(self, df):
        """Mapeia as colunas de acordo com o mapeamento"""
        columns = df.columns
        column_rename = {}
        
        for standard, aliases in self.column_mapping.items():
            for alias in aliases:
                if alias in columns:
                    column_rename[alias] = standard
                    break
        
        return df.rename(columns=column_rename)

    def load_framfrat(self, file, area_m2: float, pixel_per_m: float) -> pd.DataFrame:
        """
        Carrega dados FRAMFRAT de planilha Excel
        
        Args:
            file: Arquivo Excel
            area_m2: Área real em metros quadrados
            pixel_per_m: Pixels por metro
        
        Returns:
            DataFrame com dados processados
        """
        try:
            # Ler Excel
            df = pd.read_excel(file)
            
            # Mapear as colunas do DataFrame
            df = self.map_columns(df)
            
            # Validar colunas essenciais
            required = ['length', 'aperture']
            
            if not all(col in df.columns for col in required):
                raise ValueError(f"Colunas obrigatórias ausentes: {required}. Colunas disponíveis: {df.columns.tolist()}")
            
            # CORREÇÃO: Converter de mm para m (FRAMFRAT sempre exporta em mm)
            # Comprimento: mm -> m
            df['length'] = df['length'] / 1000
            st.info("✅ Comprimentos convertidos de mm para m")
            
            # Abertura: mm -> m
            df['aperture'] = df['aperture'] / 1000
            st.info("✅ Aberturas convertidas de mm para m")
            
            # Remover valores inválidos
            df = df[(df['length'] > 0) & (df['aperture'] > 0)]
            
            # Adicionar metadados
            df.attrs['area'] = area_m2
            df.attrs['scale'] = pixel_per_m
            
            # Calcular propriedades derivadas
            df['aspect_ratio'] = df['aperture'] / df['length']
            
            # st.dataframe(df)
            # st.markdown("")
            # st.dataframe(df['aspect_ratio'])

            mask_invalid = df['aspect_ratio'] > 1
            invalid_df = (
            df.loc[
                mask_invalid,
                ["ID_Fratura", "aperture", "length"]
                + [c for c in df.columns if c not in ["ID_Fratura","aperture","length"]]
            ].sort_values("ID_Fratura", ascending=False))

            cols_usadas = ["ID_Fratura", "length", "aperture", "aspect_ratio"]
            display_df = pd.DataFrame({
                "ID da fratura": invalid_df["ID_Fratura"],
                "Comprimento (m)": invalid_df["length"].astype(float).round(4),
                "Abertura (mm)": (invalid_df["aperture"].astype(float) * 1000).round(2),
            })

            # Exibir no Streamlit
            if len(display_df) > 0:
                st.warning(f"⚠️ {len(display_df)} fraturas com abertura > comprimento detectadas!")
                st.dataframe(display_df, width='stretch', hide_index=True) 

            
            # Estatísticas para debug
            st.write(f"Comprimento: min={df['length'].min():.4f}m, max={df['length'].max():.4f}m")
            st.write(f"Abertura: min={df['aperture'].min()*1000:.2f}mm, max={df['aperture'].max()*1000:.2f}mm")
            
            return df
            
        except Exception as e:
            raise Exception(f"Erro ao carregar FRAMFRAT: {str(e)}")
    
    def load_scanline(self, file, length_m: float) -> pd.DataFrame:
        """
        Carrega dados de scanline
        
        Args:
            file: Arquivo texto/CSV
            length_m: Comprimento da scanline em metros
        
        Returns:
            DataFrame com dados processados
        """
        try:
            # Detectar formato
            content = file.read().decode('utf-8')
            lines = content.strip().split('\n')
            
            # Tentar diferentes separadores
            separators = ['\t', ',', ' ', ';']
            data = []
            
            for line in lines:
                if line.strip():
                    # Tentar parsear linha
                    values = None
                    for sep in separators:
                        parts = line.strip().split(sep)
                        if len(parts) >= 2:
                            try:
                                values = [float(p) for p in parts[:2]]
                                break
                            except:
                                continue
                    
                    if values:
                        data.append(values)
            
            if not data:
                raise ValueError("Não foi possível parsear os dados")
            
            # Criar DataFrame
            df = pd.DataFrame(data, columns=['position', 'aperture'])
            
            # Calcular comprimentos entre fraturas
            df = df.sort_values('position')
            df['length'] = df['position'].diff().fillna(df['position'].iloc[0])
            
            # Determinar unidade baseado na magnitude
            # Se max position > 100, provavelmente está em mm
            if df['position'].max() > length_m * 10:
                df['position'] = df['position'] / 1000
                df['length'] = df['length'] / 1000
                st.info("✅ Posições convertidas de mm para m")
            
            # Se max aperture > 0.1, provavelmente está em mm
            if df['aperture'].max() > 0.1:
                df['aperture'] = df['aperture'] / 1000
                st.info("✅ Aberturas convertidas de mm para m")
            
            # Adicionar metadados
            df.attrs['scanline_length'] = length_m
            
            return df[df['aperture'] > 0]
            
        except Exception as e:
            raise Exception(f"Erro ao carregar scanline: {str(e)}")




# import pandas as pd
# import numpy as np
# import streamlit as st
# from typing import Optional, Dict, Any
# import warnings

# class FractureDataLoader:
#     """Carregador e validador de dados de fraturas"""
    
#     def __init__(self):
#         self.column_mapping = {
#             'length': ['length', 'comprimento', 'Comprimento (mm)', 'l', 'L', 'trace_length'],
#             'aperture': ['aperture', 'Abertura Média (mm)', 'abertura', 'b', 'B', 'width', 'opening'],
#             'orientation': ['orientation', 'Orientação (graus)', 'orientacao', 'azimuth', 'strike', 'direction'],
#             'x': ['x', 'centroid_x', 'center_x', 'pos_x'],
#             'y': ['y', 'centroid_y', 'center_y', 'pos_y']
#         }
    

#     # Função para mapear as colunas de acordo com o mapeamento
#     def map_columns(self, df):
#         columns = df.columns
#         column_rename = {}
#         for standard, aliases in self.column_mapping.items():
#             for alias in aliases:
#                 if alias in columns:
#                     column_rename[alias] = standard
#                     break
#         return df.rename(columns=column_rename)

#     def load_framfrat(self, file, area_m2: float, pixel_per_m: float) -> pd.DataFrame:
#         """
#         Carrega dados FRAMFRAT de planilha Excel
        
#         Args:
#             file: Arquivo Excel
#             area_m2: Área real em metros quadrados
#             pixel_per_m: Pixels por metro
        
#         Returns:
#             DataFrame com dados processados
#         """
#         try:
#             # Ler Excel
#             df = pd.read_excel(file)
            
#             # Mapear as colunas do DataFrame
#             df = self.map_columns(df)
            
#             # Validar colunas essenciais
#             required = ['length', 'aperture']
            
#             if not all(col in df.columns for col in required):
#                 raise ValueError(f"Colunas obrigatórias ausentes: {required}")
            
#             # Converter unidades se necessário (assumir mm -> m)
#             if df['length'].max() > 100:  # Provavelmente em mm
#                 df['length'] = df['length'] / 1000
#                 warnings.warn("Comprimentos convertidos de mm para m")
            
#             if df['aperture'].max() > 1:  # Provavelmente em mm
#                 df['aperture'] = df['aperture'] / 1000
#                 warnings.warn("Aberturas convertidas de mm para m")
            
#             # Remover valores inválidos
#             df = df[(df['length'] > 0) & (df['aperture'] > 0)]
            
#             # Adicionar metadados
#             df.attrs['area'] = area_m2
#             df.attrs['scale'] = pixel_per_m
            
#             # Calcular propriedades derivadas
#             df['aspect_ratio'] = df['aperture'] / df['length']
            
#             # Validar proporções
#             if (df['aspect_ratio'] > 1).any():
#                 warnings.warn("Fraturas com abertura > comprimento detectadas!")
            
#             return df
            
#         except Exception as e:
#             raise Exception(f"Erro ao carregar FRAMFRAT: {str(e)}")
    
#     def load_scanline(self, file, length_m: float) -> pd.DataFrame:
#         """
#         Carrega dados de scanline
        
#         Args:
#             file: Arquivo texto/CSV
#             length_m: Comprimento da scanline em metros
        
#         Returns:
#             DataFrame com dados processados
#         """
#         try:
#             # Detectar formato
#             content = file.read().decode('utf-8')
#             lines = content.strip().split('\n')
            
#             # Tentar diferentes separadores
#             separators = ['\t', ',', ' ', ';']
#             data = []
            
#             for line in lines:
#                 if line.strip():
#                     # Tentar parsear linha
#                     values = None
#                     for sep in separators:
#                         parts = line.strip().split(sep)
#                         if len(parts) >= 2:
#                             try:
#                                 values = [float(p) for p in parts[:2]]
#                                 break
#                             except:
#                                 continue
                    
#                     if values:
#                         data.append(values)
            
#             if not data:
#                 raise ValueError("Não foi possível parsear os dados")
            
#             # Criar DataFrame
#             df = pd.DataFrame(data, columns=['position', 'aperture'])
            
#             # Calcular comprimentos entre fraturas
#             df = df.sort_values('position')
#             df['length'] = df['position'].diff().fillna(df['position'].iloc[0])
            
#             # Converter unidades se necessário
#             if df['aperture'].max() > 1:  # Provavelmente em mm
#                 df['aperture'] = df['aperture'] / 1000
#                 warnings.warn("Aberturas convertidas de mm para m")
            
#             if df['position'].max() > length_m * 10:  # Provavelmente em mm
#                 df['position'] = df['position'] / 1000
#                 df['length'] = df['length'] / 1000
#                 warnings.warn("Posições convertidas de mm para m")
            
#             # Adicionar metadados
#             df.attrs['scanline_length'] = length_m
            
#             return df[df['aperture'] > 0]
            
#         except Exception as e:
#             raise Exception(f"Erro ao carregar scanline: {str(e)}")
    
#     def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Mapeia nomes de colunas variados para padrão"""
#         df_lower = df.rename(columns=str.lower)
        
#         for standard, variations in self.column_mapping.items():
#             for var in variations:
#                 if var in df_lower.columns:
#                     df = df.rename(columns={
#                         [c for c in df.columns if c.lower() == var][0]: standard
#                     })
#                     break
        
#         return df









# import pandas as pd
# import numpy as np
# import streamlit as st
# from typing import Optional, Dict, Any
# import warnings

# class FractureDataLoader:
#     """Carregador e validador de dados de fraturas"""
    
#     def __init__(self):
#         self.column_mapping = {
#             'length': ['length', 'comprimento', 'Comprimento (mm)', 'l', 'L', 'trace_length'],
#             'aperture': ['aperture', 'Abertura Média (mm)', 'Abertura Mínima (mm)', 'abertura', 'b', 'B', 'width', 'opening'],
#             'orientation': ['orientation', 'Orientação (graus)', 'orientacao', 'azimuth', 'strike', 'direction'],
#             'x': ['x', 'centroid_x', 'center_x', 'pos_x'],
#             'y': ['y', 'centroid_y', 'center_y', 'pos_y']
#         }
    
#     def map_columns(self, df):
#         """Mapeia as colunas de acordo com o mapeamento"""
#         columns = df.columns
#         column_rename = {}
        
#         for standard, aliases in self.column_mapping.items():
#             for alias in aliases:
#                 if alias in columns:
#                     column_rename[alias] = standard
#                     break
        
#         return df.rename(columns=column_rename)

#     def load_framfrat(self, file, area_m2: float, pixel_per_m: float) -> pd.DataFrame:
#         """
#         Carrega dados FRAMFRAT de planilha Excel
        
#         Args:
#             file: Arquivo Excel
#             area_m2: Área real em metros quadrados
#             pixel_per_m: Pixels por metro
        
#         Returns:
#             DataFrame com dados processados
#         """
#         try:
#             # Ler Excel
#             df = pd.read_excel(file)
            
#             # Debug: mostrar colunas originais
#             #st.write(f"Colunas originais: {df.columns.tolist()}")
            
#             # Mapear as colunas do DataFrame
#             df = self.map_columns(df)
            
#             # Validar colunas essenciais
#             required = ['length', 'aperture']
            
#             if not all(col in df.columns for col in required):
#                 raise ValueError(f"Colunas obrigatórias ausentes: {required}. Colunas disponíveis: {df.columns.tolist()}")
            
#             # CORREÇÃO: Converter de mm para m (FRAMFRAT sempre exporta em mm)
#             # Comprimento: mm -> m
#             df['length'] = df['length'] / 1000
#             st.info("✅ Comprimentos convertidos de mm para m")
            
#             # Abertura: mm -> m
#             df['aperture'] = df['aperture'] / 1000
#             st.info("✅ Aberturas convertidas de mm para m")
            
#             # Remover valores inválidos
#             df = df[(df['length'] > 0) & (df['aperture'] > 0)]
            
#             # Adicionar metadados
#             df.attrs['area'] = area_m2
#             df.attrs['scale'] = pixel_per_m
            
#             # Calcular propriedades derivadas
#             df['aspect_ratio'] = df['aperture'] / df['length']
            
#             # st.dataframe(df)
#             # st.markdown("")
#             # st.dataframe(df['aspect_ratio'])

#             mask_invalid = df['aspect_ratio'] > 1
#             invalid_df = (
#             df.loc[
#                 mask_invalid,
#                 ["ID_Fratura", "aperture", "length"]
#                 + [c for c in df.columns if c not in ["ID_Fratura","aperture","length"]]
#             ].sort_values("ID_Fratura", ascending=False))

#             cols_usadas = ["ID_Fratura", "length", "aperture", "aspect_ratio"]
#             display_df = pd.DataFrame({
#                 "ID da fratura": invalid_df["ID_Fratura"],
#                 "Comprimento (m)": invalid_df["length"].astype(float).round(4),
#                 "Abertura (mm)": (invalid_df["aperture"].astype(float) * 1000).round(2),
#             })

#             # Exibir no Streamlit
#             if len(display_df) > 0:
#                 st.warning(f"⚠️ {len(display_df)} fraturas com abertura > comprimento detectadas!")
#                 st.dataframe(display_df, width='stretch', hide_index=True) 

            
#             # Estatísticas para debug
#             st.write(f"Comprimento: min={df['length'].min():.4f}m, max={df['length'].max():.4f}m")
#             st.write(f"Abertura: min={df['aperture'].min()*1000:.2f}mm, max={df['aperture'].max()*1000:.2f}mm")
            
#             return df
            
#         except Exception as e:
#             raise Exception(f"Erro ao carregar FRAMFRAT: {str(e)}")
    
#     def load_scanline(self, file, length_m: float) -> pd.DataFrame:
#         """
#         Carrega dados de scanline
        
#         Args:
#             file: Arquivo texto/CSV
#             length_m: Comprimento da scanline em metros
        
#         Returns:
#             DataFrame com dados processados
#         """
#         try:
#             # Detectar formato
#             content = file.read().decode('utf-8')
#             lines = content.strip().split('\n')
            
#             # Tentar diferentes separadores
#             separators = ['\t', ',', ' ', ';']
#             data = []
            
#             for line in lines:
#                 if line.strip():
#                     # Tentar parsear linha
#                     values = None
#                     for sep in separators:
#                         parts = line.strip().split(sep)
#                         if len(parts) >= 2:
#                             try:
#                                 values = [float(p) for p in parts[:2]]
#                                 break
#                             except:
#                                 continue
                    
#                     if values:
#                         data.append(values)
            
#             if not data:
#                 raise ValueError("Não foi possível parsear os dados")
            
#             # Criar DataFrame
#             df = pd.DataFrame(data, columns=['position', 'aperture'])
            
#             # Calcular comprimentos entre fraturas
#             df = df.sort_values('position')
#             df['length'] = df['position'].diff().fillna(df['position'].iloc[0])
            
#             # Determinar unidade baseado na magnitude
#             # Se max position > 100, provavelmente está em mm
#             if df['position'].max() > length_m * 10:
#                 df['position'] = df['position'] / 1000
#                 df['length'] = df['length'] / 1000
#                 st.info("✅ Posições convertidas de mm para m")
            
#             # Se max aperture > 0.1, provavelmente está em mm
#             if df['aperture'].max() > 0.1:
#                 df['aperture'] = df['aperture'] / 1000
#                 st.info("✅ Aberturas convertidas de mm para m")
            
#             # Adicionar metadados
#             df.attrs['scanline_length'] = length_m
            
#             return df[df['aperture'] > 0]
            
#         except Exception as e:
#             raise Exception(f"Erro ao carregar scanline: {str(e)}")




# # import pandas as pd
# # import numpy as np
# # import streamlit as st
# # from typing import Optional, Dict, Any
# # import warnings

# # class FractureDataLoader:
# #     """Carregador e validador de dados de fraturas"""
    
# #     def __init__(self):
# #         self.column_mapping = {
# #             'length': ['length', 'comprimento', 'Comprimento (mm)', 'l', 'L', 'trace_length'],
# #             'aperture': ['aperture', 'Abertura Média (mm)', 'abertura', 'b', 'B', 'width', 'opening'],
# #             'orientation': ['orientation', 'Orientação (graus)', 'orientacao', 'azimuth', 'strike', 'direction'],
# #             'x': ['x', 'centroid_x', 'center_x', 'pos_x'],
# #             'y': ['y', 'centroid_y', 'center_y', 'pos_y']
# #         }
    

# #     # Função para mapear as colunas de acordo com o mapeamento
# #     def map_columns(self, df):
# #         columns = df.columns
# #         column_rename = {}
# #         for standard, aliases in self.column_mapping.items():
# #             for alias in aliases:
# #                 if alias in columns:
# #                     column_rename[alias] = standard
# #                     break
# #         return df.rename(columns=column_rename)

# #     def load_framfrat(self, file, area_m2: float, pixel_per_m: float) -> pd.DataFrame:
# #         """
# #         Carrega dados FRAMFRAT de planilha Excel
        
# #         Args:
# #             file: Arquivo Excel
# #             area_m2: Área real em metros quadrados
# #             pixel_per_m: Pixels por metro
        
# #         Returns:
# #             DataFrame com dados processados
# #         """
# #         try:
# #             # Ler Excel
# #             df = pd.read_excel(file)
            
# #             # Mapear as colunas do DataFrame
# #             df = self.map_columns(df)
            
# #             # Validar colunas essenciais
# #             required = ['length', 'aperture']
            
# #             if not all(col in df.columns for col in required):
# #                 raise ValueError(f"Colunas obrigatórias ausentes: {required}")
            
# #             # Converter unidades se necessário (assumir mm -> m)
# #             if df['length'].max() > 100:  # Provavelmente em mm
# #                 df['length'] = df['length'] / 1000
# #                 warnings.warn("Comprimentos convertidos de mm para m")
            
# #             if df['aperture'].max() > 1:  # Provavelmente em mm
# #                 df['aperture'] = df['aperture'] / 1000
# #                 warnings.warn("Aberturas convertidas de mm para m")
            
# #             # Remover valores inválidos
# #             df = df[(df['length'] > 0) & (df['aperture'] > 0)]
            
# #             # Adicionar metadados
# #             df.attrs['area'] = area_m2
# #             df.attrs['scale'] = pixel_per_m
            
# #             # Calcular propriedades derivadas
# #             df['aspect_ratio'] = df['aperture'] / df['length']
            
# #             # Validar proporções
# #             if (df['aspect_ratio'] > 1).any():
# #                 warnings.warn("Fraturas com abertura > comprimento detectadas!")
            
# #             return df
            
# #         except Exception as e:
# #             raise Exception(f"Erro ao carregar FRAMFRAT: {str(e)}")
    
# #     def load_scanline(self, file, length_m: float) -> pd.DataFrame:
# #         """
# #         Carrega dados de scanline
        
# #         Args:
# #             file: Arquivo texto/CSV
# #             length_m: Comprimento da scanline em metros
        
# #         Returns:
# #             DataFrame com dados processados
# #         """
# #         try:
# #             # Detectar formato
# #             content = file.read().decode('utf-8')
# #             lines = content.strip().split('\n')
            
# #             # Tentar diferentes separadores
# #             separators = ['\t', ',', ' ', ';']
# #             data = []
            
# #             for line in lines:
# #                 if line.strip():
# #                     # Tentar parsear linha
# #                     values = None
# #                     for sep in separators:
# #                         parts = line.strip().split(sep)
# #                         if len(parts) >= 2:
# #                             try:
# #                                 values = [float(p) for p in parts[:2]]
# #                                 break
# #                             except:
# #                                 continue
                    
# #                     if values:
# #                         data.append(values)
            
# #             if not data:
# #                 raise ValueError("Não foi possível parsear os dados")
            
# #             # Criar DataFrame
# #             df = pd.DataFrame(data, columns=['position', 'aperture'])
            
# #             # Calcular comprimentos entre fraturas
# #             df = df.sort_values('position')
# #             df['length'] = df['position'].diff().fillna(df['position'].iloc[0])
            
# #             # Converter unidades se necessário
# #             if df['aperture'].max() > 1:  # Provavelmente em mm
# #                 df['aperture'] = df['aperture'] / 1000
# #                 warnings.warn("Aberturas convertidas de mm para m")
            
# #             if df['position'].max() > length_m * 10:  # Provavelmente em mm
# #                 df['position'] = df['position'] / 1000
# #                 df['length'] = df['length'] / 1000
# #                 warnings.warn("Posições convertidas de mm para m")
            
# #             # Adicionar metadados
# #             df.attrs['scanline_length'] = length_m
            
# #             return df[df['aperture'] > 0]
            
# #         except Exception as e:
# #             raise Exception(f"Erro ao carregar scanline: {str(e)}")
    
# #     def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
# #         """Mapeia nomes de colunas variados para padrão"""
# #         df_lower = df.rename(columns=str.lower)
        
# #         for standard, variations in self.column_mapping.items():
# #             for var in variations:
# #                 if var in df_lower.columns:
# #                     df = df.rename(columns={
# #                         [c for c in df.columns if c.lower() == var][0]: standard
# #                     })
# #                     break
        
# #         return df