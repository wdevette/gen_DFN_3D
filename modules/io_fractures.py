"""
io_fractures.py - VERSÃO CORRIGIDA v2

CORREÇÕES:
1. Detecta unidade pelo NOME da coluna (mm vs m)
2. Abertura é OPCIONAL (não obrigatória)
3. Orientação é OPCIONAL
4. Avisa o usuário quais análises estão disponíveis
5. Suporta Tabela 7 (sem orientação) e Tabela 8 (sem abertura)
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, Any, Tuple
import warnings

# Importar funções de alerta
try:
    from func_tools import show_error, show_success, show_info
except ImportError:
    # Fallback se func_tools não estiver disponível
    def show_error(msg, bg=None, border=None):
        st.error(msg.replace('<strong>', '**').replace('</strong>', '**'))
    def show_success(msg, bg=None, border=None):
        st.success(msg.replace('<strong>', '**').replace('</strong>', '**'))
    def show_info(msg, bg=None, border=None):
        st.info(msg.replace('<strong>', '**').replace('</strong>', '**'))


class FractureDataLoader:
    """
    Carregador e validador de dados de fraturas
    
    PADRONIZAÇÃO: TODAS AS UNIDADES EM METROS
    - Comprimento: metros (m) - OBRIGATÓRIO
    - Abertura: metros (m) - OPCIONAL
    - Orientação: graus (°) - OPCIONAL
    - Posição: metros (m) - OPCIONAL
    - Área: metros quadrados (m²) - OPCIONAL
    
    FLEXIBILIDADE:
    - Aceita dados com apenas comprimento (análise power-law básica)
    - Aceita dados com comprimento + abertura (D-L scaling possível)
    - Aceita dados com comprimento + orientação (clustering possível)
    """
    
    def __init__(self):
        self.column_mapping = {
            'length': ['length', 'comprimento', 'Comprimento (m)', 'Comprimento (mm)', 'l', 'L', 'trace_length'],
            'aperture': ['aperture', 'Abertura Média (mm)', 'Abertura Mínima (mm)', 'Abertura Máxima (mm)', 'Abertura (mm)', 'abertura', 'b', 'B', 'width', 'opening'],
            'orientation': ['orientation', 'Orientação (graus)', 'Orientação (°)', 'orientacao', 'azimuth', 'strike', 'direction'],
            'position': ['position', 'Posição (m)', 'posicao', 'pos'],
            'ID_Fratura': ['ID_Fratura', 'id_fratura', 'id', 'ID', 'fratura_id'],
            'ID_Segmento': ['ID_Segmento','ID_Set', 'id_set', 'set', 'family', 'familia'],
            'x': ['x', 'centroid_x', 'center_x', 'pos_x'],
            'y': ['y', 'centroid_y', 'center_y', 'pos_y'],
            'area_Fract' : ['Área da Fratura (mm²)', 'area fratura']
        }
        
        # Mapeamento de colunas que indicam unidade mm no nome
        self.mm_indicators = ['(mm)', 'mm', '_mm']
        
        # Armazenar quais colunas estão disponíveis
        self.available_columns = {
            'length': False,
            'aperture': False,
            'orientation': False
        }
    
    def _detect_unit_from_column_name(self, original_columns: list, target: str) -> str:
        """
        Detecta a unidade original da coluna pelo nome
        
        Returns:
            'mm' se a coluna indica milímetros, 'm' caso contrário
        """
        aliases = self.column_mapping.get(target, [])
        
        for col in original_columns:
            if col in aliases or col.lower() in [a.lower() for a in aliases]:
                # Verificar se o nome indica mm
                for indicator in self.mm_indicators:
                    if indicator.lower() in col.lower():
                        return 'mm'
        return 'm'
    
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



    def load_framfrat(self, file, area_m2: float, pixel_per_mm: float) -> Tuple[pd.DataFrame, Dict[str, bool]]:
        """
        Carrega dados FRAMFRAT de planilha Excel e converte para METROS
        
        FLEXÍVEL: Aceita dados com ou sem abertura/orientação
        - Comprimento: OBRIGATÓRIO
        - Abertura: OPCIONAL (necessária para D-L scaling)
        - Orientação: OPCIONAL (necessária para clustering)
        
        Args:
            file: Arquivo Excel
            area_m2: Área real em METROS QUADRADOS
            pixel_per_mm: Pixels por milímetro (para referência de escala)
        
        Returns:
            Tuple: (DataFrame com dados processados em METROS, Dict com colunas disponíveis)
        """
        try:
            # Ler Excel
            df = pd.read_excel(file)
            
            # Mostrar colunas originais para debug
            st.info(f"📋 Colunas detectadas: {df.columns.tolist()}")
            
            # Mapear as colunas do DataFrame
            df = self.map_columns(df)
            
            # Verificar quais colunas estão disponíveis
            available = {
                'length': 'length' in df.columns,
                'aperture': 'aperture' in df.columns,
                'orientation': 'orientation' in df.columns
            }
            self.available_columns = available
            
            # APENAS comprimento é obrigatório
            if not available['length']:
                raise ValueError(f"Coluna 'length' (comprimento) é OBRIGATÓRIA mas não foi encontrada. Colunas disponíveis: {df.columns.tolist()}")
            
            # Avisar sobre colunas opcionais ausentes
            if not available['aperture']:
                st.warning("⚠️ Coluna **abertura** não encontrada. D-L Scaling Validation **não será possível**.")
            if not available['orientation']:
                st.warning("⚠️ Coluna **orientação** não encontrada. Clustering de famílias **não será possível**.")
            
            # Converter comprimento para numérico
            df['length'] = pd.to_numeric(df['length'], errors='coerce')
            
            # CONVERSÃO DE COMPRIMENTO: Detectar unidade e converter para METROS
            if df['length'].max() > 100:  # Comprimentos > 100 provavelmente em mm
                st.info("🔄 Comprimento detectado em **mm** → Convertendo para metros")
                df['length'] = df['length'] / 1000  # mm → m
            elif df['length'].max() > 10:  # Entre 10-100, pode ser cm ou mm
                st.warning("⚠️ Unidade de comprimento ambígua. Assumindo **mm** → convertendo para metros")
                df['length'] = df['length'] / 1000
            else:  # Valores < 10, provavelmente já em metros
                st.info("✓ Comprimento detectado já em **metros**")
            
            # CONVERSÃO DE ABERTURA (se disponível)
            if available['aperture']:
                df['aperture'] = pd.to_numeric(df['aperture'], errors='coerce')
                
                # Detectar unidade de abertura
                if df['aperture'].max() > 0.1:  # > 100mm seria muito grande para abertura
                    st.info("🔄 Abertura detectada em **mm** → Convertendo para metros")
                    df['aperture'] = df['aperture'] / 1000  # mm → m
                else:
                    st.info("✓ Abertura detectada já em **metros**")
            
            # CONVERSÃO DE ORIENTAÇÃO (se disponível)
            if available['orientation']:
                df['orientation'] = pd.to_numeric(df['orientation'], errors='coerce')
                df['orientation'] = df['orientation'] % 360  # Normalizar para [0, 360)
            
            # Número de fraturas identificadas
            show_success(f"<strong>{len(df)}</strong> fraturas foram identificadas.")
            st.markdown('')

            # Remover valores inválidos de comprimento (obrigatório)
            df = df[df['length'] > 0]
            
            # Remover valores inválidos de abertura (se existir)
            if available['aperture']:
                df = df[df['aperture'] > 0]
            
            # Adicionar metadados
            df.attrs['area'] = area_m2
            df.attrs['scale'] = pixel_per_mm
            df.attrs['unit'] = 'meters'
            df.attrs['has_aperture'] = available['aperture']
            df.attrs['has_orientation'] = available['orientation']
            
            # Calcular aspect ratio APENAS se abertura disponível
            if available['aperture']:
                df['aspect_ratio'] = df['aperture'] / df['length']
                
                # Verificar fraturas com abertura > comprimento
                mask_invalid = df['aspect_ratio'] > 1
                
                if 'ID_Fratura' in df.columns:
                    invalid_df = df.loc[mask_invalid, ["ID_Fratura", "aperture", "length"]].sort_values("ID_Fratura", ascending=False)
                else:
                    invalid_df = df.loc[mask_invalid, ["aperture", "length"]].copy()
                    invalid_df['ID_Fratura'] = invalid_df.index

                if len(invalid_df) > 0:
                    display_df = pd.DataFrame({
                        "ID da fratura": invalid_df["ID_Fratura"] if "ID_Fratura" in invalid_df.columns else invalid_df.index,
                        "Comprimento (m)": invalid_df["length"].astype(float).round(4),
                        "Abertura (mm)": (invalid_df["aperture"] * 1000).astype(float).round(2),
                    })
                    show_error(f"⚠️ <strong>{len(display_df)}</strong> fraturas com <strong>abertura maior que comprimento</strong> detectadas!")
                    st.markdown('')
                    with st.expander('Mais detalhes'):
                        st.dataframe(display_df, use_container_width=True, hide_index=True) 
                    
                    eliminar_fraturas = st.checkbox(
                        'Excluir fraturas com aberturas maiores que os comprimentos', 
                        help='Eliminar fraturas que as aberturas sejam maiores que o comprimento.'
                    )

                    if eliminar_fraturas:
                        if 'ID_Fratura' in df.columns:
                            df = df[~df["ID_Fratura"].isin(invalid_df['ID_Fratura'])]
                        else:
                            df = df[~df.index.isin(invalid_df.index)]
                        
                        show_success(f"✅ {len(df)} fraturas serão consideradas após alteração.")
                        st.markdown('')
            
            # Estatísticas para validação
            show_success("✅ Dados convertidos e validados em <strong>metros</strong>")
            st.markdown("")
            
            # Mostrar estatísticas baseadas nas colunas disponíveis
            cols = st.columns(3 if available['orientation'] else 2)
            with cols[0]:
                st.metric("Comprimento", f"{df['length'].min():.3f} - {df['length'].max():.3f} m")
            
            if available['aperture']:
                with cols[1]:
                    st.metric("Abertura", f"{df['aperture'].min()*1000:.2f} - {df['aperture'].max()*1000:.2f} mm")
            
            if available['orientation']:
                col_idx = 2 if available['aperture'] else 1
                with cols[col_idx]:
                    st.metric("Orientação", f"{df['orientation'].min():.1f}° - {df['orientation'].max():.1f}°")
            
            # Mostrar quais análises estão disponíveis
            st.markdown("---")
            st.markdown("#### 📊 Análises Disponíveis")
            
            analysis_available = []
            analysis_unavailable = []
            
            # Power-law de comprimento: sempre disponível
            analysis_available.append("✅ Power-Law (comprimento)")
            
            # D-L Scaling: precisa de abertura
            if available['aperture']:
                analysis_available.append("✅ D-L Scaling Validation")
            else:
                analysis_unavailable.append("❌ D-L Scaling (requer abertura)")
            
            # Clustering: precisa de orientação
            if available['orientation']:
                analysis_available.append("✅ Clustering de Famílias")
            else:
                analysis_unavailable.append("❌ Clustering (requer orientação)")
            
            col1, col2 = st.columns(2)
            with col1:
                for a in analysis_available:
                    st.write(a)
            with col2:
                for a in analysis_unavailable:
                    st.write(a)
            
            # Selecionar colunas finais
            col_org = list(df.columns)
            lista_col_final = ['ID_Fratura', 'ID_Segmento', 'length', 'aperture', 'orientation', 'position', 'area_Fract', 'x', 'y', 'aspect_ratio']

            df_mod = df.loc[:, [c for c in lista_col_final if c in col_org]]
            
            # Retornar DataFrame e info sobre colunas disponíveis
            return df_mod, available
            
        except Exception as e:
            raise Exception(f"Erro ao carregar FRAMFRAT: {str(e)}")


    def load_scanline(self, file, length_m: float) -> pd.DataFrame:
        """
        Carrega dados de scanline - padronizado em METROS

        CORREÇÃO: Detecta unidade pelo NOME da coluna

        Estrutura esperada do arquivo:
        - Excel (.xlsx): Colunas com nomes mapeáveis
        - CSV (.csv): Colunas separadas por vírgula
        - TXT (.txt): Valores separados por tab, espaço ou vírgula
        
        Colunas esperadas (nomes flexíveis via mapeamento):
        - ID_Fratura ou id: Identificador da fratura
        - Comprimento (m) ou length: Comprimento da fratura em METROS
        - Abertura (mm) ou aperture: Abertura em milímetros (será convertida para metros)
        - Orientação (°) ou orientation: Orientação em graus
        - Posição (m) ou position: Posição ao longo da scanline em METROS
        - ID_Segmento ou ID_set: Identificador do conjunto/família (opcional)
        
        Args:
            file: Arquivo Excel/CSV/TXT
            length_m: Comprimento total da scanline em METROS
        
        Returns:
            DataFrame com dados processados em METROS       
        """
        try:
            # Detectar tipo de arquivo e carregar
            filename = file.name if hasattr(file, 'name') else str(file)
            
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(file)
            elif filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                # TXT - tentar parsear
                content = file.read().decode('utf-8') if hasattr(file, 'read') else open(file).read()
                lines = content.strip().split('\n')
                # ... (código de parsing)
                df = pd.read_csv(file, sep=None, engine='python')
            
            # ============================================================
            # CORREÇÃO PRINCIPAL: Detectar unidade pelo NOME da coluna
            # ============================================================
            original_columns = df.columns.tolist()
            
            # Detectar unidade de abertura ANTES de renomear
            aperture_unit = self._detect_unit_from_column_name(original_columns, 'aperture')
            length_unit = self._detect_unit_from_column_name(original_columns, 'length')
            
            st.info(f"🔍 Unidade detectada - Abertura: **{aperture_unit}**, Comprimento: **{length_unit}**")
            
            # Mapear colunas para nomes padrão
            df = self.map_columns(df)
            
            # Validar coluna essencial
            if 'aperture' not in df.columns:
                raise ValueError(f"Coluna 'aperture' não encontrada")
            
            # ============================================================
            # CONVERSÃO BASEADA NO NOME DA COLUNA (não nos valores!)
            # ============================================================
            
            # 1. ABERTURA: Se detectado mm no nome → converter para m
            df['aperture'] = pd.to_numeric(df['aperture'], errors='coerce')
            
            if aperture_unit == 'mm':
                st.info("🔄 Convertendo abertura: mm → metros (detectado pelo nome da coluna)")
                df['aperture'] = df['aperture'] / 1000  # mm → m
            else:
                # Fallback: verificar valores se nome não indica unidade
                if df['aperture'].max() > 0.1:  # > 100mm seria muito grande
                    st.warning("⚠️ Valores de abertura > 0.1 detectados. Assumindo mm → convertendo para metros")
                    df['aperture'] = df['aperture'] / 1000
            
            # 2. COMPRIMENTO: Se detectado mm no nome → converter para m
            if 'length' in df.columns:
                df['length'] = pd.to_numeric(df['length'], errors='coerce')
                
                if length_unit == 'mm':
                    st.info("🔄 Convertendo comprimento: mm → metros")
                    df['length'] = df['length'] / 1000
                elif df['length'].max() > 100:  # > 100m seria muito grande
                    st.warning("⚠️ Comprimentos > 100 detectados. Assumindo mm → convertendo para metros")
                    df['length'] = df['length'] / 1000
            else:
                # Calcular a partir das posições
                if 'position' in df.columns:
                    df = df.sort_values('position')
                    df['length'] = df['position'].diff().fillna(df['position'].iloc[0])
            
            # 3. POSIÇÃO: Verificar e converter se necessário
            if 'position' in df.columns:
                df['position'] = pd.to_numeric(df['position'], errors='coerce')
                if df['position'].max() > 1000:
                    df['position'] = df['position'] / 1000
            
            # 4. ORIENTAÇÃO: Manter em graus
            if 'orientation' in df.columns:
                df['orientation'] = pd.to_numeric(df['orientation'], errors='coerce') % 360
            
            # Verificar quais colunas estão disponíveis
            has_aperture = 'aperture' in df.columns
            has_orientation = 'orientation' in df.columns
            has_length = 'length' in df.columns
            
            # Remover valores inválidos
            if has_length:
                df = df[df['length'] > 0].copy()
            if has_aperture:
                df = df[df['aperture'] > 0].copy()
            
            # Adicionar metadados
            df.attrs['scanline_length'] = length_m
            df.attrs['unit'] = 'meters'
            df.attrs['has_aperture'] = has_aperture
            df.attrs['has_orientation'] = has_orientation
            
            # Validação final
            st.success(f"✅ **{len(df)}** fraturas carregadas")
            
            # Mostrar estatísticas baseadas nas colunas disponíveis
            n_cols = 1 + int(has_aperture) + int(has_orientation)
            cols = st.columns(n_cols)
            col_idx = 0
            
            if has_length:
                with cols[col_idx]:
                    st.metric("Comprimento", f"{df['length'].min():.3f} - {df['length'].max():.3f} m")
                col_idx += 1
            
            if has_aperture:
                with cols[col_idx]:
                    st.metric("Abertura", f"{df['aperture'].min()*1000:.2f} - {df['aperture'].max()*1000:.2f} mm")
                col_idx += 1
            
            if has_orientation:
                with cols[col_idx]:
                    st.metric("Orientação", f"{df['orientation'].min():.1f}° - {df['orientation'].max():.1f}°")
            
            # Calcular aspect ratio (se ambos disponíveis)
            if has_length and has_aperture:
                df['aspect_ratio'] = df['aperture'] / df['length']
            
            # Avisar sobre análises disponíveis
            if not has_aperture:
                st.warning("⚠️ Sem abertura: D-L Scaling **não disponível**")
            if not has_orientation:
                st.warning("⚠️ Sem orientação: Clustering de famílias **não disponível**")
            
            return df
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar scanline: {str(e)}")
            raise





