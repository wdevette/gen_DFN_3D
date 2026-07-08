import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
from datetime import datetime

# Importar módulos customizados
from modules.io_fractures import FractureDataLoader
from modules.powerlaw_fits import PowerLawFitter
from modules.intensity_spacing import IntensitySpacingAnalyzer
from modules.dfn_generator import DFNGenerator
from modules.visualizations import FractureVisualizer
from modules.results_exporter import ResultsExporter
from func_tools import force_dark_plotly_layout, show_error, show_success, show_info


# NOVOS MÓDULOS - Sistema de Correção de Vieses
from modules.bias_corrections import BiasCorrector, CorrectionResults
from modules.powerlaw_analysis import PowerLawAnalyzer, calculate_powerlaw_dual, PowerLawParams
from modules.bias_visualizations import (
    plot_correction_pipeline,
    plot_terzaghi_weights,
    plot_dl_scaling_validation,
    plot_powerlaw_fit,
    plot_marrett_factor_sensitivity,
    display_correction_summary
)

from modules.orientation_clustering import (
    cluster_orientations_2d, 
    cluster_orientations_3d,
    extract_orientation_stats,
    auto_determine_n_sets
)



# Configuração da página
st.set_page_config(
    page_title="DFNGen - Tool",
    page_icon=r"assets/fram_fractt.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Logo no app
#st.logo("assets/fram_fractt.png", size='small',icon_image=None)

# =============================================================================
# FUNÇÃO HELPER: Obtém dados de fraturas independente da fonte
# =============================================================================
def get_current_fracture_data():
    """
    Retorna o dataset de fraturas atualmente carregado.
    Prioriza FRAMFRAT sobre Scanline se ambos existirem.
    
    Returns:
        tuple: (DataFrame ou None, str ou None) - dados e fonte
    """
    if 'framfrat_data' in st.session_state and st.session_state.framfrat_data is not None:
        return st.session_state.framfrat_data, "FRAMFRAT"
    elif 'scanline_data' in st.session_state and st.session_state.scanline_data is not None:
        return st.session_state.scanline_data, "Scanline"
    return None, None


# CSS customizado
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px;}
</style>
""", unsafe_allow_html=True)

# Título e descrição
st.title("🕹️ Gerador de DFN por meio de análise de fraturas")
#st.markdown("")
st.markdown("### Análise de Redes de Fraturas Discretas")
# st.markdown("""
# **Análise integrada de fraturas** baseada em Marrett (1996) e Ortega et al. (2006)
# - Lei de potência para distribuições de tamanho
# - Intensidade e espaçamento size-cognizant
# - Geração de DFN estocástica
# """)
st.markdown("")
st.markdown("")

# Inicializar session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'framfrat_data' not in st.session_state:
    st.session_state.framfrat_data = None
if 'scanline_data' not in st.session_state:
    st.session_state.scanline_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'l_min_framfrat' not in st.session_state:
    st.session_state.l_min_framfrat = 0.001
if 'b_min_framfrat' not in st.session_state:
    st.session_state.b_min_framfrat = 0.0001
if 'l_min_scanline' not in st.session_state:
    st.session_state.l_min_scanline = 0.001
if 'b_min_scanline' not in st.session_state:
    st.session_state.b_min_scanline = 0.0001
if 'image_area' not in st.session_state:
    st.session_state.image_area = None  #Padrão: 0,01 ==> 100 cm² #None
if 'scanline_length' not in st.session_state:
    st.session_state.scanline_length = None


# Estados para DFN 2D com visualização reativa
if 'dfn_2d_generated' not in st.session_state:
    st.session_state.dfn_2d_generated = False
if 'dfn_2d_data' not in st.session_state:
    st.session_state.dfn_2d_data = None
if 'dfn_2d_stats' not in st.session_state:
    st.session_state.dfn_2d_stats = None
if 'dfn_2d_params' not in st.session_state:
    st.session_state.dfn_2d_params = None
if 'intensi_fract_cached' not in st.session_state:
    st.session_state.intensi_fract_cached = None

# Estados para DFN 3D
if 'dfn_3d' not in st.session_state:
    st.session_state.dfn_3d = None
if 'dfn_3d_df' not in st.session_state:
    st.session_state.dfn_3d_df = None
if 'dfn_3d_domain' not in st.session_state:
    st.session_state.dfn_3d_domain = [50.0, 50.0, 20.0]  # Default válido

# Estado para rastrear o tipo de análise atual
if 'current_analysis_type' not in st.session_state:
    st.session_state.current_analysis_type = None

# NOVOS ESTADOS - Sistema de Correção de Vieses
if 'bias_corrections_applied' not in st.session_state:
    st.session_state.bias_corrections_applied = False
if 'bias_correction_results' not in st.session_state:
    st.session_state.bias_correction_results = None
if 'powerlaw_params' not in st.session_state:
    st.session_state.powerlaw_params = None

# NOVOS ESTADOS - Parâmetros Power-Law Globais (OLS + MLE)
if 'powerlaw_results_full' not in st.session_state:
    st.session_state.powerlaw_results_full = None
if 'powerlaw_method_selected' not in st.session_state:
    st.session_state.powerlaw_method_selected = 'auto'
if 'powerlaw_global_params' not in st.session_state:
    st.session_state.powerlaw_global_params = None

# NOVO ESTADO - Colunas disponíveis nos dados
if 'available_columns' not in st.session_state:
    st.session_state.available_columns = {'length': False, 'aperture': False, 'orientation': False}

# Função para limpar dados quando o tipo de análise mudar
def clear_incompatible_data(new_analysis_type):
    """
    Limpa dados incompatíveis quando o usuário muda o tipo de análise.
    
    Args:
        new_analysis_type: Novo tipo de análise selecionado ('FRAMFRAT' ou 'Scanline')
    """
    if st.session_state.current_analysis_type != new_analysis_type:
        # Mudança detectada - limpar dados antigos
        if new_analysis_type == "FRAMFRAT":
            # Mudando para FRAMFRAT - limpar dados de Scanline
            if st.session_state.scanline_data is not None:
                st.session_state.scanline_data = None
                st.session_state.data_loaded = False
                st.session_state.analysis_results = {}
                st.session_state.dfn_2d_generated = False
                st.session_state.dfn_2d_data = None
                st.session_state.dfn_2d_stats = None
                st.session_state.dfn_2d_params = None
                st.session_state.intensi_fract_cached = None
                
        elif new_analysis_type == "Scanline":
            # Mudando para Scanline - limpar dados de FRAMFRAT
            if st.session_state.framfrat_data is not None:
                st.session_state.framfrat_data = None
                st.session_state.data_loaded = False
                st.session_state.analysis_results = {}
                st.session_state.dfn_2d_generated = False
                st.session_state.dfn_2d_data = None
                st.session_state.dfn_2d_stats = None
                st.session_state.dfn_2d_params = None
                st.session_state.intensi_fract_cached = None
                st.session_state.image_area = None
                st.session_state.scanline_length = None
        
        # Atualizar o tipo atual
        st.session_state.current_analysis_type = new_analysis_type


# Sidebar simplificado
with st.sidebar:

    st.header("ℹ️ Informações")
    
    st.markdown("""
    ### 💡 Guia de Uso
    
    **1. Dados** 📋
    - Selecione o tipo de análise
    - Configure parâmetros e filtros
    - Faça upload do arquivo
    
    **2. Ajustes** 📈
    - Escolha o método (OLS ou MLE)
    - Visualize os ajustes das leis de potência
    
        **3. Correção de Vieses** 🔧 NOVO!
    - Aplique correções científicas
    - Terzaghi, Marrett, D-L Scaling
    
    **4. Intensidade** 📏
    - Analise P10 e espaçamento
    - Compare diferentes fontes
    
    **4. DFN** 🗺️
    - Gere redes 2D e 3D
    - Configure parâmetros estocásticos
    
    **5. Exportar** 💾
    - Baixe resultados e relatórios
    """)
    
    st.divider()
    
    st.markdown("""
    ### 📚 Referências
    - Marrett (1996)
    - Ortega et al. (2006)
    - **NOVO**: Terzaghi (1965)
    - **NOVO**: D-L Scaling (2008)
    """)
    
    st.divider()
    
    st.markdown("""
    ### ⚙️ Versão
    **v2.0** - Sistema com Correção de Vieses
    """)

#st.sidebar.image(image='assets/fram_fractt.png', width=170)


# Área principal - Abas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Dados", 
    "📈 Power-Law & Ajustes",
    #"🔧 Correção de Vieses",  # ← NOVA TAB!
    "📏 Intensidade & Espaçamento",
    "📐 Gerador de DFN",  
    "💾 Exportar"
])


# Tab 1: Upload de Dados
with tab1:
    st.header("📊 Upload de Dados")
    
    # Seleção do tipo de análise
    st.subheader("📋 Tipo de Análise")
    analysis_type = st.radio(
        "Selecione o tipo de dados que deseja analisar:",
        options=["FRAMFRAT", "Scanline"],
        index=None,
        horizontal=True,
        help="FRAMFRAT: Análise de imagens 2D | Scanline: Análise linear 1D"
    )
    
    # Limpar dados incompatíveis quando o tipo de análise mudar
    if analysis_type is not None:
        clear_incompatible_data(analysis_type)
    
    if analysis_type is None:
        show_info("👆 Por favor, selecione o tipo de análise para continuar")
        st.markdown("")
        show_info('Garanta que as aberturas e comprimentos estevam em milimetros (mm)')
        st.markdown("")
    
    elif analysis_type == "FRAMFRAT":
        st.divider()
        
        # Indicador de status com opção de limpar
        if st.session_state.framfrat_data is not None:
            col_status1, col_status2 = st.columns([2, 1])
            with col_status1:
                st.markdown("")
                show_success("✅ Dados FRAMFRAT já processados na memória")
                st.markdown("")
                
            with col_status2:
                st.markdown("")
                if st.button("🗑️ Limpar", key="clear_framfrat", help="Limpar dados processados"):
                    st.session_state.framfrat_data = None
                    st.session_state.data_loaded = False
                    st.session_state.analysis_results = {}
                    st.rerun()
        
        #DADOS FRAMFRAT
        st.subheader("FRAMFRAT (.xlsx)")
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("##### 📂 Upload de Arquivo")
            uploaded_framfrat = st.file_uploader(
                "Arquivo FRAMFRAT (.xlsx)",
                type=['xlsx', 'xls'],
                help="Arquivo Excel com colunas: comprimento, abertura, orientação (opcional), x, y",
                key="framfrat_upload"
            )
            
            st.markdown("##### ⚙️ Parâmetros da Imagem")
            sub_col1, sub_col2 = st.columns([1, 1], gap='medium')
            with sub_col1:
                image_area = st.number_input(
                    "Área da imagem (m²)",
                    #min_value=0.01,
                    #value=0.01,  # Área padrão: 100 cm² = 0.01 m²
                    step=100.0,
                    help="Área real representada pela imagem analisada",
                    key="img_area"
                )

                # Atualizar session_state quando mudar
                st.session_state.image_area = image_area
            
            with sub_col2:
                pixel_per_mm = st.number_input(
                    "Resolução/Escala (pixels/mm)",
                    min_value=0.1,
                    value=1.0, # MUDOU: era 100.0 pixels/m, agora 10 pixels/mm
                    step=1.0,
                    help="Número de pixels por metro na imagem",
                    key="pixel_scale"
                )
        
        # Processar dados quando botão é clicado
        with col2:                        
            st.markdown("##### 🔍 Filtros de Dados")
            
            sub_col3, sub_col4 = st.columns(2)
            with sub_col3: 
                l_min = st.number_input(
                    "Comprimento mínimo (m)", 
                    min_value=0.0, 
                    value=0.001,  # 1 mm em metros
                    step=0.001,
                    format="%.3f",  #"%.3f",
                    help="Filtrar fraturas menores que este valor",
                    key="l_min_framfrat"
                )
            with sub_col4:
                # b_min é opcional - só relevante se dados tiverem abertura
                b_min = st.number_input(
                    "Abertura mínima (m) - opcional", 
                    min_value=0.0, 
                    value=0.0001,  # 0.1 mm em metros
                    step=0.0001,
                    format="%.4f",
                    help="Filtrar fraturas com abertura menor que este valor (apenas se dados tiverem abertura)",
                    key="b_min_framfrat"
                )

            # NOVO: Checkbox para x_min automático
            auto_xmin = st.checkbox(
                "🎯 Detectar x_min automaticamente (Clauset et al. 2009)",
                value=False,
                help="Calcula o x_min ótimo minimizando a estatística KS",
                key="auto_xmin_framfrat"
            )

            if uploaded_framfrat and image_area and l_min:
                with st.spinner("Processando dados FRAMFRAT..."):
                        try:
                            loader = FractureDataLoader()
                            # load_framfrat agora retorna (DataFrame, Dict de colunas disponíveis)
                            framfrat_data, available_cols = loader.load_framfrat(
                                uploaded_framfrat,
                                image_area,
                                pixel_per_mm
                            )

                            # Salvar parâmetros no session state 
                            st.session_state.framfrat_data = framfrat_data
                            st.session_state.data_loaded = True
                            st.session_state.analysis_type = "FRAMFRAT"
                            st.session_state.available_columns = available_cols  # NOVO: quais colunas disponíveis
                            
                        except Exception as e:
                            show_error(f"❌ Erro ao processar FRAMFRAT: {str(e)}")
                            st.exception(e)          
            else:
                st.markdown("")
                show_info("⚠️ Carregue e preencha os valores dos campos para pré-processamento dos dados")
      
        if uploaded_framfrat:
            
            st.markdown("")
            if st.button(
                "🚀 Processar Dados FRAMFRAT",
                type="primary",
                #width='stretch',
                disabled=not uploaded_framfrat,
                help="Clique para processar os dados carregados",
                key="btn_process_framfrat",
            ):
                # Preview dos dados (só mostra se dados foram processados)
                if st.session_state.framfrat_data is not None:
                    framfrat_data = st.session_state.framfrat_data
                    
                    st.divider()
                    show_success(f"✅ <strong>{len(framfrat_data)}</strong> fraturas processadas")
                    st.markdown("")  
                    
                    #Intensidade das fraturas p/ FRAMFRAT (2D) P20, P21, P22
                    intensity_spacy_insta = IntensitySpacingAnalyzer()
                    intensi_fract = intensity_spacy_insta.calculate_from_framfrat(framfrat_data, image_area)
                    
                    with st.expander("📝 Preview dos dados FRAMFRAT", expanded=True):
                        # Mostrar primeiras linhas
                        preview_df = framfrat_data[['ID_Fratura', 'ID_Segmento', 'length', 'aperture']].head(5).copy()

                        # Detectar se devemos exibir ID_Segmento (não-nulo e não string vazia)
                        show_segmento = (
                            "ID_Segmento" in preview_df.columns and
                            preview_df["ID_Segmento"].replace(r"^\s*$", pd.NA, regex=True).notna().any()
                        )
                        
                        # Criar DataFrame para display com unidades corretas
                        display_df = pd.DataFrame({
                            'ID_Fratura': preview_df['ID_Fratura'],
                            'Comprimento (m)': preview_df['length'],
                            'Abertura (mm)': preview_df['aperture'] * 1000  # Converter para mm para exibição
                        })

                        display_df['Comprimento (m)'] = display_df['Comprimento (m)'] \
                            .apply(lambda x: f"{x:.3f}".replace('.', ','))
                        display_df['Abertura (mm)'] = display_df['Abertura (mm)'] \
                            .apply(lambda x: f"{x:.2f}".replace('.', ','))
                        
                        # Se houver 'ID_Segmento' válido, insere e reordena para ficar como 2ª coluna
                        if show_segmento:
                            display_df["ID_Segmento"] = preview_df["ID_Segmento"]
                            desired_order = ["ID_Fratura", "ID_Segmento", "Comprimento (m)", "Abertura (mm)"]
                            display_df = display_df.reindex(columns=desired_order)

                        st.dataframe(display_df, hide_index=True)
                        st.divider()
                        
                        # Estatísticas
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Total de Fraturas", 
                                len(framfrat_data)
                            )

                            st.metric(
                                "P20", 
                                f"{intensi_fract['P20']:.2f} 1/m²".replace(".", ","),
                                help="Número de fraturas por área [1/m²] | 'N/A'"
                            )

                        with col2:
                            st.metric(
                                "Compr. médio", 
                                f"{framfrat_data['length'].mean():.3f} m".replace(".", ",")
                            )

                            st.metric(
                                "P21",
                                f"{intensi_fract['P21']:.2f} 1/m".replace(".", ","),
                                help="Comprimento total da fraturas por área [1/m²] | 'ΣL/A'"
                            )

                        with col3:
                            st.metric(
                                "Abertura média", 
                                f"{framfrat_data['aperture'].mean() * 1000:.2f} mm".replace(".", ",")
                            )

                            st.metric( #COLOCAR HELP com a formula e a descrição do que o mesmo representa
                                "P22",
                                f"{intensi_fract['P22']:.2f}".replace(".", ","),
                                help="Área total de fraturas por área [adimensional] | 'ΣAf / A'"
                            )
                        col_aux1, col_aux2 = st.columns([3, 2], gap='large')
                        with col_aux1:
                            show_info('Para o cálculo dos índices de intensidadade de fraturas <strong>P20, P21 e P22</strong> foi considera as medidas em <strong>[metros | m]</strong>')
                        
                        # Estatísticas adicionais
                        st.divider()
                        st.write("📝 **Estatísticas Detalhadas:**")
                        stats_df = pd.DataFrame({
                            'Métrica': ['Mínimo', 'Máximo', 'Mediana', 'Desvio Padrão'],
                            'Comprimento (m)': [
                                f"{framfrat_data['length'].min():.3f}".replace(".", ","),
                                f"{framfrat_data['length'].max():.3f}".replace(".", ","),
                                f"{framfrat_data['length'].median():.3f}".replace(".", ","),
                                f"{framfrat_data['length'].std():.3f}".replace(".", ",")
                            ],
                            'Abertura (mm)': [
                                f"{framfrat_data['aperture'].min() * 1000:.2f}".replace(".", ","),
                                f"{framfrat_data['aperture'].max() * 1000:.2f}".replace(".", ","),
                                f"{framfrat_data['aperture'].median() * 1000:.2f}".replace(".", ","),
                                f"{framfrat_data['aperture'].std() * 1000:.2f}".replace(".", ",")
                            ]
                        })
                        st.table(stats_df)

        else:
            st.markdown('')
            show_info("⚠️ Carregue e preencha os valores dos campos em falta para liberar o batão de processamento dos dados")
            st.markdown("")

    elif analysis_type == "Scanline":
        st.divider()
        
        # Indicador de status com opção de limpar
        if st.session_state.scanline_data is not None:
            col_status1, col_status2 = st.columns([3, 1])
            with col_status1:
                st.markdown("")
                show_success("✅ Dados Scanline já processados na memória")
                st.markdown("")
            with col_status2:
                if st.button("🗑️ Limpar", key="clear_scanline", help="Limpar dados processados"):
                    st.session_state.scanline_data = None
                    st.session_state.data_loaded = False
                    st.session_state.analysis_results = {}
                    st.rerun()
        
        #DADOS SCANLINE
        st.subheader("📝 Análise Scanline (Linear 1D)")
        
        col1, col2 = st.columns([1, 1], gap='large')
        with col1:
            st.markdown("##### 📂 Upload de Arquivo")
            uploaded_scanline = st.file_uploader(
                "Arquivo Scanline (.txt/.csv/.xlsx)",
                type=['txt', 'csv', 'xlsx'],
                help="Arquivo com posições e aberturas das fraturas",
                key="scanline_upload"
            )
            
            if uploaded_scanline:
                st.markdown("")
                show_success("✅ Arquivo carregado!")
                st.markdown("")
                            
            st.markdown("##### 🔧 Parâmetros da Scanline")
            scanline_length = st.number_input(
                "Comprimento da scanline (m)",
                min_value=0.1,
                value=10.0,
                step=0.1,
                help="Comprimento total da linha de amostragem",
                key="scan_length"
            )

            # Atualizar session_state quando mudar
            st.session_state.scanline_length = scanline_length
            
            scanline_azimuth = st.number_input(
                "Azimute da linha (°)", 
                min_value=0, 
                max_value=360, 
                value=0,
                help="Orientação da scanline",
                key="scan_azimuth"
            )
            
            # Botão de processar
            process_scanline = st.button(
                "🚀 Processar Dados Scanline",
                type="primary",
                #width='stretch',
                disabled=not uploaded_scanline,
                help="Clique para processar os dados carregados",
                key="btn_process_scanline"
            )
        
        # Processar dados quando botão é clicado
        with col2:
            
            st.markdown("##### 🔍 Filtros de Dados")
            l_min_scan = st.number_input(
                "Espaçamento mínimo (m)", 
                min_value=0.0, 
                value=0.001, 
                step=0.001,
                format="%.3f",
                help="Filtrar fraturas com espaçamento menor que este valor",
                key="l_min_scanline"
            )
            
            b_min_scan = st.number_input(
                "Abertura mínima (m)", 
                min_value=0.0, 
                value=0.0001, 
                step=0.0001,
                format="%.4f",
                help="Filtrar fraturas com abertura menor que este valor",
                key="b_min_scanline"
            )
            
            if process_scanline and uploaded_scanline:
                with st.spinner("Processando dados Scanline..."):
                    try:
                        loader = FractureDataLoader()
                        scanline_data = loader.load_scanline(
                            uploaded_scanline,
                            scanline_length
                        )
                        st.session_state.scanline_data = scanline_data
                        st.session_state.data_loaded = True
                        st.session_state.analysis_type = "Scanline"
                        
                        st.markdown("")
                        show_success("✅ Dados processados com sucesso!")
                        st.markdown("")

                    except Exception as e:
                        st.markdown("")
                        show_error(f"❌ Erro ao processar Scanline: {str(e)}")
                        st.markdown("")
        
        # Preview dos dados (só mostra se dados foram processados)
        if st.session_state.scanline_data is not None:
            scanline_data = st.session_state.scanline_data
            
            st.divider()
            st.markdown("")
            show_success(f"✅ {len(scanline_data)} fraturas processadas")
            st.markdown("")
            

            # Calcular intensidades de fraturamento para Scanline (1D) - P10, P11
            intensity_scanline_inst = IntensitySpacingAnalyzer()
            intensi_scanline = intensity_scanline_inst.calculate_from_scanline(
                n_fractures=len(scanline_data),
                scanline_length_m=scanline_length,
                mean_length_m=scanline_data['length'].mean() if 'length' in scanline_data.columns else None,
                mean_aperture_m=scanline_data['aperture'].mean() if 'aperture' in scanline_data.columns else None
            )            


            # Preview dos dados
            st.markdown("")
            with st.expander("📝 Preview dos dados Scanline"):

                #Troca dos nomes das colunas -  para melhor visualização do utilizador
                df_scanline_data = scanline_data.copy()
                df_scanline_data.drop(columns=['aspect_ratio'], inplace=True)
                lista_scanline_header = ['ID_Fratura', "ID_Segmento", "Comprimento", "Abertura", "Orientação (°)",
                                         "Posição Scanline"]
                
                df_scanline_data.columns = lista_scanline_header
                st.dataframe(df_scanline_data.head(5), hide_index=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Fraturas", len(scanline_data))
                    
                    st.metric(
                        "P10",
                        f"{intensi_scanline.P10:.2f} 1/m".replace(".", ",") if intensi_scanline.P10 is not None else "N/A",
                        help="Frequência linear de fraturas [1/m] | 'N/L'"
                        #help="Número de fraturas por metro linear [1/m] | 'N/L'" 
                    )

                with col2:
                    st.metric("Espaçamento médio (m)", f"{scanline_data['length'].mean():.3f}".replace('.', ','))

                    st.metric(
                        "P11",
                        f"{intensi_scanline.P11:.2f}".replace(".", ",") if intensi_scanline.P11 is not None else "N/A",
                        help="Comprimento total de fraturas por comprimento do scanline [adimensional] | 'ΣL_frat / L_scanline'"
                    )
                
                with col3:
                    st.metric("Abertura média (mm)", f"{scanline_data['aperture'].mean()*1000:.2f}".replace('.', ','))

                    #Espaçamento médio calculado (1/P10)
                    spacing_medio = 1/intensi_scanline.P10 if intensi_scanline.P10 and intensi_scanline.P10 > 0 else None
                    st.metric(
                        "Espaçamento (1/P10)",
                        f"{spacing_medio:.3f} m".replace(".", ",") if spacing_medio is not None else "N/A",
                        help="Espaçamento médio calculado como inverso de P10 [m]"
                    )
                
                col_aux1, col_aux2 = st.columns([3, 2], gap='large')
                with col_aux1:
                    show_info('Para o cálculo dos índices de intensidade de fraturas <strong>P10 e P11</strong> foram consideradas as medidas em <strong>[metros | m]</strong>')
                
                # Estatísticas detalhadas
                st.divider()
                st.write("📝 **Estatísticas Detalhadas:**")
                
                stats_scan_df = pd.DataFrame({
                    'Métrica': ['Mínimo', 'Máximo', 'Mediana', 'Desvio Padrão']
                })
                
                if 'length' in scanline_data.columns:
                    stats_scan_df['Espaçamento (m)'] = [
                        f"{scanline_data['length'].min():.3f}".replace(".", ","),
                        f"{scanline_data['length'].max():.3f}".replace(".", ","),
                        f"{scanline_data['length'].median():.3f}".replace(".", ","),
                        f"{scanline_data['length'].std():.3f}".replace(".", ",")
                    ]
                
                if 'aperture' in scanline_data.columns:
                    stats_scan_df['Abertura (mm)'] = [
                        f"{scanline_data['aperture'].min() * 1000:.2f}".replace(".", ","),
                        f"{scanline_data['aperture'].max() * 1000:.2f}".replace(".", ","),
                        f"{scanline_data['aperture'].median() * 1000:.2f}".replace(".", ","),
                        f"{scanline_data['aperture'].std() * 1000:.2f}".replace(".", ",")
                    ]
                
                st.table(stats_scan_df)
   
   
   
   
   
   
   
   
   
    # Seção de comparação (aparece apenas se ambos os dados forem carregados)
    st.divider()
    
    if st.checkbox("🔄 Modo de Comparação", help="Carregue ambos os tipos de dados para comparar"):
        st.subheader("📋 Carregar Dados para Comparação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**FRAMFRAT**")
            if st.session_state.framfrat_data is not None:
                st.markdown("")
                show_success("✅ Dados FRAMFRAT processados")
                st.markdown("")
                st.metric("Fraturas", len(st.session_state.framfrat_data))
            else:
                st.markdown("")
                show_info("👆 Selecione FRAMFRAT acima e processe os dados")
                st.markdown("")
        
        with col2:
            st.write("**Scanline**")
            if st.session_state.scanline_data is not None:
                st.markdown("")
                show_success("✅ Dados Scanline processados")
                st.markdown("")
                st.metric("Fraturas", len(st.session_state.scanline_data))
            else:
                st.markdown("")
                show_info("👆 Selecione Scanline acima e processe os dados")
                st.markdown("")
        
        # Verificar se ambos estão carregados
        if st.session_state.framfrat_data is not None and st.session_state.scanline_data is not None:
            st.session_state.comparison_mode = True
            st.markdown("")
            show_success("✅ Modo de comparação ativado! Vá para a aba 'Intensidade & Espaçamento' para análise comparativa")
            st.markdown("")
        else:
            st.markdown("")
            show_info("⚠️ Processe ambos os tipos de dados para ativar o modo de comparação")
            st.markdown("")


# ============================================================================
# Tab 2: Ajustes de Lei de Potência & Pipeline de Correções de vieses
# ============================================================================
with tab2:
    st.header("🧩 Lei de Potência e Pipeline de Correções de vieses")
    
    if st.session_state.data_loaded:
        tab_PL, tab_pipe, tab_fam  = st.tabs(['🔋 Power-Law', '🔧Correção de Vieses', '🎏Famílias de Fraturas'])
    

        current_data, data_info = get_current_fracture_data()

        if current_data is None:
            st.markdown("")
            show_info("⚠️ Nenhum dado carregado. Carregue dados FRAMFRAT ou Scanline primeiro.")
            st.markdown("")

        else: 
            required_col = ['ID_Fratura', 'ID_Segmento', 'length', 'aperture', 'orientation']       
            missing_col = [col for col in required_col if col not in current_data.columns]
            if missing_col: #Troca nos nomes das colunas que não foram localizadas no dataset e exibição delas
                if 'length' in missing_col:
                    missing_col[missing_col.index("length")] = 'Comprimento'
                if 'aperture' in missing_col:
                    missing_col[missing_col.index("aperture")] = 'Abertura'
                if 'orientation' in missing_col:
                    missing_col[missing_col.index("orientation")] = 'Orientação'

                st.markdown("")
                show_info(f"⚠️ As colunas {missing_col} não foram localizadas.")
                st.markdown("")
                st.stop()

            else:
                #Power-Law
                with tab_PL:
                    st.subheader("Aplicação da Power-Law")
                    
                    # Configuração de ajuste
                    st.subheader("⚙️ Configuração de Ajuste para Lei de Potência")

                    #col_config1, col_config2, col_config3 = st.columns([0.3, 0.35, 0.35])
                    col_config1, col_config2  = st.columns([1, 3], gap='medium')
                    with col_config1:
                        fit_method = st.selectbox(
                            "Método de ajuste", 
                            [None, "OLS", "MLE"],
                            format_func=lambda x: "Selecione um método" if x is None else f"{x} ({'log-log' if x == 'OLS' else 'Clauset et al.'})",
                            help="OLS: Mínimos quadrados ordinários\\nMLE: Máxima verossimilhança"
                        )
                    
                    with col_config2:
                        st.markdown("")
                        st.markdown("")
                        col_config21, col_config22 = st.columns([0.5, 1], gap='small')

                        with col_config21:
                            auto_xmin = st.checkbox(
                                "🎯 x_min automático",
                                value=True,
                                help="Detecta x_min ótimo minimizando KS (Clauset et al. 2009)"
                            )
                    
                   # with col_config3:
                        with col_config22:
                            auto_xmax = st.checkbox(
                                "🎯 x_max automático",
                                value=False,
                                help="Detecta x_max removendo outliers"
                            )
                    
                    if fit_method is None:
                        st.markdown("")
                        show_info("⚠️ Por favor, selecione um método de ajuste para continuar")
                        st.markdown("")
                
                    else:
                        fitter = PowerLawFitter()
                        viz = FractureVisualizer()
                                                    
                        #else:
                        data = current_data
                        
                        # Calcular x_min/x_max
                        if auto_xmin:
                            with st.spinner("Calculando parâmetros ótimos..."):
                                l_min_result = fitter.find_optimal_xmin(
                                    data['length'].values, return_details=True
                                )
                                b_min_result = fitter.find_optimal_xmin(
                                    data['aperture'].values, return_details=True
                                )
                                l_min = l_min_result['x_min']
                                b_min = b_min_result['x_min']
                                
                                # Mostrar resultados x_min
                                st.markdown("")
                                show_success("✅ x_min= calculado automaticamente")
                                st.markdown("")
                                st.markdown("")

                                col_m1, col_m2 = st.columns(2)
                                with col_m1:
                                    st.metric(
                                        "l_min (comprimento)",
                                        f"{l_min:.4f} m",
                                        f"Dados usados: {l_min_result['fraction_used']*100:.1f}%"
                                    )
                                with col_m2:
                                    st.metric(
                                        "b_min (abertura)",
                                        f"{b_min:.4f} m", 
                                        f"Dados usados: {b_min_result['fraction_used']*100:.1f}%"
                                    )
                        else:
                            l_min = st.session_state.get('l_min_framfrat', 0.001)
                            b_min = st.session_state.get('b_min_framfrat', 0.0001)
                        
                        # Calcular x_max
                        if auto_xmax:
                            l_max = fitter.find_optimal_xmax(data['length'].values, l_min)
                            b_max = fitter.find_optimal_xmax(data['aperture'].values, b_min)
                        else:
                            l_max = None
                            b_max = None
                        
                        st.markdown("---")
                        
                        # Ajustes
                        results = {}
                        
                        col1,col_mid, col2 = st.columns([1, 0.01, 1])
                        
                        with col1:
                            st.subheader("📏 Comprimento (l)")
                            
                            l_fit = fitter.fit_power_law(
                                data['length'].values,
                                l_min,
                                method=fit_method,
                                x_max=l_max
                            )
                            results['length_fit'] = l_fit
                            
                            # Gráfico
                            fig_l = viz.plot_power_law_fit(data['length'].values, l_fit)
                            st.plotly_chart(fig_l, width='stretch')
                            

                            xmax_len = l_fit.get('x_max', 'N/A')
                            if isinstance(xmax_len, (int, float)):
                                xmax_len = f"{xmax_len:.3f}"
                            # Métricas
                            st.markdown(f"""
                            **Parâmetros ajustados:**
                            - Expoente (α): **{l_fit['exponent']:.3f}**
                            - R²: {l_fit['r_squared']:.3f}
                            - x_min: {l_fit['x_min']:.3f} m
                            - x_max: {xmax_len} m
                            - Dados: {l_fit['n_data']}
                            """)
                            
                            # Diagnósticos
                            diag_l = fitter.diagnose_fit_quality(
                                data['length'].values, l_min, l_fit['exponent']
                            )
                            
                            if diag_l['is_good_fit']:
                                st.markdown("")
                                show_success("✅ Ajuste de boa qualidade")
                            else:
                                for issue in diag_l['issues']:
                                    st.markdown("")
                                    show_info(f"⚠️ **{issue['type']}**: {issue['description']}")
                        
                        with col2:
                            st.subheader("📏 Abertura (b)")
                            
                            b_fit = fitter.fit_power_law(
                                data['aperture'].values,
                                b_min,
                                method=fit_method,
                                x_max=b_max
                            )
                            results['aperture_fit'] = b_fit
                            
                            # Gráfico
                            fig_b = viz.plot_power_law_fit(data['aperture'].values, b_fit)
                            st.plotly_chart(fig_b, width='stretch')
                            
                            xmax_ape = b_fit.get('x_max', 'N/A')
                            if isinstance(xmax_ape, (int, float)):
                                xmax_ape = f"{xmax_ape:.3f}"

                            # Métricas
                            st.markdown(f"""
                            **Parâmetros ajustados:**
                            - Expoente (α): **{b_fit['exponent']:.3f}**
                            - R²: {b_fit['r_squared']:.3f}
                            - x_min: {b_fit['x_min']:.3f} m
                            - x_max: {xmax_ape}
                            - Dados: {b_fit['n_data']}
                            """)
                            
                            # Diagnósticos
                            diag_b = fitter.diagnose_fit_quality(
                                data['aperture'].values, b_min, b_fit['exponent']
                            )
                            
                            if diag_b['is_good_fit']:
                                st.markdown("")
                                show_success("✅ Ajuste de boa qualidade")
                            else:
                                for issue in diag_b['issues']:
                                    st.markdown("")
                                    show_info(f"⚠️ **{issue['type']}**: {issue['description']}")
                        
                        # Salvar resultados
                        st.session_state.analysis_results = results
                        
                        # Relação b-l (mantido igual)
                        st.markdown("---")
                        col3, col4 = st.columns(2)
                        with col3:
                            st.subheader("Relação b-l")
                            bl_fit = fitter.fit_aperture_length_relation(
                                data['aperture'].values,
                                data['length'].values
                            )
                            results['bl_relation'] = bl_fit
                            
                            fig_bl = viz.plot_aperture_length_relation(
                                data['aperture'].values,
                                data['length'].values,
                                bl_fit
                            )
                            st.plotly_chart(fig_bl, width='stretch')
                            
                            st.markdown(f"""
                            **Relação b = g.l^m:**
                            - Expoente (m): {bl_fit['m']:.3f}
                            - Coeficiente (g): {bl_fit['g']:.2e}
                            - R²: {bl_fit['r_squared']:.3f}
                            """)

                #Pipiline de Correção de Vieses
                with tab_pipe:
                    
                    # ================================
                    #     TAB 1: CORREÇÃO DE VIESES 
                    # ================================
                    st.header("⚒️ Sistema de Correção de Vieses")
                    
                    st.markdown("""
                    Aplique correções científicas para corrigir vieses sistemáticos nos dados:
                    - **Terzaghi (1965)**: Correção de orientação (scanline)
                    - **Marrett (1996)**: Correção de truncamento power-law
                    - **D-L Scaling (2008)**: Validação física (FRAMFRAT)
                    """)
                    
                    st.markdown("---")
                    # Determinar tipo de dado
                    if data_info == 'FRAMFRAT':
                        #data = st.session_state.framfrat_data
                        data = current_data
                        data_type = data_info
                        st.info("📊 Tipo de Dado: **FRAMFRAT 2D**")
            
                        default_area  = st.session_state.get('image_area')

                        # FRAMFRAT Pipeline
                        st.subheader("⚙️ Parâmetros do Pipeline FRAMFRAT")
                        
                        col1, col2, col3 = st.columns([1, 1, 2], gap='medium')
                        with col1:
                            area_m2 = st.number_input(
                                "Área (m²)",
                                value=default_area, #ADD: ver como resolver a questão de passar a area_m2 da tab1 para aqui 
                                format="%.3f",
                                help="Área da amostra em m²",
                                key="bias_area"
                            )
                        
                        #Remoção do valor aguardado por default no sessão
                        st.session_state.image_area = None

                        with col2:
                            rock_type = st.selectbox(
                                "Tipo de Rocha",
                                options=[
                                    'volcanic',      # a = 0.078 (rochas vulcânicas)
                                    'trachyte',      # a = 0.078 (ITAPOAMA)
                                    'basalt',        # a = 0.078 (Ethiopian dikes)
                                    'sandstone',     # a = 0.022
                                    'limestone',     # a = 0.025
                                    'granite',       # a = 0.035
                                    'shale',         # a = 0.012
                                    'default'        # a = 0.030
                                ],
                                index=1,  # Default: trachyte para Itapoama
                                format_func=lambda x: {
                                    'volcanic': 'Rochas Vulcânicas (a=0.078)',
                                    'trachyte': 'Traquito (a=0.078) - Itapoama',
                                    'basalt': 'Basalto (a=0.078)',
                                    'sandstone': 'Arenito (a=0.022)',
                                    'limestone': 'Calcário (a=0.025)',
                                    'granite': 'Granito (a=0.035)',
                                    'shale': 'Folhelho (a=0.012)',
                                    'default': 'Padrão (a=0.030)'
                                }.get(x, x),
                                help="Tipo de rocha para D-L Scaling (Schultz et al., 2008)"
                            )
                        
                        with col3:
                            st.markdown("")
                            st.markdown("")
                            # Verificar se abertura está disponível para D-L validation
                            has_aperture = st.session_state.get('available_columns', {}).get('aperture', False)
                            
                            if has_aperture:
                                apply_dl = st.checkbox(
                                    "Aplicar D-L Validation",
                                    value=True,
                                    help="Remove fraturas fisicamente impossíveis"
                                )
                            else:
                                apply_dl = False
                                st.warning("⚠️ D-L Validation **não disponível** (dados sem abertura)")
                        
                        # ============================================================
                        # ANÁLISE POWER-LAW - OLS e MLE SIMULTÂNEOS
                        # ============================================================
                        st.markdown("### 📊 Análise Power-Law (OLS + MLE)")
                        
                        # Verificar se já existem parâmetros globais da aba anterior
                        if st.session_state.powerlaw_global_params is not None:
                            params = st.session_state.powerlaw_global_params
                            st.info(f"ℹ️ Parâmetros disponíveis da análise anterior: α = {params.alpha:.3f}, l_min = {params.l_min:.4f} m, l_max = {params.l_max:.4f} m")
                            
                            use_global = st.checkbox(
                                "✅ Usar parâmetros globais (RECOMENDADO)",
                                value=True,
                                help="Use os parâmetros já calculados na aba de Lei de Potência"
                            )
                        else:
                            use_global = False
                            st.warning("⚠️ Calcule os parâmetros Power-Law abaixo ou na aba 'Lei de Potência'")
                        
                        if not use_global:
                            # Parâmetros de percentil
                            col_p1, col_p2 = st.columns(2)
                            with col_p1:
                                perc_min = st.slider("Percentil l_min (%)", 1.0, 20.0, 5.0, 1.0, key="perc_min_bias")
                            with col_p2:
                                perc_max = st.slider("Percentil l_max (%)", 80.0, 99.0, 95.0, 1.0, key="perc_max_bias")
                            
                            if st.button("🔍 Calcular Power-Law (OLS + MLE)", type="primary", key="calc_pl"):
                                with st.spinner("Analisando distribuição power-law..."):
                                    try:
                                        analyzer = PowerLawAnalyzer()
                                        pl_results = analyzer.analyze_both_methods(
                                            lengths=data['length'].values,
                                            percentile_min=perc_min,
                                            percentile_max=perc_max,
                                            validate=True
                                        )
                                        
                                        st.session_state.powerlaw_results_full = pl_results
                                        st.success("✅ Análise concluída!")
                                        
                                    except Exception as e:
                                        st.error(f"❌ Erro: {str(e)}")
                                        st.exception(e)
                        
                        # Exibir resultados se disponíveis
                        if st.session_state.powerlaw_results_full is not None or st.session_state.powerlaw_global_params is not None:
                            
                            if use_global and st.session_state.powerlaw_global_params is not None:
                                # Usar parâmetros globais
                                params = st.session_state.powerlaw_global_params
                                alpha_to_use = params.alpha
                                l_min_to_use = params.l_min
                                l_max_to_use = params.l_max
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("α (usado)", f"{alpha_to_use:.3f}")
                                with col2:
                                    st.metric("l_min", f"{l_min_to_use:.4f} m")
                                with col3:
                                    st.metric("l_max", f"{l_max_to_use:.4f} m")
                                with col4:
                                    st.metric("Método", params.method_used)
                                
                                # Criar dict compatível para pipeline
                                st.session_state.powerlaw_params = {
                                    'alpha': alpha_to_use,
                                    'l_min': l_min_to_use,
                                    'l_max': l_max_to_use,
                                    'validation': {'is_valid': params.is_valid, 'warnings': params.warnings}
                                }
                                
                            elif st.session_state.powerlaw_results_full is not None:
                                pl = st.session_state.powerlaw_results_full
                                
                                # Parâmetros de range (GLOBAIS)
                                st.markdown("#### 📏 Parâmetros de Range")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("l_min", f"{pl['l_min']:.4f} m")
                                with col2:
                                    st.metric("l_max", f"{pl['l_max']:.4f} m")
                                
                                # Comparação OLS vs MLE
                                st.markdown("#### 📊 Comparação OLS vs MLE")
                                
                                col_ols, col_mle, col_weighted = st.columns(3)
                                
                                with col_ols:
                                    ols_valid = "✅" if pl['ols']['validation']['is_valid'] else "⚠️"
                                    st.markdown(f"**{ols_valid} OLS**")
                                    st.metric("α OLS", f"{pl['ols']['alpha']:.3f}")
                                    st.metric("R² OLS", f"{pl['ols']['r_squared']:.4f}")
                                
                                with col_mle:
                                    mle_valid = "✅" if pl['mle']['validation']['is_valid'] else "⚠️"
                                    st.markdown(f"**{mle_valid} MLE**")
                                    st.metric("α MLE", f"{pl['mle']['alpha']:.3f}")
                                    st.metric("R² MLE", f"{pl['mle']['r_squared']:.4f}")
                                
                                with col_weighted:
                                    st.markdown("**⚖️ Ponderado**")
                                    st.metric("α Ponderado", f"{pl['weighted']['alpha']:.3f}")
                                    st.caption(f"Pesos: OLS={pl['weighted']['weight_ols']:.2f}, MLE={pl['weighted']['weight_mle']:.2f}")
                                
                                # Recomendação automática
                                rec = pl['recommendation']
                                st.success(f"🎯 **Recomendado: {rec['method']}** — {rec['reason']}")
                                
                                # Seleção do método
                                st.markdown("#### 🎛️ Selecione o Método")
                                
                                method_options = {
                                    'auto': f"🎯 Auto ({rec['method']})",
                                    'OLS': f"OLS (α={pl['ols']['alpha']:.3f})",
                                    'MLE': f"MLE (α={pl['mle']['alpha']:.3f})",
                                    'weighted': f"Ponderado (α={pl['weighted']['alpha']:.3f})"
                                }
                                
                                selected_method = st.radio(
                                    "Método:",
                                    options=list(method_options.keys()),
                                    format_func=lambda x: method_options[x],
                                    index=0,
                                    horizontal=True,
                                    key="pl_method_bias"
                                )
                                
                                # Obter alpha pelo método selecionado
                                analyzer = PowerLawAnalyzer()
                                analyzer.results = pl  # Restaurar resultados
                                params = analyzer.get_params(method=selected_method)
                                
                                alpha_to_use = params.alpha
                                l_min_to_use = params.l_min
                                l_max_to_use = params.l_max
                                
                                # Armazenar globalmente
                                st.session_state.powerlaw_global_params = params
                                
                                # Criar dict compatível para pipeline
                                st.session_state.powerlaw_params = {
                                    'alpha': alpha_to_use,
                                    'l_min': l_min_to_use,
                                    'l_max': l_max_to_use,
                                    'validation': {'is_valid': params.is_valid, 'warnings': params.warnings}
                                }
                                
                                # Mostrar parâmetros selecionados
                                st.markdown("#### ✅ Parâmetros Finais")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("α (usado)", f"{alpha_to_use:.3f}")
                                with col2:
                                    st.metric("l_min", f"{l_min_to_use:.4f} m")
                                with col3:
                                    st.metric("l_max", f"{l_max_to_use:.4f} m")
                                
                                # Visualização
                                with st.expander("📈 Visualizar Ajuste Power-Law"):
                                    fig = plot_powerlaw_fit(
                                        lengths=data['length'].values,
                                        alpha=alpha_to_use,
                                        l_min=l_min_to_use,
                                        l_max=l_max_to_use
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                        
                        # Aplicar Pipeline
                        if st.session_state.powerlaw_params is not None:
                            st.markdown("### 🚀 Aplicar Pipeline de Correções")
                            
                            if st.button("🔧 Executar Pipeline Completo", type="primary", key="exec_pipeline"):
                                with st.spinner("Aplicando correções..."):
                                    try:
                                        pl = st.session_state.powerlaw_params
                                        corrector = BiasCorrector()
                                        
                                        #length_col = 'length_m' if 'length_m' in data.columns else 'length'
                                        #aperture_col = 'aperture_m' if 'aperture_m' in data.columns else 'aperture'
                                        
                                        # Usar parâmetros globais
                                        alpha_use = pl['alpha']
                                        l_min_use = pl['l_min']
                                        l_max_use = pl['l_max']
                                        
                                        # Debug: mostrar parâmetros sendo usados
                                        st.info(f"📊 Usando: α={alpha_use:.3f}, l_min={l_min_use:.4f}m, l_max={l_max_use:.4f}m, rock={rock_type}")
                                        
                                        results = corrector.framfrat_pipeline(
                                            data=data,
                                            area_m2=area_m2,
                                            rock_type=rock_type,
                                            alpha=alpha_use,
                                            l_min=l_min_use,
                                            l_max=l_max_use,
                                            length_col='length',
                                            aperture_col='aperture',
                                            apply_dl_validation=apply_dl,
                                            tolerance=5.0
                                        )

                                        st.session_state.bias_correction_results = results
                                        st.session_state.bias_corrections_applied = True
                                        
                                        st.markdown("")
                                        show_success("✅ Pipeline executado com sucesso!")
                                        st.markdown("")
                                        
                                    except Exception as e:
                                        st.error(f"❌ Erro ao aplicar correções: {str(e)}")
                                        st.exception(e)
                        
                        # Exibir Resultados
                        if st.session_state.bias_corrections_applied and st.session_state.bias_correction_results:
                            st.markdown("---")
                            st.markdown("## 📊 Resultados das Correções")
                            
                            results = st.session_state.bias_correction_results
                            display_correction_summary(results)
                            
                            st.markdown("### 📈 Pipeline de Correções")
                            fig_pipeline = plot_correction_pipeline(results)
                            if fig_pipeline:
                                st.plotly_chart(fig_pipeline, width='stretch')
                            
                            with st.expander("🔍  das Correções"):
                                for i, corr in enumerate(results.get('corrections', [])):
                                    st.markdown(f"#### {i+1}. {corr.method}")
                                    st.code(corr.summary())
                                    
                                    if corr.method.startswith("D-L"):
                                        fig = plot_dl_scaling_validation(corr)
                                        if fig:
                                            st.plotly_chart(fig, width='stretch')


                    # Scanline Pipeline
                    elif data_info == 'Scanline':
                        #data = st.session_state.scanline_data
                        data = current_data
                        st.info("📏 Tipo de Dado: **Scanline 1D**")
                        
                        st.subheader("⚙️ Parâmetros do Pipeline Scanline")
                        
                        default_length  = st.session_state.get('scanline_length')

                        col1, col2, col3 = st.columns([1, 1, 2], gap='medium')
                        col1,col2, col_aux = st.columns([1, 1, 3], gap='medium')
                        with col1:
                            scanline_length = st.number_input(
                                "Comprimento (m)",
                                value=default_length,  #data.attrs.get('scanline_length', 50.0),
                                help="Comprimento do scanline",
                                key="bias_length"
                            )
                        
                        with col2:
                            scanline_azimuth = st.number_input(
                                "Azimute (°)",
                                value=scanline_azimuth, #data.attrs.get('scanline_azimuth', 0.0),
                                help="Direção do scanline",
                                key="bias_azimuth"
                            )

                        #st.markdown("")
                        colV_1, colV_2, colV_3 = st.columns(3)
                        
                        with colV_1:
                            apply_terzaghi = st.checkbox(
                                "Correção de Terzaghi",
                                help="Aplicação da correção de Terzaghi nas orientações dos dados Scanline."
                            )
                            if apply_terzaghi:   # só aparece se marcado
                                with st.expander("Correção de Terzaghi", expanded=True):
                                    st.write("..Correção de Terzaghi..")

                        with colV_2:
                                apply_marrett = st.checkbox(
                                    "Correção de Marrett",
                                    help='Aplicação da correção de Marrett "truncamento" das fraturas.'
                                )
                                if apply_marrett:   # só aparece se marcado
                                    with st.expander("Correção de Marrett", expanded=True):
                                        st.write("..Correção de Marrett..")

                        with colV_3:
                            apply_dl = st.checkbox(
                                "Aplicar D-L Validation",
                                #value=True,
                                help="Remove fraturas fisicamente impossíveis."
                            )
                            if apply_dl:   # só aparece se marcado
                                with st.expander("Aplicar D-L Validation", expanded=True):
                                    st.write("..Aplicar D-L Validation...")      
                        #st.markdown("")
                        st.markdown("### 📊 Análise Power-Law Automática")
                        if st.button("🔍 Calcular Parâmetros", type="primary", key="calc_pl_scan"):
                            with st.spinner("Analisando..."):
                                try:
                                    analyzer = PowerLawAnalyzer()
                                    #length_col = 'length_m' if 'length_m' in data.columns else 'length'
                                    
                                    pl_results = analyzer.analyze_powerlaw(
                                        lengths=data['length'].values,
                                        validate=True
                                    )
                                    
                                    st.session_state.powerlaw_params = pl_results
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("α", f"{pl_results['alpha']:.3f}")
                                    with col2:
                                        st.metric("l_min", f"{pl_results['l_min']:.3f} m")
                                    with col3:
                                        st.metric("l_max", f"{pl_results['l_max']:.3f} m")
                                    with col4:
                                        st.metric("R²", f"{pl_results['r_squared']:.4f}")
                                    
                                except Exception as e:
                                    st.error(f"Erro: {str(e)}")
                        
                        if st.session_state.powerlaw_params:
                            st.markdown("### 🚀 Aplicar Pipeline")
                            
                            if st.button("🔧 Executar", type="primary", key="exec_scan"):
                                with st.spinner("Aplicando..."):
                                    try:
                                        pl = st.session_state.powerlaw_params
                                        corrector = BiasCorrector()
                                        
                                        #length_col = 'length_m' if 'length_m' in data.columns else 'length'
                                        #orientation_col = 'orientation' if 'orientation' in data.columns else None
                                        
                                        results = corrector.scanline_pipeline(
                                            data=data,
                                            scanline_azimuth=scanline_azimuth,
                                            scanline_length=scanline_length,
                                            alpha=pl['alpha'],
                                            l_min=pl['l_min'],
                                            l_max=pl['l_max'],
                                            # length_col='length',
                                            # orientation_col=orientation_col
                                        )
                                        
                                        st.session_state.bias_correction_results = results
                                        st.session_state.bias_corrections_applied = True
                                        
                                        st.markdown("")
                                        show_success("✅ Pipeline executado!")
                                        st.markdown("")
                                        
                                    except Exception as e:
                                        st.error(f"Erro: {str(e)}")
                                        st.exception(e)
                        
                        if st.session_state.bias_corrections_applied and st.session_state.bias_correction_results:
                            st.markdown("---")
                            st.markdown("## 📊 Resultados")
                            
                            results = st.session_state.bias_correction_results
                            display_correction_summary(results)
                            
                            fig = plot_correction_pipeline(results)
                            if fig:
                                st.plotly_chart(fig, width='stretch')

                #Famílias de Fraturas
                with tab_fam:
                    st.subheader("🔄 Análise de Famílias de Fraturas")

                    # CORREÇÃO: Usar função helper para obter dados
                    #current_data, data_source = get_current_fracture_data()

                    # Verificar se há dados de orientação
                    #if 'orientation' in st.session_state.framfrat_data.columns:
                    if 'orientation' in current_data.columns:    

                        #orientations = st.session_state.framfrat_data['orientation'].dropna().values
                        orientations = current_data['orientation'].dropna().values
                        
                        if len(orientations) > 10:

                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.markdown("##### ⚙️ Configuração de Famílias")
                                
                                # Opção para determinar automaticamente ou manual
                                auto_sets = st.checkbox(
                                    "Determinar número de famílias automaticamente",
                                    value=False,
                                    help="Usa método do cotovelo para determinar número ótimo de famílias"
                                )
                                
                                if auto_sets:
                                    n_sets = auto_determine_n_sets(orientations, max_sets=4)
                                    st.markdown("")
                                    show_info(f"✅ Número ótimo detectado: <strong>{n_sets} famílias/sets</strong>")
                                    st.markdown("")

                                else:
                                    n_sets = st.selectbox(
                                        "Número de famílias (sets)",
                                        options=[1, 2, 3, 4],
                                        index=1,  # Default: 2 famílias
                                        help="Número de famílias distintas de fraturas | por definição 2 famílias/sets."
                                    )
                                
                                # Clusterizar
                                fisher_params = cluster_orientations_2d(orientations, n_sets=n_sets)
                                family_stats = extract_orientation_stats(fisher_params, dimension='2d')
                                
                                # Salvar no session_state para uso posterior
                                st.session_state.fracture_families = family_stats
                                st.session_state.fisher_params = fisher_params
                                
                                st.markdown("")
                                show_success(f"✅ {len(family_stats)} famílias identificadas")
                                st.markdown("")
                                

                            with col2:
                                st.markdown("##### 📊 Estatísticas das Famílias")
                                
                                # Criar DataFrame com estatísticas
                                stats_df = pd.DataFrame([{
                                    'Família': f"Set {s['family_id'] + 1}",
                                    'Orientação Média (°)': f"{s['orientation_mean']:.1f}",
                                    'Desvio Padrão (°)': f"{s['orientation_std']:.1f}",
                                    'N° Fraturas': s['n_fractures'],
                                    'Percentual (%)': f"{s['percentage']:.1f}"
                                } for s in family_stats])
                                
                                st.dataframe(stats_df, hide_index=True, width='stretch')
                                
                                # Diagrama de roseta colorido por família
                                fig_rose = go.Figure()
                                
                                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                                
                                for i, family in enumerate(family_stats):
                                    # Filtrar orientações desta família
                                    family_mask = np.abs(
                                        (orientations - family['orientation_mean'] + 180) % 360 - 180
                                    ) < 2 * family['orientation_std']
                                    
                                    family_orients = orientations[family_mask]
                                    
                                    if len(family_orients) > 0:
                                        counts, bin_edges = np.histogram(
                                            family_orients, 
                                            bins=36, 
                                            range=(0, 360)
                                        )
                                        theta = (bin_edges[:-1] + bin_edges[1:]) / 2
                                        
                                        fig_rose.add_trace(go.Barpolar(
                                            r=counts,
                                            theta=theta,
                                            width=10,
                                            marker_color=colors[i % len(colors)],
                                            name=f'Set {i+1}',
                                            opacity=0.7
                                        ))
                                
                                dark_style = dict(
                                    paper_bgcolor="#0f1112",
                                    plot_bgcolor="#0f1112",
                                    font=dict(size=12, color="white"),
                                    polar=dict(
                                        bgcolor="#111316",
                                        angularaxis=dict(
                                            direction="clockwise",
                                            rotation=90,
                                            tickfont=dict(color="white"),
                                            gridcolor="#333333",
                                            linecolor="white",
                                            tickcolor="white"
                                        ),
                                        radialaxis=dict(visible=True,
                                            tickfont=dict(color="white"),
                                            gridcolor="#333333",
                                            linecolor="white"
                                        )
                                    ),
                                    legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.2)")
                                )

                                fig_rose.update_layout(
                                    title='Diagrama de Roseta - Famílias de Fraturas',
                                    showlegend=True,
                                    height=500,
                                    **dark_style
                                )
                                
                                st.plotly_chart(fig_rose, width='stretch')
                        
                        else:
                            st.markdown("")
                            show_info(f"⚠️ Poucos dados de orientação, <strong>{len(orientations)}</strong> disponíveis para análise de famílias.")
                            st.markdown("")
                    
                    else:
                        st.markdown("")
                        show_info("⚠️ Dados de orientação não estão disponíveis neste dataset")
                        st.markdown("")

  
    else:
        st.markdown("")
        show_info("Por favor, carregue os dados primeiro na aba <strong>Dados</strong>")
        st.markdown("")

# ============================================================================
# TAB 3: INTENSIDADE & ESPAÇAMENTO
# ============================================================================

with tab3:# Tab 4: Intensidade e Espaçamento

    st.header("📏 Análise de Intensidade e Espaçamento")
    
    if st.session_state.data_loaded:
        analyzer = IntensitySpacingAnalyzer()
        viz = FractureVisualizer()
               
        # Obter parâmetros do session_state
        if st.session_state.framfrat_data is not None:
            image_area = st.session_state.framfrat_data.attrs.get('area', 1.0)
            l_min = st.session_state.get('l_min_framfrat', 0.001)
        else:
            l_min = st.session_state.get('l_min_scanline', 0.001)
            
        if st.session_state.scanline_data is not None:
            scanline_length = st.session_state.scanline_data.attrs.get('scanline_length', 10.0)


        col1, col2 = st.columns(2)        
        with col1:
            st.subheader("Intensidade P10 (Size-Cognizant)")
            
            # Calcular intensidades para diferentes limiares
            if st.session_state.framfrat_data is not None:
                max_length_f = st.session_state.framfrat_data['length'].max()
            else:
                max_length_f = 1.0
                
            if st.session_state.scanline_data is not None:
                max_length_s = st.session_state.scanline_data['length'].max()
            else:
                max_length_s = 1.0
            
            thresholds = np.logspace(
                np.log10(l_min), 
                np.log10(max(max_length_f, max_length_s)), 
                50
            )
            
            intensities_framfrat = []
            intensities_scanline = []
            
            for threshold in thresholds:
                if st.session_state.framfrat_data is not None:
                    p10_f = analyzer.calculate_p10(
                        st.session_state.framfrat_data,
                        threshold,
                        image_area
                    )
                    intensities_framfrat.append(p10_f)
                
                if st.session_state.scanline_data is not None:
                    p10_s = analyzer.calculate_p10_scanline(
                        st.session_state.scanline_data,
                        threshold,
                        scanline_length
                    )
                    intensities_scanline.append(p10_s)
            
            # Plotar curva de intensidade
            fig_intensity = go.Figure()
            
            if intensities_framfrat:
                fig_intensity.add_trace(go.Scatter(
                    x=thresholds,
                    y=intensities_framfrat,
                    mode='lines',
                    name='FRAMFRAT',
                    line=dict(color='blue', width=2)
                ))
            
            if intensities_scanline:
                fig_intensity.add_trace(go.Scatter(
                    x=thresholds,
                    y=intensities_scanline,
                    mode='lines',
                    name='Scanline',
                    line=dict(color='red', width=2)
                ))
            
            fig_intensity.update_layout(
                title="Intensidade vs Limiar de Tamanho",
                xaxis_title="Limiar de comprimento (m)",
                yaxis_title="P10 (fraturas/m)",
                xaxis_type="log",
                yaxis_type="log",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_intensity, width='stretch')
        
        with col2:
            st.subheader("Espaçamento Médio")
            
            # Calcular espaçamentos
            spacings_framfrat = [1/i if i > 0 else np.nan for i in intensities_framfrat]
            spacings_scanline = [1/i if i > 0 else np.nan for i in intensities_scanline]
            
            # Plotar curva de espaçamento
            fig_spacing = go.Figure()
            
            if spacings_framfrat:
                fig_spacing.add_trace(go.Scatter(
                    x=thresholds,
                    y=spacings_framfrat,
                    mode='lines',
                    name='FRAMFRAT',
                    line=dict(color='blue', width=2)
                ))
            
            if spacings_scanline:
                fig_spacing.add_trace(go.Scatter(
                    x=thresholds,
                    y=spacings_scanline,
                    mode='lines',
                    name='Scanline',
                    line=dict(color='red', width=2)
                ))
            
            fig_spacing.update_layout(
                title="Espaçamento vs Limiar de Tamanho",
                xaxis_title="Limiar de comprimento (m)",
                yaxis_title="Espaçamento médio (m)",
                xaxis_type="log",
                yaxis_type="log",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_spacing, width='stretch')
        
        # Comparação normalizada
        st.divider()
        st.subheader("📋 Comparação Normalizada")
        
        # Obter o máximo apropriado
        if st.session_state.framfrat_data is not None:
            max_for_slider = st.session_state.framfrat_data['length'].quantile(0.5)
        elif st.session_state.scanline_data is not None:
            max_for_slider = st.session_state.scanline_data['length'].quantile(0.5)
        else:
            max_for_slider = 1.0
        
        # Selecionar limiar comum
        common_threshold = st.slider(
            "Limiar comum de tamanho (m)",
            min_value=float(l_min),
            max_value=float(max_for_slider),
            value=float(min(l_min * 10, max_for_slider)),
            format="%.4f"
        )
        
        col1, col2, col3 = st.columns(3)
        
        if st.session_state.framfrat_data is not None:
            p10_f_common = analyzer.calculate_p10(
                st.session_state.framfrat_data,
                common_threshold,
                image_area
            )
            with col1:
                st.metric(
                    "P10 FRAMFRAT",
                    f"{p10_f_common:.3f} fraturas/m",
                    f"Espaçamento: {1/p10_f_common:.3f} m"
                )
        
        if st.session_state.scanline_data is not None:
            p10_s_common = analyzer.calculate_p10_scanline(
                st.session_state.scanline_data,
                common_threshold,
                scanline_length
            )
            with col2:
                st.metric(
                    "P10 Scanline",
                    f"{p10_s_common:.3f} fraturas/m",
                    f"Espaçamento: {1/p10_s_common:.3f} m"
                )


        if st.session_state.framfrat_data is not None and st.session_state.scanline_data is not None:
            ratio = p10_f_common / p10_s_common
            with col3:
                st.metric(
                    "Razão FRAMFRAT/Scanline",
                    f"{ratio:.2f}",
                    "Fator de intensificação" if ratio > 1 else "Fator de redução"
                )
    else:
        #st.info("📁 Por favor, carregue os dados primeiro")
        st.markdown("")
        show_info("📁 Por favor, carregue os dados primeiro")
        st.markdown("")


# Tab 5: Geração DFN  - Com Visualização Reativa
with tab4:
    
    st.header("📐 Gerador de DFN")

    tab_DFN2D, tab_DFN3D = st.tabs(['DFN 2D', 'DFN 3D'])

    with tab_DFN2D:
        
        st.header("🗺️ Geração de DFN 2D")
        if st.session_state.data_loaded and st.session_state.analysis_results:
            
            # Obter área da imagem e parâmetros básicos
            if st.session_state.framfrat_data is not None:
                image_area = st.session_state.framfrat_data.attrs.get('area', 1.0)
                l_min = st.session_state.get('l_min_framfrat', 0.001)
            elif st.session_state.scanline_data is not None:
                # Para Scanline, usar o comprimento mínimo real dos dados
                image_area = 1.0  # Área padrão para Scanline
                scanline_lengths = st.session_state.scanline_data['length'].values
                l_min = max(float(np.min(scanline_lengths[scanline_lengths > 0])), 0.1)  # Mínimo de 10cm
            else:
                image_area = 1.0
                l_min = 0.1  # Default razoável: 10cm
            
            # Calcular intensi_fract apenas se não estiver em cache ou dados mudaram
            if st.session_state.intensi_fract_cached is None:
            
                analyzer = IntensitySpacingAnalyzer()
                
                if st.session_state.framfrat_data is not None:
                    intensi_fract = analyzer.calculate_from_framfrat(st.session_state.framfrat_data, image_area)
                    inten_frat_3D_gen = analyzer.convert_2d_to_3d()
                    st.session_state.intensi_fract_cached = intensi_fract
                elif st.session_state.scanline_data is not None:
                    # Calcular intensidades a partir de dados de Scanline
                    scanline_data = st.session_state.scanline_data
                    scanline_length_m = scanline_data.attrs.get('scanline_length', 10.0)
                    n_fractures = len(scanline_data)
                    mean_length_m = float(np.mean(scanline_data['length'].values))
                    mean_aperture_m = float(np.mean(scanline_data['aperture'].values))
                    
                    intensi_fract = analyzer.calculate_from_scanline(
                        n_fractures=n_fractures,
                        scanline_length_m=scanline_length_m,
                        mean_length_m=mean_length_m,
                        mean_aperture_m=mean_aperture_m
                    )
                    # Converter para dicionário para compatibilidade
                    intensi_fract = intensi_fract.to_dict() if hasattr(intensi_fract, 'to_dict') else intensi_fract
                    st.session_state.intensi_fract_cached = intensi_fract
                else:
                    intensi_fract = None
            else:
                intensi_fract = st.session_state.intensi_fract_cached
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Configurações DFN 2D")
                
                # Semente aleatória
                random_seed_2d = st.number_input(
                    "🎲 Semente aleatória", 
                    min_value=0, 
                    value=42,
                    help="Para reprodutibilidade da geração",
                    key="seed_2d"
                )
                
                st.divider()
                
                # Domínio
                domain_width = st.number_input(
                    "Largura do domínio (m)",
                    min_value=1.0,
                    value=10.0,  # Domínio padrão: 10m
                    step=1.0
                )
                
                domain_height = st.number_input(
                    "Altura do domínio (m)",
                    min_value=1.0,
                    value=10.0,  # Domínio padrão: 10m
                    step=1.0
                )
                
                # Usar famílias identificadas
                use_families = st.checkbox(
                    "Usar famílias identificadas",
                    value=True if hasattr(st.session_state, 'fracture_families') else False,
                    help="Gerar fraturas respeitando as famílias identificadas na análise"
                )
                            
                # Usar parâmetros ajustados
                use_fitted = st.checkbox(
                    "Usar parâmetros ajustados",
                    value=True,
                    help="Usa os parâmetros das leis de potência ajustadas"
                )
                
                # Controles de Visualização
                st.divider()
                st.subheader("Controles de Visualização")
                
                show_centers_2d = st.checkbox(
                    "Mostrar Centros das Fraturas",
                    value=False,
                    help="Exibe o ponto central de cada fratura"
                )
                
                show_numbers_2d = st.checkbox(
                    "Mostrar Numeração das Fraturas",
                    value=False,
                    help="Exibe o número de cada fratura"
                )
                
                # Detectar mudanças nos parâmetros de geração
                current_gen_params = {
                    'seed': random_seed_2d,
                    'domain_width': domain_width,
                    'domain_height': domain_height,
                    'use_families': use_families,
                    'use_fitted': use_fitted,
                    'l_min': l_min
                }
                
                params_changed = (
                    st.session_state.dfn_2d_params != current_gen_params
                    if st.session_state.dfn_2d_params else True
                )
                
                # Mostrar indicador se parâmetros mudaram
                if params_changed and st.session_state.dfn_2d_generated:
                    show_info("⚠️ Parâmetros alterados. Clique em <strong>🎲 Gerar DFN 2D</strong> para recalcular.", bg= "#132638", border= "#2c7be5" )
                    st.markdown("")

                # Botão de gerar
                generate_2d = st.button(
                    "🎲 Gerar DFN 2D",
                    type="primary",
                    width='stretch'
                )
            
            with col2:
                # Função para renderizar visualização (sempre executa)
                def render_current_visualization():
                    if st.session_state.dfn_2d_data is not None:
                        viz = FractureVisualizer()
                        
                        # Criar visualização com controles atuais
                        fig_dfn = viz.plot_dfn_2d(
                            st.session_state.dfn_2d_data,
                            (domain_width, domain_height),
                            show_centers=show_centers_2d,
                            show_numbers=show_numbers_2d,
                            color_by_family=use_families
                        )
                        
                        st.plotly_chart(fig_dfn, width='stretch')
                        
                        # Mostrar estatísticas salvas (não recalcula!)
                        if st.session_state.dfn_2d_stats:
                            st.divider()
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Total de fraturas", 
                                        st.session_state.dfn_2d_stats['total_fractures'])
                                st.metric("Comprimento total (m)", 
                                        f"{st.session_state.dfn_2d_stats['total_length']:.2f}".replace(".", ","))
                            
                            with col2:
                                st.metric("Comprimento médio (m)", 
                                        f"{st.session_state.dfn_2d_stats['mean_length']:.3f}".replace(".", ","))
                                st.metric("Abertura média (mm)", 
                                        f"{st.session_state.dfn_2d_stats['mean_aperture'] * 1000:.2f}".replace(".", ","))
                            
                            with col3:
                                st.metric("P21 (m/m²)", 
                                        f"{st.session_state.dfn_2d_stats['P21']:.4f}".replace(".", ","))
                                st.metric("Porosidade (%)", 
                                        f"{st.session_state.dfn_2d_stats['porosity']:.3f}".replace(".", ","))
                
                # Gerar novo DFN apenas se necessário
                if generate_2d and (not st.session_state.dfn_2d_generated or params_changed):
                    with st.spinner("Gerando DFN 2D..."):
                        # Usar a semente específica desta aba
                        generator = DFNGenerator(random_seed_2d)
                        
                        # Preparar famílias se usar
                        families = None
                        if use_families and hasattr(st.session_state, 'fracture_families'):
                            from modules.dfn_generator import FractureFamily
                            families = [
                                FractureFamily(
                                    orientation_mean=f['orientation_mean'],
                                    orientation_std=f['orientation_std'],
                                    weight=f['percentage'] / 100
                                )
                                for f in st.session_state.fracture_families
                            ]
                        
                        # Preparar parâmetros
                        if use_fitted and 'length_fit' in st.session_state.analysis_results:
                            # IMPORTANTE: Usar x_min do ajuste power-law, não o filtro de resolução
                            # O x_min do ajuste representa o comprimento mínimo real das fraturas
                            x_min_fit = st.session_state.analysis_results['length_fit'].get('x_min', l_min)
                            # Garantir que x_min seja razoável (pelo menos 1% da largura do domínio)
                            x_min_safe = max(x_min_fit, domain_width * 0.01)
                            
                            params = {
                                'exponent': st.session_state.analysis_results['length_fit']['exponent'],
                                'x_min': x_min_safe,
                                'coefficient': st.session_state.analysis_results['length_fit']['coefficient'],
                            }
                            
                            # Adicionar parâmetros de abertura se disponíveis
                            if 'bl_relation' in st.session_state.analysis_results:
                                params['g'] = st.session_state.analysis_results['bl_relation']['g']
                                params['m'] = st.session_state.analysis_results['bl_relation']['m']
                            
                            # Adicionar orientação se disponível (verificar se framfrat_data existe)
                            if (st.session_state.framfrat_data is not None and 
                                'orientation' in st.session_state.framfrat_data.columns):
                                orientations = st.session_state.framfrat_data['orientation'].values
                                params['orientation_mean'] = np.mean(orientations)
                                params['orientation_std'] = np.std(orientations)
                        else:
                            params = {
                                'exponent': 2.0,
                                'x_min': 10.0,
                                'coefficient': 100
                            }
                        
                        # Gerar DFN
                        dfn_2d = generator.generate_2d_dfn(
                            params=params,
                            domain_size=(domain_width, domain_height),
                            families=families,
                            intensi_fract=intensi_fract
                        )
                        
                        # Converter para DataFrame e calcular estatísticas
                        dfn_df = pd.DataFrame([f.to_dict() for f in dfn_2d])
                        
                        # SALVAR DADOS E ESTATÍSTICAS NO SESSION_STATE
                        st.session_state.dfn_2d_data = dfn_2d
                        st.session_state.dfn_2d_generated = True
                        st.session_state.dfn_2d_params = current_gen_params
                        
                        # Calcular e salvar estatísticas UMA VEZ
                        st.session_state.dfn_2d_stats = {
                            'total_fractures': len(dfn_2d),
                            'total_length': dfn_df['length'].sum(),
                            'mean_length': dfn_df['length'].mean(),
                            'mean_aperture': dfn_df['aperture'].mean(),
                            'P21': dfn_df['length'].sum() / (domain_width * domain_height),
                            'porosity': (dfn_df['aperture'] * dfn_df['length']).sum() / 
                                    (domain_width * domain_height) * 100
                        }
                        
                        show_success("✅ DFN 2D gerado com sucesso!")
                
                # SEMPRE renderizar visualização se houver dados
                # Isso permite atualização automática quando controles mudam
                if st.session_state.dfn_2d_generated:
                    render_current_visualization()
        else:
            show_info("📁 Por favor, complete as análises anteriores primeiro") 



    # Tab 6: DFN 3D
    with tab_DFN3D:

        st.header("🎲 Geração de DFN 3D")
        if st.session_state.data_loaded and st.session_state.analysis_results:
            # Obter l_min baseado no tipo de dados carregados
            if st.session_state.framfrat_data is not None:
                l_min = st.session_state.get('l_min_framfrat', 0.1)
            elif st.session_state.scanline_data is not None:
                # Para Scanline, usar o comprimento mínimo real dos dados
                scanline_lengths = st.session_state.scanline_data['length'].values
                l_min = max(float(np.min(scanline_lengths[scanline_lengths > 0])), 0.1)
            else:
                l_min = 0.1  # Default razoável
            
            st.subheader("Configurações DFN 3D")
            col1, col2, col3 = st.columns(3) # DOMÍNIO 3D
            # Garantir que dfn_3d_domain nunca seja None
            _dfn_domain = st.session_state.get('dfn_3d_domain') or [50.0, 50.0, 20.0]
            domain_x = col1.number_input("Dimensão X (m)", min_value=5.0, value=float(_dfn_domain[0]), step=5.0)
            domain_y = col2.number_input("Dimensão Y (m)", min_value=5.0, value=float(_dfn_domain[1]), step=5.0)    
            domain_z = col3.number_input("Dimensão Z (m)", min_value=5.0, value=float(_dfn_domain[2]), step=5.0)

            col_L, col_R = st.columns([1, 1], gap='large')

            with col_L:
                # Orientação preferencial
                st.divider()
                st.write("**Orientação Preferencial /Set 1**")
                col_left, col_mid= st.columns([1, 1], gap='large')
                dip_mean = col_left.slider("Dip médio (°)", min_value=0, max_value=90, value=45, help="Ângulo de mergulho médio do Set 1")
                dip_dir_mean = col_mid.slider("Dip Direction médio (°)", min_value=0, max_value=360, value=90, help="Direção de mergulho média do Set 1")

            with col_R:
                st.divider()
                st.write("**Mais configurações**")
                col_left, col_mid = st.columns([1, 1], gap='medium')

                # Semente aleatória
                random_seed_3d = col_left.number_input(
                    "🎲 Semente aleatória", 
                    min_value=0, 
                    value=42,
                    help="Para reprodutibilidade da geração",
                    key="seed_3d"
                )
                
                # Fator de correção de abertura
                aperture_correction = col_mid.number_input(
                    "🔧 Fator correção abertura",
                    min_value=0.01,
                    max_value=1.0,
                    value=1.0,
                    step=0.05,
                    format="%.2f",
                    help="Divide a abertura por este fator (1.0 = sem correção, 0.1 = divide por 10)",
                    key="aperture_correction_3d"
                )
                
                if aperture_correction < 1.0:
                    st.info(f"ℹ️ Abertura será multiplicada por {aperture_correction:.2f} (reduzida em {(1-aperture_correction)*100:.0f}%)")
                                
            st.divider()
                    # ========== CONFIGURAÇÃO DE FAMÍLIAS 3D ==========
            st.subheader("🔄 Famílias de Fraturas 3D")
            
            # ========== CONFIGURAÇÃO INTEGRADA DE FAMÍLIAS ==========
            family_mode = st.radio(
                "Modo de configuração",
                options=["🔍 Automático", "⚡ Manual Simples", "⚙️ Manual Avançado"],
                index=1,  # Default: Manual Simples
                horizontal=True,
                help="Automático: detecta dos dados | Simples: número e pesos | Avançado: controle total"
            )
            
            family_configs = []
            use_families_3d = True
            
            # Inicializar no session_state se não existir
            if 'use_families_3d' not in st.session_state:
                st.session_state.use_families_3d = True
            
            # ==================== MODO AUTOMÁTICO ====================
            if family_mode == "🔍 Automático":
                st.markdown("")
                
                # PRIORIDADE 1: Usar famílias já calculadas na Tab Power-Law
                if hasattr(st.session_state, 'fracture_families') and st.session_state.fracture_families:
                    st.success("✅ Usando famílias da análise Power-Law")
                    
                    for stat in st.session_state.fracture_families:
                        family_configs.append({
                            'dip': dip_mean,
                            'dip_dir': stat['orientation_mean'],
                            'weight': stat['percentage'] / 100.0,
                            'dip_std': 10.0,
                            'dip_dir_std': stat.get('orientation_std', 15.0)
                        })
                    
                    # Mostrar tabela resumo
                    summary_data = []
                    for i, fc in enumerate(family_configs):
                        summary_data.append({
                            'Família': f'Set {i+1}',
                            'Dip (°)': f"{fc['dip']:.0f}",
                            'Dip Dir (°)': f"{fc['dip_dir']:.0f}",
                            'Peso': f"{fc['weight']*100:.1f}%"
                        })
                    
                    st.dataframe(
                        pd.DataFrame(summary_data), 
                        hide_index=True, 
                        width='stretch'
                    )
                
                # PRIORIDADE 2: Detectar famílias dos dados FRAMFRAT
                else:
                    has_orientation_data = (
                        st.session_state.framfrat_data is not None and 
                        'orientation' in st.session_state.framfrat_data.columns and
                        st.session_state.framfrat_data['orientation'].notna().sum() > 10
                    )
                    
                    if has_orientation_data:
                        col_auto1, col_auto2 = st.columns([1, 2])
                        
                        with col_auto1:
                            n_families_auto = st.slider(
                                "Famílias a detectar",
                                min_value=1, max_value=4, value=2,
                                help="Número de grupos de orientação a identificar nos dados"
                            )
                        
                        with col_auto2:
                            try:
                                from orientation_clustering import cluster_orientations_2d, extract_orientation_stats
                                
                                orientations = st.session_state.framfrat_data['orientation'].dropna().values
                                
                                fisher_params = cluster_orientations_2d(orientations, n_sets=n_families_auto)
                                family_stats = extract_orientation_stats(fisher_params, dimension='2d')
                                
                                for stat in family_stats:
                                    family_configs.append({
                                        'dip': dip_mean,
                                        'dip_dir': stat['orientation_mean'],
                                        'weight': stat['percentage'] / 100.0,
                                        'dip_std': 10.0,
                                        'dip_dir_std': stat.get('orientation_std', 15.0)
                                    })
                                
                                st.success(f"✅ {len(family_configs)} famílias detectadas")
                                
                            except Exception as e:
                                st.warning(f"⚠️ Erro na detecção: {str(e)[:50]}...")
                                family_configs = [
                                    {'dip': dip_mean, 'dip_dir': dip_dir_mean, 'weight': 0.5, 'dip_std': 10.0, 'dip_dir_std': 20.0},
                                    {'dip': dip_mean, 'dip_dir': (dip_dir_mean + 90) % 360, 'weight': 0.5, 'dip_std': 10.0, 'dip_dir_std': 20.0}
                                ]
                        
                        if family_configs:
                            summary_data = []
                            for i, fc in enumerate(family_configs):
                                summary_data.append({
                                    'Família': f'Set {i+1}',
                                    'Dip (°)': f"{fc['dip']:.0f}",
                                    'Dip Dir (°)': f"{fc['dip_dir']:.0f}",
                                    'Peso': f"{fc['weight']*100:.1f}%"
                                })
                            
                            st.dataframe(
                                pd.DataFrame(summary_data), 
                                hide_index=True, 
                                width='stretch'
                            )
                    
                    else:
                        st.warning("⚠️ Analise as famílias na Tab 'Power-Law' primeiro, ou use o modo Manual.")
                        
                        family_configs = [
                            {'dip': dip_mean, 'dip_dir': dip_dir_mean, 'weight': 0.5, 'dip_std': 10.0, 'dip_dir_std': 20.0},
                            {'dip': dip_mean, 'dip_dir': (dip_dir_mean + 90) % 360, 'weight': 0.5, 'dip_std': 10.0, 'dip_dir_std': 20.0}
                        ]
                        
                        st.info("ℹ️ Usando configuração padrão: 2 famílias conjugadas (90° entre elas)")
            
            # ==================== MODO MANUAL SIMPLES ====================
            elif family_mode == "⚡ Manual Simples":
                st.markdown("")
                
                col_simple1, col_simple2 = st.columns(2)
                
                with col_simple1:
                    n_families_simple = st.selectbox(
                        "Número de famílias",
                        options=[1, 2, 3, 4],
                        index=1,  # Default: 2
                        key="n_families_simple"
                    )
                
                with col_simple2:
                    # Opções pré-definidas de distribuição de pesos
                    if n_families_simple == 1:
                        weight_options = {"Única (100%)": [100]}
                    elif n_families_simple == 2:
                        weight_options = {
                            "Igual (50/50)": [50, 50],
                            "Dominante (70/30)": [70, 30],
                            "Muito dominante (80/20)": [80, 20]
                        }
                    elif n_families_simple == 3:
                        weight_options = {
                            "Igual (33/33/34)": [33, 33, 34],
                            "Uma dominante (50/30/20)": [50, 30, 20],
                            "Duas dominantes (40/40/20)": [40, 40, 20]
                        }
                    else:  # 4 famílias
                        weight_options = {
                            "Igual (25/25/25/25)": [25, 25, 25, 25],
                            "Duas dominantes (35/35/15/15)": [35, 35, 15, 15],
                            "Uma dominante (40/25/20/15)": [40, 25, 20, 15]
                        }
                    
                    weight_choice = st.selectbox(
                        "Distribuição de pesos",
                        options=list(weight_options.keys()),
                        index=0,
                        key="weight_distribution_simple"
                    )
                    
                    selected_weights = weight_options[weight_choice]
                
                # Calcular orientações automaticamente (fraturas conjugadas)
                if n_families_simple == 1:
                    angle_offsets = [0]
                elif n_families_simple == 2:
                    angle_offsets = [0, 90]
                elif n_families_simple == 3:
                    angle_offsets = [0, 60, 120]
                else:
                    angle_offsets = [0, 45, 90, 135]
                
                for i in range(n_families_simple):
                    family_configs.append({
                        'dip': dip_mean,
                        'dip_dir': (dip_dir_mean + angle_offsets[i]) % 360,
                        'weight': selected_weights[i] / 100.0,
                        'dip_std': 10.0,
                        'dip_dir_std': 20.0
                    })
                
                # Mostrar tabela resumo
                st.markdown("")
                summary_data = []
                for i, fc in enumerate(family_configs):
                    summary_data.append({
                        'Família': f'Set {i+1}',
                        'Dip (°)': f"{fc['dip']:.0f}",
                        'Dip Dir (°)': f"{fc['dip_dir']:.0f}",
                        'Peso': f"{fc['weight']*100:.0f}%"
                    })
                
                st.dataframe(
                    pd.DataFrame(summary_data), 
                    hide_index=True, 
                    width='stretch'
                )
                
                st.caption("💡 *Orientações distribuídas automaticamente como fraturas conjugadas*")
            
            # ==================== MODO MANUAL AVANÇADO ====================
            else:  # Manual Avançado
                st.markdown("")
                
                col_adv1, col_adv2 = st.columns([1, 3])
                
                with col_adv1:
                    n_families_adv = st.selectbox(
                        "Número de famílias",
                        options=[1, 2, 3, 4],
                        index=1,
                        key="n_families_adv"
                    )
                
                with col_adv2:
                    st.markdown("**Configuração detalhada:**")
                
                # Configuração individual de cada família
                for i in range(n_families_adv):
                    with st.expander(f"Set {i+1} - Configuração", expanded=(i == 0)):
                        col_a, col_b, col_c = st.columns(3)
                        
                        # Valores default distribuídos
                        if n_families_adv == 1:
                            default_dip_dir = dip_dir_mean
                        elif n_families_adv == 2:
                            default_dip_dir = dip_dir_mean + (i * 90)
                        else:
                            default_dip_dir = dip_dir_mean + (i * 360 // n_families_adv)
                        
                        default_weight = 100 // n_families_adv
                        
                        family_dip = col_a.number_input(
                            "Dip (°)", 
                            min_value=0,
                            max_value=90,
                            value=dip_mean,
                            step=5,
                            key=f"family_adv_{i}_dip"
                        )
                        
                        family_dip_dir = col_b.number_input(
                            "Dip Dir (°)",
                            min_value=0,
                            max_value=360,
                            value=int(default_dip_dir % 360),
                            step=10,
                            key=f"family_adv_{i}_dip_dir"
                        )
                        
                        family_weight = col_c.number_input(
                            "Peso (%)",
                            min_value=5,
                            max_value=100,
                            value=default_weight,
                            step=5,
                            key=f"family_adv_{i}_weight",
                            help="Percentual de fraturas desta família"
                        )
                        
                        family_configs.append({
                            'dip': family_dip,
                            'dip_dir': family_dip_dir,
                            'weight': family_weight / 100.0,
                            'dip_std': 10.0,
                            'dip_dir_std': 20.0
                        })
                
                # Normalizar pesos
                total_weight = sum(f['weight'] for f in family_configs)
                if total_weight > 0:
                    for f in family_configs:
                        f['weight'] = f['weight'] / total_weight
            
            # ========== SALVAR CONFIGURAÇÃO NO SESSION STATE ==========
            st.session_state.family_configs_3d = family_configs
            use_families_3d = len(family_configs) > 1
            st.session_state.use_families_3d = use_families_3d  # Salvar no session_state
            
            # Informação sobre famílias configuradas
            if len(family_configs) > 1:
                total_str = " + ".join([f"{int(fc['weight']*100)}%" for fc in family_configs])
                st.markdown("")
                show_info(f"✓ Configurado: <strong>{len(family_configs)} famílias</strong> ({total_str})")
            
            st.divider()
            # ========== CONTROLES DE VISUALIZAÇÃO ==========
            st.subheader('🎛️ Controles de Visualização')
            
            # Inicializar estado se não existir
            if 'viz_mode' not in st.session_state:
                st.session_state.viz_mode = 'ellipsoids'
            if 'show_centers_3d' not in st.session_state:
                st.session_state.show_centers_3d = False
            if 'show_numbers_3d' not in st.session_state:
                st.session_state.show_numbers_3d = False
                
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**Tipo de visualização das fraturas**")
                viz_options = { #Radio Buttons para selação do modo de visualização das fraturas
                    'lines': '📈 Linhas',
                    'ellipsoids': '⭕ Elipsóides'
                }
                
                viz_mode = st.radio(
                    "Tipo de visualização",
                    options=list(viz_options.keys()),
                    format_func=lambda x: viz_options[x],
                    index=list(viz_options.keys()).index(st.session_state.viz_mode),
                    key='viz_mode_radio',
                    label_visibility='collapsed'
                )
                
                # Atualizar estado quando mudar algum parametro
                if viz_mode != st.session_state.viz_mode:
                    st.session_state.viz_mode = viz_mode
            
            with col2:
                show_numbers = st.checkbox(
                    '🔢 Numeração das Fraturas',
                    value=st.session_state.show_numbers_3d,
                    help='Numerar as fraturas',
                    key='show_numbers_checkbox'
                )
                if show_numbers != st.session_state.show_numbers_3d:
                    st.session_state.show_numbers_3d = show_numbers
                
                # ACTION: Checkbox para centros
                show_centers = st.checkbox(
                    '🎯 Centros das Fraturas',
                    value=st.session_state.show_centers_3d,
                    help='Mostrar centros das fraturas',
                    key='show_centers_checkbox'
                )
                if show_centers != st.session_state.show_centers_3d:
                    st.session_state.show_centers_3d = show_centers
            
            with col3:
                color_by_family_3d = st.checkbox(
                    '🎨 Colorir por Família',
                    value=use_families_3d,
                    help='Colorir fraturas de acordo com sua família',
                    key='color_by_family_3d'
                )

            st.markdown("")
            st.markdown("")
            
            # Botão de gerar
            col_esq, col_dir = st.columns([1, 4], gap='large')
            
            generate_3d = col_esq.button("🎲 Gerar DFN 3D", type="primary", key='btn_generate_3d')
            
            # ========== LÓGICA DE GERAÇÃO ==========
            if generate_3d:
                with st.spinner("Gerando DFN 3D..."):
                    generator = DFNGenerator(random_seed_3d)

                    # Preparar famílias
                    families_3d = None
                    # Usar use_families_3d do session_state para garantir persistência
                    use_families_3d_current = st.session_state.get('use_families_3d', False)
                    if use_families_3d_current and hasattr(st.session_state, 'family_configs_3d') and st.session_state.family_configs_3d:
                        from modules.dfn_generator import FractureFamily
                        
                        families_3d = []
                        print(f"DEBUG - Criando famílias com pesos:")
                        for i, config in enumerate(st.session_state.family_configs_3d):
                            print(f"  Família {i}: weight={config['weight']:.4f} ({config['weight']*100:.1f}%)")
                            families_3d.append(FractureFamily(
                                orientation_mean=config['dip'],
                                orientation_std=config.get('dip_std', 10.0),
                                dip_dir_mean=config['dip_dir'],
                                dip_dir_std=config.get('dip_dir_std', 20.0),
                                weight=config['weight']
                            ))
                        print(f"DEBUG - Total de famílias criadas: {len(families_3d)}")
                    
                    # Preparar parâmetros
                    if 'length_fit' in st.session_state.analysis_results:
                        # IMPORTANTE: Usar x_min do ajuste power-law, não o filtro de resolução
                        x_min_fit = st.session_state.analysis_results['length_fit'].get('x_min', l_min)
                        # Garantir que x_min seja razoável (pelo menos 1% da menor dimensão do domínio)
                        min_domain_dim = min(domain_x, domain_y, domain_z)
                        x_min_safe = max(x_min_fit, min_domain_dim * 0.01)
                        
                        params_3d = {
                            'exponent': st.session_state.analysis_results['length_fit']['exponent'],
                            'x_min': x_min_safe,
                            'coefficient': st.session_state.analysis_results['length_fit']['coefficient'],
                            'dip_mean': dip_mean,
                            'dip_std': 10,
                            'dip_dir_mean': dip_dir_mean,
                            'dip_dir_std': 20
                        }
                        
                        if 'bl_relation' in st.session_state.analysis_results:
                            params_3d['g'] = st.session_state.analysis_results['bl_relation']['g']
                            params_3d['m'] = st.session_state.analysis_results['bl_relation']['m']
                    else:
                        params_3d = {
                            'exponent': 2.0,
                            'x_min': 10.0, #mm #0.01,
                            'coefficient': 100,
                            'dip_mean': dip_mean,
                            'dip_dir_mean': dip_dir_mean
                        }
                    
                    intensity_spacy_insta = IntensitySpacingAnalyzer()
    
                    # Verificar se há dados do FRAMFRAT ou Scanline para calcular intensidades
                    if st.session_state.framfrat_data is not None:
                        # Obter área da imagem
                        image_area = st.session_state.framfrat_data.attrs.get('area', 1.0)
                        
                        # Calcular índices 2D (P20, P21, P22)
                        intensi_fract_2d = intensity_spacy_insta.calculate_from_framfrat(
                            st.session_state.framfrat_data, 
                            image_area
                        )
                        
                        # Converter para 3D (P30, P31, P32, P33)
                        inten_frat_3D_gen = intensity_spacy_insta.convert_2d_to_3d()
                        
                    elif st.session_state.scanline_data is not None:
                        # Calcular intensidades a partir de dados de Scanline
                        scanline_data = st.session_state.scanline_data
                        scanline_length_m = scanline_data.attrs.get('scanline_length', 10.0)
                        n_fractures = len(scanline_data)
                        mean_length_m = float(np.mean(scanline_data['length'].values))
                        mean_aperture_m = float(np.mean(scanline_data['aperture'].values))
                        
                        # Calcular índices 1D (P10, P11)
                        intensi_fract_1d = intensity_spacy_insta.calculate_from_scanline(
                            n_fractures=n_fractures,
                            scanline_length_m=scanline_length_m,
                            mean_length_m=mean_length_m,
                            mean_aperture_m=mean_aperture_m
                        )
                        
                        # Converter para 3D (P30, P31, P32, P33)
                        inten_frat_3D_gen = intensity_spacy_insta.convert_1d_to_3d()
                        
                    else:
                        st.error("❌ Dados não encontrados. Por favor, carregue os dados na aba 'Dados' primeiro.")
                        st.stop()

                    #generator = DFNGenerator(seed=42)
                    dfn_3d = generator.generate_3d_dfn(
                        params=params_3d, #{'exponent': 2.5, 'x_min': 0.1, 'x_max': 100},
                        domain_size=(domain_x, domain_y, domain_z), #(1000, 1000, 1000),  # mm
                        families=families_3d, 
                        intensi_fract=inten_frat_3D_gen,
                    )

                    # Criar DataFrame e verificar estrutura
                    dfn_3d_df = pd.DataFrame([f.to_dict() for f in dfn_3d])
                    
                    # Debug: verificar colunas criadas
                    print(f"DFN 3D DataFrame criado com colunas: {dfn_3d_df.columns.tolist()}")
                    print(f"Número de fraturas geradas: {len(dfn_3d_df)}")
                                    
                    # Salvar no estado
                    st.session_state.dfn_3d = dfn_3d
                    st.session_state.dfn_3d_df = dfn_3d_df
                    st.session_state.dfn_3d_domain = (domain_x, domain_y, domain_z)
                    
                    st.divider()
                    st.markdown("")
                    show_success("✅ DFN 3D gerado com sucesso!")
                    st.markdown("")

            # ========== FUNÇÃO DE RENDERIZAÇÃO REATIVA ==========
            def render_current_view():
                """
                ACTION: Renderiza a visualização 3D com base no estado atual.
                Chamada automaticamente quando widgets mudam.
                """
                if 'dfn_3d_df' not in st.session_state or st.session_state.dfn_3d_df is None:
                    st.markdown("")
                    show_info("⚠️ Clique no botão 'Gerar DFN 3D' para visualizar o gráfico.")
                    st.markdown("")
                    return
                
                viz = FractureVisualizer()
                domain_size = st.session_state.dfn_3d_domain
                
                # Obter metadados da geração
                dfn_metadata = st.session_state.get('dfn_3d_metadata', {})
                n_fractures_theoretical = dfn_metadata.get('n_fractures_calculated', len(st.session_state.dfn_3d_df))
                
                # Obter pesos das famílias
                family_weights = {}
                if hasattr(st.session_state, 'family_configs_3d') and st.session_state.family_configs_3d:
                    for i, fc in enumerate(st.session_state.family_configs_3d):
                        family_weights[i] = fc.get('weight', 1.0 / len(st.session_state.family_configs_3d))
                
                with st.spinner("Atualizando visualização DFN 3D..."):
                    # Chamar plot_dfn_3d com parâmetros do estado
                    fig_dfn_3d = viz.plot_dfn_3d(
                        fractures_df=st.session_state.dfn_3d_df,
                        domain_size=domain_size,
                        shape_mode=st.session_state.viz_mode,
                        show_centers=st.session_state.show_centers_3d,
                        show_numbers=st.session_state.show_numbers_3d,
                        color_by_family=st.session_state.get('color_by_family_3d', False),
                        family_col='family',
                        n_fractures_theoretical=n_fractures_theoretical,
                        family_weights=family_weights
                    )
                    
                    st.plotly_chart(fig_dfn_3d, width='stretch')
                    
                    # ========== ESTATÍSTICAS ==========
                    dfn_3d_df = st.session_state.dfn_3d_df
                    
                    # Verificar se o DataFrame está vazio
                    if dfn_3d_df is None or len(dfn_3d_df) == 0:
                        st.warning("⚠️ Nenhuma fratura foi gerada. Verifique os parâmetros e tente novamente.")
                        return
                    
                    # Verificar se as colunas necessárias existem
                    required_cols = ['radius', 'aperture', 'family']
                    missing_cols = [col for col in required_cols if col not in dfn_3d_df.columns]
                    
                    if missing_cols:
                        st.warning(f"⚠️ Dados incompletos. Colunas faltando: {missing_cols}. Clique em 'Gerar DFN 3D' novamente.")
                        return
                    
                    dfn_3d_df['area'] = np.pi * dfn_3d_df['radius']**2

                    st.divider()
                    st.subheader("📊 Estatísticas do DFN 3D")
                    
                    # ========== OBTER METADADOS DA GERAÇÃO ==========
                    dfn_metadata = st.session_state.get('dfn_3d_metadata', {})
                    n_fractures_calculated = dfn_metadata.get('n_fractures_calculated', len(dfn_3d_df))
                    n_fractures_generated = dfn_metadata.get('n_fractures_generated', len(dfn_3d_df))
                    n_fractures_limited = dfn_metadata.get('n_fractures_limited', False)
                    scale_factor = dfn_metadata.get('scale_factor', 1.0)
                    
                    volume = domain_size[0] * domain_size[1] * domain_size[2]
                    
                    # Médias das fraturas geradas (não escalam)
                    mean_radius = dfn_3d_df['radius'].mean()
                    mean_aperture_raw = dfn_3d_df['aperture'].mean()
                    mean_area = dfn_3d_df['area'].mean()
                    
                    # Aplicar fator de correção de abertura se configurado
                    aperture_factor = st.session_state.get('aperture_correction_3d', 1.0)
                    mean_aperture = mean_aperture_raw * aperture_factor
                    
                    # ========== VALORES TEÓRICOS (escalonados para n_calculated) ==========
                    total_area_theoretical = mean_area * n_fractures_calculated
                    P32_theoretical = total_area_theoretical / volume
                    porosity_theoretical = (mean_aperture * total_area_theoretical) / volume
                    P30_theoretical = n_fractures_calculated / volume
                    mean_length = 2 * mean_radius
                    P10_equiv_theoretical = P30_theoretical * mean_length
                    k_estimate = (mean_aperture**3) / 12

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total de fraturas", f"{n_fractures_calculated}")
                        st.metric("Área total (m²)", f"{total_area_theoretical:.2f}")

                    with col2:
                        st.metric("P32 (m²/m³)", f"{P32_theoretical:.2f}")
                        # Mostrar abertura com indicador de correção
                        abertura_label = "Abertura média (mm)"
                        if aperture_factor < 1.0:
                            abertura_label = f"Abertura corrigida (mm) ×{aperture_factor:.2f}"
                        st.metric(abertura_label, f"{mean_aperture * 1000:.2f} mm")
                    
                    with col3:
                        st.metric("Porosidade 3D (%)", f'{porosity_theoretical * 100:.2f} %')
                        st.metric("Raio médio (m)", f"{mean_radius:.2f} m")

                    with col4:
                        st.metric("Permeabilidade (mD)", f"{k_estimate * 1e12:.2f}", 
                                    help="Estimativa simplificada de permeabilidade (k = b³/12)")
                        
                        # Mostrar P20 para FRAMFRAT ou P10 para Scanline
                        current_type = st.session_state.get('analysis_type', 'FRAMFRAT')
                        if current_type == "FRAMFRAT":
                            # P20 = N / Área (fraturas por m²)
                            P20_theoretical = n_fractures_calculated / (domain_x * domain_y)
                            st.metric("P20 (1/m²)", f"{P20_theoretical:.2f}",
                                      help="Densidade de fraturas 2D: P20 = N / Área")
                        else:
                            st.metric("P10 equiv. (1/m)", f"{P10_equiv_theoretical:.2f}",
                                      help="Intensidade linear equivalente: P10 ≈ P30 × L_médio")

                    # ========== VALIDAÇÃO DOS RESULTADOS ==========
                    validation_warnings = []
                    
                    # Verificar abertura
                    abertura_mm = mean_aperture * 1000
                    if abertura_mm > 10:
                        validation_warnings.append(
                            f"⚠️ **Abertura muito alta** ({abertura_mm:.1f} mm). "
                            f"Valores típicos: 0.1-5 mm. Verificar dados de entrada."
                        )
                    
                    # Verificar porosidade
                    if porosity_theoretical * 100 > 10:
                        validation_warnings.append(
                            f"⚠️ **Porosidade irreal** ({porosity_theoretical*100:.1f}%). "
                            f"Valores típicos: < 5%. Causado pela abertura alta."
                        )
                    
                    # Verificar permeabilidade
                    perm_mD = k_estimate * 1e12
                    if perm_mD > 100000:
                        validation_warnings.append(
                            f"⚠️ **Permeabilidade irreal** ({perm_mD:.0f} mD). "
                            f"Valores típicos: 10-10.000 mD. Causado pela abertura alta."
                        )
                    
                    # Verificar P32
                    if P32_theoretical > 10:
                        validation_warnings.append(
                            f"⚠️ **P32 muito alto** ({P32_theoretical:.1f} m²/m³). "
                            f"Valores típicos: 0.1-5 m²/m³."
                        )
                    
                    # Exibir warnings se houver
                    if validation_warnings:
                        st.divider()
                        st.markdown("### ⚠️ Validação dos Resultados")
                        for warning in validation_warnings:
                            st.warning(warning)
                        
                        st.info(
                            "💡 **Recomendação**: A abertura medida pelo FRAMFRAT pode incluir "
                            "zona de alteração/intemperismo. Considere aplicar um fator de correção "
                            "na abertura (ex: dividir por 10-20) para obter valores mais realistas."
                        )

                    # ========== NOTIFICAÇÕES DE LIMITAÇÃO ==========
                    if n_fractures_limited:
                        st.info(
                            f"ℹ️ **Amostragem**: O modelo teórico contém **{n_fractures_calculated}** fraturas. "
                            f"Para visualização, foram geradas **{n_fractures_generated}** fraturas. "
                            f"As estatísticas acima são **teóricas** (calculadas para {n_fractures_calculated} fraturas)."
                        )

                    # Estatísticas por família
                    if 'family' in dfn_3d_df.columns and st.session_state.get('color_by_family_3d', False):
                        st.divider()
                        st.subheader("📈 Estatísticas por Família")
                        
                        # Obter pesos CONFIGURADOS (não gerados)
                        configured_weights = {}
                        if hasattr(st.session_state, 'family_configs_3d') and st.session_state.family_configs_3d:
                            for i, fc in enumerate(st.session_state.family_configs_3d):
                                configured_weights[i] = fc.get('weight', 1.0 / len(st.session_state.family_configs_3d))
                        
                        # Calcular distribuição baseada nos PESOS CONFIGURADOS
                        family_stats = []
                        for family_id in sorted(dfn_3d_df['family'].unique()):
                            family_data = dfn_3d_df[dfn_3d_df['family'] == family_id]
                            
                            # Usar peso CONFIGURADO, não o percentual gerado
                            if family_id in configured_weights:
                                pct_configured = configured_weights[family_id]
                            else:
                                # Fallback: calcular a partir das fraturas geradas
                                pct_configured = len(family_data) / len(dfn_3d_df)
                            
                            # Número teórico de fraturas nesta família (baseado no peso configurado)
                            n_theoretical_family = int(n_fractures_calculated * pct_configured)
                            
                            # Área escalada usando o peso configurado
                            area_sample = family_data['area'].sum()
                            area_theoretical = area_sample * (n_fractures_calculated * pct_configured) / max(len(family_data), 1)
                            
                            family_stats.append({
                                'Família': f'Fam {family_id + 1}',
                                'N° Fraturas': n_theoretical_family,
                                'Percentual (%)': f"{pct_configured*100:.1f}",
                                'Raio Médio (m)': f"{family_data['radius'].mean():.3f}",
                                'Dip Médio (°)': f"{family_data['dip'].mean():.1f}",
                                'Dip Dir Médio (°)': f"{family_data['dip_direction'].mean():.1f}",
                                'Área Total (m²)': f"{area_theoretical:.2f}"
                            })
                        
                        stats_df = pd.DataFrame(family_stats)
                        st.dataframe(stats_df, hide_index=True, width='stretch')
                        
                        # Gráfico de distribuição por família
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=[s['Família'] for s in family_stats],
                                values=[int(s['N° Fraturas']) for s in family_stats],
                                marker=dict(colors=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'][:len(family_stats)]),
                                textinfo='label+percent',
                                hovertemplate='%{label}<br>N° Fraturas: %{value}<br>Percentual: %{percent}<extra></extra>'
                            )])
                            fig_pie.update_layout(
                                title=f"Distribuição de Fraturas por Família (Total: {n_fractures_calculated})",
                                height=300
                            )
                            st.plotly_chart(fig_pie, width='stretch')
                        
                        with col_chart2:
                            # Diagrama de roseta estereográfico simplificado
                            fig_stereo = go.Figure()
                            
                            colors_stereo = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
                            
                            for family_id in sorted(dfn_3d_df['family'].unique()):
                                family_data = dfn_3d_df[dfn_3d_df['family'] == family_id]
                                
                                fig_stereo.add_trace(go.Scatterpolar(
                                    r=[1] * len(family_data),
                                    theta=family_data['dip_direction'].values,
                                    mode='markers',
                                    marker=dict(
                                        size=8,
                                        color=colors_stereo[family_id % len(colors_stereo)],
                                        opacity=0.6
                                    ),
                                    name=f'Set {family_id + 1}'
                                ))
                            
                            fig_stereo.update_layout(
                                title="Orientações por Família (Dip Direction)",
                                polar=dict(
                                    radialaxis=dict(visible=False),
                                    angularaxis=dict(direction="clockwise", rotation=90)
                                ),
                                height=300
                            )
                            st.plotly_chart(fig_stereo, width='stretch')
                    
                    # Distribuições estatísticas
                    st.divider()
                    st.subheader("📊 Distribuições Estatísticas")
                    
                    col_hist1, col_hist2 = st.columns(2)
                    
                    with col_hist1:
                        # Histograma de raios
                        fig_radius = go.Figure()
                        fig_radius.add_trace(go.Histogram(
                            x=dfn_3d_df['radius'],
                            nbinsx=30,
                            marker_color='#3498DB',
                            opacity=0.75
                        ))
                        fig_radius.update_layout(
                            title="Distribuição de Raios",
                            xaxis_title="Raio (mm)",
                            yaxis_title="Frequência",
                            height=300
                        )
                        st.plotly_chart(fig_radius, width='stretch')
                    
                    with col_hist2:
                        # Histograma de aberturas
                        fig_aperture = go.Figure()
                        fig_aperture.add_trace(go.Histogram(
                            x=dfn_3d_df['aperture'] * 1000,  # Converter para mm para exibição
                            nbinsx=30,
                            marker_color='#E74C3C',
                            opacity=0.75
                        ))
                        fig_aperture.update_layout(
                            title="Distribuição de Aberturas",
                            xaxis_title="Abertura (mm)",
                            yaxis_title="Frequência",
                            height=300
                        )
                        st.plotly_chart(fig_aperture, width='stretch')

            # Renderizar visualização (reativo aos widgets)
            render_current_view()
                
        else:
            st.markdown("")
            show_info("📋 Por favor, complete as análises anteriores primeiro")
            st.markdown("")


# Tab 7: Exportar
with tab5:
    st.header("💾 Exportação de Resultados")
    
    if st.session_state.data_loaded:
        exporter = ResultsExporter()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Dados Processados")
            
            # Exportar dados tratados
            if st.button("📥 Exportar Dados Tratados (CSV)"):
                if st.session_state.framfrat_data is not None:
                    csv_data = exporter.export_to_csv(st.session_state.framfrat_data)
                    st.download_button(
                        label="Download FRAMFRAT CSV",
                        data=csv_data,
                        file_name=f"framfrat_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
                if st.session_state.scanline_data is not None:
                    csv_scanline = exporter.export_to_csv(st.session_state.scanline_data)
                    st.download_button(
                        label="Download Scanline CSV",
                        data=csv_scanline,
                        file_name=f"scanline_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            # Exportar parâmetros ajustados
            if st.button("📊 Exportar Parâmetros (JSON)"):
                if st.session_state.analysis_results:
                    json_params = exporter.export_parameters(st.session_state.analysis_results)
                    st.download_button(
                        label="Download Parâmetros JSON",
                        data=json_params,
                        file_name=f"parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
        
        with col2:
            st.subheader("🗺️ Modelos DFN")
            
            # Exportar DFN 2D
            if hasattr(st.session_state, 'dfn_2d'):
                if st.button("📥 Exportar DFN 2D (GeoJSON)"):
                    geojson_data = exporter.export_dfn_2d_geojson(st.session_state.dfn_2d)
                    st.download_button(
                        label="Download DFN 2D GeoJSON",
                        data=geojson_data,
                        file_name=f"dfn_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
                        mime="application/geo+json"
                    )
            
            # Exportar DFN 3D
            if hasattr(st.session_state, 'dfn_3d'):
                if st.button("📥 Exportar DFN 3D (VTK)"):
                    vtk_data = exporter.export_dfn_3d_vtk(st.session_state.dfn_3d)
                    st.download_button(
                        label="Download DFN 3D VTK",
                        data=vtk_data,
                        file_name=f"dfn_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vtk",
                        mime="application/x-vtk"
                    )
        
        # Relatório completo
        st.divider()
        st.subheader("📄 Relatório Completo")
        
        if st.button("📋 Gerar Relatório Completo (Excel)", type="primary"):
            with st.spinner("Gerando relatório..."):
                # Coletar metadados
                metadata = {}
                
                if st.session_state.framfrat_data is not None:
                    metadata['image_area'] = st.session_state.framfrat_data.attrs.get('area', 1.0)
                    metadata['pixel_scale'] = st.session_state.framfrat_data.attrs.get('scale', 100.0)
                    metadata['l_min'] = st.session_state.get('l_min_framfrat', 0.001)
                    metadata['b_min'] = st.session_state.get('b_min_framfrat', 0.0001)
                
                if st.session_state.scanline_data is not None:
                    metadata['scanline_length'] = st.session_state.scanline_data.attrs.get('scanline_length', 10.0)
                    metadata['l_min_scan'] = st.session_state.get('l_min_scanline', 0.001)
                    metadata['b_min_scan'] = st.session_state.get('b_min_scanline', 0.0001)
                
                excel_data = exporter.generate_full_report(
                    st.session_state.framfrat_data,
                    st.session_state.scanline_data,
                    st.session_state.analysis_results,
                    metadata
                )

                st.download_button(
                    label="📥 Download Relatório Excel",
                    data=excel_data,
                    file_name=f"fracture_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        # Salvar/Carregar sessão
        st.divider()
        st.subheader("💼 Gerenciar Sessão")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Salvar Sessão"):
                session_data = exporter.save_session(st.session_state)
                st.download_button(
                    label="Download Sessão",
                    data=session_data,
                    file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_session = st.file_uploader("Carregar Sessão", type=['json'], key="session_upload")
            if uploaded_session and st.button("📂 Restaurar Sessão"):
                exporter.load_session(uploaded_session, st.session_state)

                st.markdown("")
                show_success("✅ Sessão restaurada!")
                st.markdown("")

                st.rerun()
    else:
        st.markdown("")
        show_info("📁 Por favor, carregue os dados primeiro")
        st.markdown("")

# Rodapé com referências
st.markdown("""
---
### 📚 Referências Científicas

- **Marrett, R.** (1996). Aggregate properties of fracture populations. *Journal of Structural Geology*, 18(2-3), 169-178.
- **Ortega, O.J., Marrett, R.A., & Laubach, S.E.** (2006). A scale-independent approach to fracture intensity and average spacing measurement. *AAPG Bulletin*, 90(2), 193-208.
- **Terzaghi, R.D.** (1965). Sources of error in joint surveys. *Géotechnique*, 15(3), 287-304.
- **Schultz, R.A. et al.** (2008). Displacement-length scaling relations for faults on the terrestrial planets. *Journal of Structural Geology*, 30(11), 1405-1411.

⚠️ **Observações importantes:**
- A área da imagem (FRAMFRAT) é crucial para normalização correta das densidades
- O comprimento da scanline é fundamental para cálculo de P10
- Comparações entre fontes requerem limiar comum de tamanho (Ortega et al., 2006)
- **NOVO**: Correções de vieses aumentam substancialmente as intensidades calculadas
""")

