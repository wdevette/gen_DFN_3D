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
from func_tools import force_dark_plotly_layout

from modules.orientation_clustering import (
    cluster_orientations_2d, 
    cluster_orientations_3d,
    extract_orientation_stats,
    auto_determine_n_sets
)

# Configuração da página
st.set_page_config(
    page_title="Análise de Fraturas - Marrett & Ortega",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 2px;}
    .stTabs [data-baseweb="tab"] {padding: 10px 20px;}
</style>
""", unsafe_allow_html=True)

# Título e descrição
st.title("⛏️ Sistema de Análise de Fraturas")
st.markdown("""
**Análise integrada de fraturas** baseada em Marrett (1996) e Ortega et al. (2006)
- Lei de potência para distribuições de tamanho
- Intensidade e espaçamento size-cognizant
- Geração de DFN estocástica
""")

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
    
    **3. Intensidade** 📏
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
    """)
    
    st.divider()
    
    st.markdown("""
    ### ⚙️ Versão
    **v1.0** - Sistema de Análise de Fraturas
    """)

# Área principal - Abas
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dados", 
    "📈 Ajustes", 
    "📏 Intensidade & Espaçamento",
    "🗺️ DFN 2D", 
    "🎲 DFN 3D", 
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
    
    if analysis_type is None:
        st.info("👆 Por favor, selecione o tipo de análise para continuar")
    
    elif analysis_type == "FRAMFRAT":
        st.divider()
        
        # Indicador de status com opção de limpar
        if st.session_state.framfrat_data is not None:
            col_status1, col_status2 = st.columns([3, 1])
            with col_status1:
                st.success("✅ Dados FRAMFRAT já processados na memória")
            with col_status2:
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
            
            if uploaded_framfrat:
                st.success("✅ Arquivo carregado!")
            
            st.markdown("##### ⚙️ Parâmetros da Imagem")
            image_area = st.number_input(
                "Área da imagem (mm²)",
                min_value=0.01,
                value=1000000.0,  # MUDOU: era 1.0, agora 1m² = 1000000mm²
                step=100.0,
                help="Área real representada pela imagem analisada",
                key="img_area"
            )
            
            pixel_per_mm = st.number_input(
                "Resolução/Escala (pixels/mm)",
                min_value=0.1,
                value=10.0, # MUDOU: era 100.0 pixels/m, agora 10 pixels/mm
                step=1.0,
                help="Número de pixels por metro na imagem",
                key="pixel_scale"
            )

            # Botão de processar
            process_framfrat = st.button(
                "🚀 Processar Dados FRAMFRAT",
                type="primary",
                #width='stretch',
                disabled=not uploaded_framfrat,
                help="Clique para processar os dados carregados",
                key="btn_process_framfrat",
            )
        
        # Processar dados quando botão é clicado
        with col2:                        
            st.markdown("##### 🔍 Filtros de Dados")
            l_min = st.number_input(
                "Comprimento mínimo (mm)", 
                min_value=0.0, 
                value=1.0, #0.001, 
                step=0.1, #0.001,
                format="%.1f",  #"%.3f",
                help="Filtrar fraturas menores que este valor",
                key="l_min_framfrat"
            )
            
            b_min = st.number_input(
                "Abertura mínima (mm)", 
                min_value=0.0, 
                value=0.1, #0.0001, 
                step=0.1, #0.0001,
                format="%.2f", #"%.4f",
                help="Filtrar fraturas com abertura menor que este valor",
                key="b_min_framfrat"
            )
            
            if process_framfrat and uploaded_framfrat:
                with st.spinner("Processando dados FRAMFRAT..."):
                    try:
                        loader = FractureDataLoader()
                        framfrat_data = loader.load_framfrat(
                            uploaded_framfrat,
                            image_area,
                            pixel_per_mm
                        )
                        st.session_state.framfrat_data = framfrat_data
                        st.session_state.data_loaded = True
                        st.session_state.analysis_type = "FRAMFRAT"
                        # Salvar parâmetros no session state

                        st.success("✅ Dados processados com sucesso!")
                                                            
                    except Exception as e:
                        st.error(f"❌ Erro ao processar FRAMFRAT: {str(e)}")

        # Preview dos dados (só mostra se dados foram processados)
        if st.session_state.framfrat_data is not None:
            framfrat_data = st.session_state.framfrat_data
            
            st.divider()
            st.markdown("")
            st.success(f"### ✅ {len(framfrat_data)} fraturas processadas")            
            
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
                    'Comprimento (mm)': preview_df['length'],
                    'Abertura (mm)': (preview_df['aperture'])
                })

                display_df['Comprimento (mm)'] = display_df['Comprimento (mm)'] \
                    .apply(lambda x: f"{x:.2f}".replace('.', ','))

                display_df['Abertura (mm)'] = display_df['Abertura (mm)'] \
                    .apply(lambda x: f"{x:.4f}".replace('.', ','))
                
                # Se houver 'ID_Segmento' válido, insere e reordena para ficar como 2ª coluna
                if show_segmento:
                    display_df["ID_Segmento"] = preview_df["ID_Segmento"]
                    desired_order = ["ID_Fratura", "ID_Segmento", "Comprimento (mm)", "Abertura (mm)"]
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
                with col2:
                    st.metric(
                        "Compr. médio", 
                        f"{framfrat_data['length'].mean():.3f} mm".replace(".", ",")
                    )
                with col3:
                    st.metric(
                        "Abertura média", 
                        f"{framfrat_data['aperture'].mean():.4f} mm".replace(".", ",")
                    )
                
                # Estatísticas adicionais
                st.divider()
                st.write("📝 **Estatísticas Detalhadas:**")
                
                stats_df = pd.DataFrame({
                    'Métrica': ['Mínimo', 'Máximo', 'Mediana', 'Desvio Padrão'],
                    'Comprimento (mm)': [
                        f"{framfrat_data['length'].min():.4f}".replace(".", ","),
                        f"{framfrat_data['length'].max():.4f}".replace(".", ","),
                        f"{framfrat_data['length'].median():.4f}".replace(".", ","),
                        f"{framfrat_data['length'].std():.4f}".replace(".", ",")
                    ],
                    'Abertura (mm)': [
                        f"{framfrat_data['aperture'].min():.4f}".replace(".", ","),
                        f"{framfrat_data['aperture'].max():.4f}".replace(".", ","),
                        f"{framfrat_data['aperture'].median():.4f}".replace(".", ","),
                        f"{framfrat_data['aperture'].std():.4f}".replace(".", ",")
                    ]
                })
                st.table(stats_df)
                    
    
    elif analysis_type == "Scanline":
        st.divider()
        
        # Indicador de status com opção de limpar
        if st.session_state.scanline_data is not None:
            col_status1, col_status2 = st.columns([3, 1])
            with col_status1:
                st.success("✅ Dados Scanline já processados na memória")
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
                "Arquivo Scanline (.txt/.csv)",
                type=['txt', 'csv'],
                help="Arquivo com posições e aberturas das fraturas",
                key="scanline_upload"
            )
            
            if uploaded_scanline:
                st.success("✅ Arquivo carregado!")
            
            st.markdown("##### 🔧 Parâmetros da Scanline")
            scanline_length = st.number_input(
                "Comprimento da scanline (m)",
                min_value=0.1,
                value=10.0,
                step=0.1,
                help="Comprimento total da linha de amostragem",
                key="scan_length"
            )
            
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
                        
                        st.success("✅ Dados processados com sucesso!")

                    except Exception as e:
                        st.error(f"❌ Erro ao processar Scanline: {str(e)}")
        
        # Preview dos dados (só mostra se dados foram processados)
        if st.session_state.scanline_data is not None:
            scanline_data = st.session_state.scanline_data
            
            st.divider()
            st.success(f"### ✅ {len(scanline_data)} fraturas processadas")
            
            # Preview dos dados
            with st.expander("📝 Preview dos dados Scanline"):
                st.dataframe(scanline_data.head(10))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Fraturas", len(scanline_data))
                with col2:
                    st.metric("Espaçamento médio (m)", f"{scanline_data['length'].mean():.3f}")
                with col3:
                    st.metric("Abertura média (mm)", f"{scanline_data['aperture'].mean()*1000:.2f}")
    
    # Seção de comparação (aparece apenas se ambos os dados forem carregados)
    st.divider()
    
    if st.checkbox("🔄 Modo de Comparação", help="Carregue ambos os tipos de dados para comparar"):
        st.subheader("📋 Carregar Dados para Comparação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**FRAMFRAT**")
            if st.session_state.framfrat_data is not None:
                st.success("✅ Dados FRAMFRAT processados")
                st.metric("Fraturas", len(st.session_state.framfrat_data))
            else:
                st.info("👆 Selecione FRAMFRAT acima e processe os dados")
        
        with col2:
            st.write("**Scanline**")
            if st.session_state.scanline_data is not None:
                st.success("✅ Dados Scanline processados")
                st.metric("Fraturas", len(st.session_state.scanline_data))
            else:
                st.info("👆 Selecione Scanline acima e processe os dados")
        
        # Verificar se ambos estão carregados
        if st.session_state.framfrat_data is not None and st.session_state.scanline_data is not None:
            st.session_state.comparison_mode = True
            st.success("✅ Modo de comparação ativado! Vá para a aba 'Intensidade & Espaçamento' para análise comparativa")
        else:
            st.warning("⚠️ Processe ambos os tipos de dados para ativar o modo de comparação")

# Tab 2: Ajustes de Lei de Potência
with tab2:
    st.header("🧩 Lei de Potência e Famílias de Fraturas")
    
    if st.session_state.data_loaded:   

        tab_powerL, tab_fratFam = st.tabs(['Lei de Potência', 'Famílias de Fraturas'])

        with tab_powerL:

            # Seletor de método de ajuste
            st.subheader("⚙️ Configuração de Ajuste para Lei de Potência")

            col_config1, col_config2 = st.columns([0.4, 1])
            with col_config1:
                fit_method = st.selectbox(
                    "Selecione o Método de ajuste", 
                    [None, "OLS", "MLE"],
                    format_func=lambda x: "Selecione um método" if x is None else f"{x} ({'log-log' if x == 'OLS' else 'Clauset et al.'})",
                    help="OLS: Mínimos quadrados ordinários em escala log-log\nMLE: Máxima verossimilhança (Clauset et al. 2009)"
                )
            
            with col_config2:
                if fit_method:
                    st.markdown("")
                    st.info(f"✔ Método: **{fit_method}**")
            
            if fit_method is None:
                st.warning("⚠️ Por favor, selecione um método de ajuste para continuar")
           
            else:
                #st.divider()
                fitter = PowerLawFitter()
                viz = FractureVisualizer()
                st.markdown('chegou')
                # Obter valores de filtro
                if 'l_min_framfrat' in st.session_state:
                    l_min = st.session_state.l_min_framfrat
                else:
                    l_min = 0.001
                    
                if 'b_min_framfrat' in st.session_state:
                    b_min = st.session_state.b_min_framfrat
                else:
                    b_min = 0.0001
            
                # Ajustar leis de potência
                results = {}
                
                
                col1, col2, col3 = st.columns(3)
                with col1: # Comprimento
                    st.subheader("Comprimento (l)")
                    if st.session_state.framfrat_data is not None:
                        l_fit = fitter.fit_power_law(
                            st.session_state.framfrat_data['length'].values,
                            l_min,
                            method=fit_method
                        )
                        results['length_fit'] = l_fit
                        
                        fig_l = viz.plot_power_law_fit(
                            st.session_state.framfrat_data['length'].values,
                            l_fit
                        )

                        # força tema escuro sem perder tuas configs essenciais
                        fig_l = force_dark_plotly_layout(fig_l)
                        st.plotly_chart(fig_l, width='stretch')
                        
                        # Mostrar métricas apropriadas baseadas no método
                        if fit_method == "OLS":
                            st.info(f"""
                            **Parâmetros ajustados:**
                            - Expoente (e): {l_fit['exponent']:.3f}
                            - Coeficiente (h): {l_fit['coefficient']:.2e}
                            - R²: {l_fit['r_squared']:.3f}
                            - p-valor: {l_fit['p_value']:.4f}
                            """)
                        else:  # MLE
                            st.info(f"""
                            **Parâmetros ajustados:**
                            - Expoente ($\\alpha$): {l_fit['exponent']:.3f}
                            - Coeficiente: {l_fit['coefficient']:.2e}
                            - Estatística KS: {l_fit['ks_statistic']:.3f}
                            - Erro padrão: {l_fit['sigma']:.3f}
                            """)
                
                # Abertura
                with col2:
                    st.subheader("Abertura (b)")
                    if st.session_state.framfrat_data is not None:
                        b_fit = fitter.fit_power_law(
                            st.session_state.framfrat_data['aperture'].values,
                            b_min,
                            method=fit_method
                        )
                        results['aperture_fit'] = b_fit
                        
                        fig_b = viz.plot_power_law_fit(
                            st.session_state.framfrat_data['aperture'].values,
                            b_fit
                        )

                        fig_b = force_dark_plotly_layout(fig_b)
                        st.plotly_chart(fig_b, width='stretch')
                        
                        # Mostrar métricas apropriadas
                        if fit_method == "OLS":
                            st.info(f"""
                            **Parâmetros ajustados:**
                            - Expoente (c): {b_fit['exponent']:.3f}
                            - Coeficiente (a): {b_fit['coefficient']:.2e}
                            - R²: {b_fit['r_squared']:.3f}
                            - p-valor: {b_fit['p_value']:.4f}
                            """)
                        else:  # MLE
                            st.info(f"""
                            **Parâmetros ajustados:**
                            - Expoente ($\\alpha$): {b_fit['exponent']:.3f}
                            - Coeficiente: {b_fit['coefficient']:.2e}
                            - Estatística KS: {b_fit['ks_statistic']:.3f}
                            - Erro padrão: {b_fit['sigma']:.3f}
                            """)
                
                # Relação b-l
                with col3:
                    st.subheader("Relação b-l")
                    if st.session_state.framfrat_data is not None:
                        bl_fit = fitter.fit_aperture_length_relation(
                            st.session_state.framfrat_data['aperture'].values,
                            st.session_state.framfrat_data['length'].values
                        )
                        results['bl_relation'] = bl_fit
                        
                        fig_bl = viz.plot_aperture_length_relation(
                            st.session_state.framfrat_data['aperture'].values,
                            st.session_state.framfrat_data['length'].values,
                            bl_fit
                        )

                        fig_bl = force_dark_plotly_layout(fig_bl)
                        st.plotly_chart(fig_bl, width='stretch')
                        
                        st.info(f"""
                        **Relação b = g·l^m:**
                        - Expoente (m): {bl_fit['m']:.3f}
                        - Coeficiente (g): {bl_fit['g']:.2e}
                        - R²: {bl_fit['r_squared']:.3f}
                        - p-valor: {bl_fit['p_value']:.4f}
                        """)
                
                # Salvar resultados
                st.session_state.analysis_results = results
    

        with tab_fratFam:
            st.subheader("🔄 Análise de Famílias de Fraturas")
                    # Verificar se há dados de orientação
            if 'orientation' in st.session_state.framfrat_data.columns:
                orientations = st.session_state.framfrat_data['orientation'].dropna().values
                
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
                            st.info(f"✓ Número ótimo detectado: **{n_sets} famílias/sets**")
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
                        
                        st.success(f"✅ {len(family_stats)} famílias identificadas")
                    
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
                    st.warning("⚠️ Poucos dados de orientação disponíveis para análise de famílias")
            else:
                st.info("ℹ️ Dados de orientação não disponíveis neste dataset")

    else:
        st.info("📊 Por favor, carregue os dados primeiro na aba 'Dados'")

# Tab 3: Intensidade e Espaçamento
with tab3:
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
        st.info("📁 Por favor, carregue os dados primeiro")

# Tab 4: DFN 2D
with tab4:
    st.header("🗺️ Geração de DFN 2D")
    
    if st.session_state.data_loaded and st.session_state.analysis_results:
        # Obter área da imagem
        if st.session_state.framfrat_data is not None:
            image_area = st.session_state.framfrat_data.attrs.get('area', 1.0)
            l_min = st.session_state.get('l_min_framfrat', 0.001)
        else:
            image_area = 1.0
            l_min = 0.001
        
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
                "Largura do domínio (mm)",
                min_value=10.0, # 0.1,
                value=float(np.sqrt(image_area)),
                step=10.0 # 0.1
            )
            
            domain_height = st.number_input(
                "Altura do domínio (m)",
                min_value=10.0, #0.1,
                value=float(np.sqrt(image_area)),
                step=10.0 #0.1
            )

             # NOVO: Usar famílias identificadas
            use_families = st.checkbox(
                "Usar famílias identificadas",
                value=True if hasattr(st.session_state, 'fracture_families') else False,
                help="Gerar fraturas respeitando as famílias identificadas na análise"
            )

            
            # Número de fraturas
            n_fractures = st.number_input(
                "Número de fraturas",
                min_value=10,
                value=100,
                step=10,
                help="Baseado na intensidade P10"
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
            
            fracture_shape_2d = st.selectbox(
                "Formato da Fratura",
                options=['lines', 'rectangles'],
                format_func=lambda x: {'lines': 'Linhas', 'rectangles': 'Retângulos'}.get(x, x),
                help="Escolha como representar as fraturas 2D. 'Discos' não se aplica a DFN 2D."
            )
            
            show_centers_2d = st.checkbox(
                "Mostrar Centros das Fraturas",
                value=False,
                help="Exibe o ponto central de cada fratura com uma cor de destaque."
            )
            
            show_numbers_2d = st.checkbox(
                "Mostrar Numeração das Fraturas",
                value=False,
                help="Exibe o número de contagem próximo ao centro de cada fratura."
            )
            
            # Botão de gerar
            generate_2d = st.button(
                "🎲 Gerar DFN 2D",
                type="primary",
                width='stretch'
            )
        
        with col2:
            if generate_2d:
                with st.spinner("Gerando DFN 2D..."):
                    # Usar a semente específica desta aba
                    generator = DFNGenerator(random_seed_2d)
                    viz = FractureVisualizer()

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
                        params = {
                            'exponent': st.session_state.analysis_results['length_fit']['exponent'],
                            'x_min': l_min,
                            'coefficient': st.session_state.analysis_results['length_fit']['coefficient'],
                        }
                        
                        # Adicionar parâmetros de abertura se disponíveis
                        if 'bl_relation' in st.session_state.analysis_results:
                            params['g'] = st.session_state.analysis_results['bl_relation']['g']
                            params['m'] = st.session_state.analysis_results['bl_relation']['m']
                        
                        # Adicionar orientação se disponível
                        if 'orientation' in st.session_state.framfrat_data.columns:
                            orientations = st.session_state.framfrat_data['orientation'].values
                            params['orientation_mean'] = np.mean(orientations)
                            params['orientation_std'] = np.std(orientations)
                    else:
                        params = {
                            'exponent': 2.0,
                            'x_min': 10.0, #mm #0.01,
                            'coefficient': 100
                        }
                    
                    # Gerar DFN com famílias
                    dfn_2d = generator.generate_2d_dfn(
                        params=params,
                        domain_size=(domain_width, domain_height),
                        n_fractures=n_fractures,
                        families=families
                    )
                    
                    # Visualizar com cores por família
                    fig_dfn = viz.plot_dfn_2d(
                        dfn_2d,
                        (domain_width, domain_height),
                        fracture_shape=fracture_shape_2d,
                        show_centers=show_centers_2d,
                        show_numbers=show_numbers_2d,
                        color_by_family=use_families
                    )
                    
                    st.plotly_chart(fig_dfn, width='stretch')
                    
                    # Converter lista de fraturas para DataFrame para estatísticas
                    dfn_df = pd.DataFrame([f.to_dict() for f in dfn_2d])
                    
                    # Estatísticas do DFN
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total de fraturas", len(dfn_2d))
                        st.metric("Comprimento total (mm)", f"{dfn_df['length'].sum():.2f}".replace(".", ","))
                    
                    with col2:
                        st.metric("Comprimento médio (mm)", f"{dfn_df['length'].mean():.2f}".replace(".", ","))
                        st.metric("Abertura média (mm)", f"{dfn_df['aperture'].mean():.3f}".replace(".", ","))
                    
                    with col3:
                        p21 = dfn_df['length'].sum() / (domain_width * domain_height)
                        st.metric("P21 (mm/mm²)", f"{p21:.4f}".replace(".", ","))
                        porosity = (dfn_df['aperture'] * dfn_df['length']).sum() / (domain_width * domain_height)
                        st.metric("Porosidade (%)", f"{porosity * 100:.3f}".replace(".", ","))

                        # st.metric("P21 (mm/mm²)", f"{dfn_df['length'].sum() / (domain_width * domain_height):.3f}")
                        # porosity = (dfn_df['aperture'] * dfn_df['length']).sum() / (domain_width * domain_height)
                        # st.metric("Porosidade (%)", f"{porosity * 100:.3f}")
                    
                    # Salvar DFN gerado
                    st.session_state.dfn_2d = dfn_2d
    else:
        st.info("📁Por favor, complete as análises anteriores primeiro")



# Tab 5: DFN 3D
with tab5:
    st.header("🎲 Geração de DFN 3D")
    
    if st.session_state.data_loaded and st.session_state.analysis_results:
        # Obter l_min
        l_min = st.session_state.get('l_min_framfrat', 1.0) #0.001)
        
        st.subheader("Configurações DFN 3D")

        col1, col2, col3 = st.columns(3) # DOMÍNIO 3D
        # domain_x = col1.number_input("Dimensão X (m)", min_value=10.0, value=st.session_state.get('dfn_3d_domain', [100.0, 100.0, 20.0])[0], step=1.0)
        # domain_y = col2.number_input("Dimensão Y (m)", min_value=10.0, value=st.session_state.get('dfn_3d_domain', [100.0, 100.0, 20.0])[1], step=1.0)    
        # domain_z = col3.number_input("Dimensão Z (m)", min_value=5.0, value=st.session_state.get('dfn_3d_domain', [100.0, 100.0, 20.0])[2], step=1.0)

        domain_x = col1.number_input("Dimensão X (m)", min_value=100.0, value=st.session_state.get('dfn_3d_domain', [10000.0, 10000.0, 2000.0])[0], step=100.0)
        domain_y = col2.number_input("Dimensão Y (m)", min_value=100.0, value=st.session_state.get('dfn_3d_domain', [10000.0, 10000.0, 2000.0])[1], step=100.0)    
        domain_z = col3.number_input("Dimensão Z (m)", min_value=50.0, value=st.session_state.get('dfn_3d_domain', [10000.0, 10000.0, 2000.0])[2], step=100.0)

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
                            
            # Número de fraturas
            n_fractures_3d = col_mid.number_input("Número de fraturas 3D", min_value=10, value=200, step=10)
        
        st.divider()
        
                # ========== CONFIGURAÇÃO DE FAMÍLIAS 3D ==========
        st.subheader("🔄 Famílias de Fraturas 3D")
        
        col_fam1, col_fam2 = st.columns([1, 2])
        
        with col_fam1:
            use_families_3d = st.checkbox(
                "Usar múltiplas famílias",
                value=True,
                help="Gerar fraturas em múltiplas famílias com orientações distintas",
                key="use_families_3d"
            )
            
            if use_families_3d:
                n_families_3d = st.selectbox(
                    "Número de famílias",
                    options=[2, 3, 4],
                    index=0,  # Default: 2 famílias
                    help="Número de famílias distintas de fraturas",
                    key="n_families_3d"
                )
                
                st.info(f"✓ Gerando **{n_families_3d} famílias** de fraturas")
        
        with col_fam2:
            if use_families_3d:
                st.markdown("##### ⚙️ Configuração das Famílias")
                
                # Armazenar configurações de cada família
                if 'family_configs_3d' not in st.session_state:
                    st.session_state.family_configs_3d = []
                
                family_configs = [] # Configuração simplificada das famílias
                for i in range(n_families_3d):
                    with st.expander(f"Set {i+1} - Configuração", expanded=(i==0)):
                        col_a, col_b, col_c = st.columns(3)
                        
                        # Orientações padrão distribuídas uniformemente
                        default_dip = dip_mean if i == 0 else 45
                        default_dip_dir = dip_dir_mean + (i * 180 // n_families_3d)
                        default_weight = 1.0 / n_families_3d
                        
                        family_dip = col_a.number_input(
                            "Dip (°)", 
                            min_value=0,
                            max_value=90,
                            value=default_dip,
                            step=5,
                            key=f"family_{i}_dip"
                        )
                        
                        family_dip_dir = col_b.number_input(
                            "Dip Dir (°)",
                            min_value=0,
                            max_value=360,
                            value=int(default_dip_dir % 360),
                            step=10,
                            key=f"family_{i}_dip_dir"
                        )
                        
                        family_weight = col_c.number_input(
                            "Peso (%)",
                            min_value=5,
                            max_value=100,
                            value=int(default_weight * 100),
                            step=5,
                            key=f"family_{i}_weight",
                            help="Percentual de fraturas desta família"
                        )
                        
                        family_configs.append({
                            'dip': family_dip,
                            'dip_dir': family_dip_dir,
                            'weight': family_weight / 100.0
                        })
                
                # Normalizar pesos
                total_weight = sum(f['weight'] for f in family_configs)
                for f in family_configs:
                    f['weight'] = f['weight'] / total_weight
                
                st.session_state.family_configs_3d = family_configs
        
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
        # if 'color_by_sets' not in st.session_state:
        #     st.session_state.color_by_sets = False
        # if 'num_sets' not in st.session_state:
        #     st.session_state.num_sets = None
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Tipo de visualização das fraturas**")
            viz_options = { #Radio Buttons para selação do modo de visualização das fraturas
                'lines': '📈 Linhas',
                'rectangles': '⬜ Retângulos', 
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
        
        # with col3:
        #     num_sets = st.selectbox(
        #         'Número de sets',
        #         options=[None, 1, 2, 3, 4],
        #         index=0,
        #         format_func=lambda x: 'Número de famílias' if x is None else str(x),
        #         help='Número de famílias das fraturas.',
        #         key='num_sets_select'
        #     )
        with col3:
            color_by_family_3d = st.checkbox(
                '🎨 Colorir por Família',
                value=use_families_3d,
                help='Colorir fraturas de acordo com sua família',
                key='color_by_family_3d'
            )

            # # ACTION: Ativar coloração por família
            # if num_sets is not None:
            #     st.session_state.color_by_sets = True
            #     st.session_state.num_sets = num_sets
            # else:
            #     st.session_state.color_by_sets = False
            #     st.session_state.num_sets = None
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
                if use_families_3d and hasattr(st.session_state, 'family_configs_3d'):
                    from modules.dfn_generator import FractureFamily
                    
                    families_3d = []
                    for config in st.session_state.family_configs_3d:
                        families_3d.append(FractureFamily(
                            orientation_mean=config['dip'],
                            orientation_std=10.0,  # Desvio padrão fixo
                            dip_dir_mean=config['dip_dir'],
                            dip_dir_std=20.0,  # Desvio padrão fixo
                            weight=config['weight']
                        ))
                
                # Preparar parâmetros
                if 'length_fit' in st.session_state.analysis_results:
                    params_3d = {
                        'exponent': st.session_state.analysis_results['length_fit']['exponent'],
                        'x_min': l_min,
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
                # # Gerar DFN 3D
                # dfn_3d = generator.generate_3d_dfn(
                #     params=params_3d,
                #     domain_size=(domain_x, domain_y, domain_z),
                #     n_fractures=n_fractures_3d
                # )

                # Gerar DFN 3D
                dfn_3d = generator.generate_3d_dfn(
                    params=params_3d,
                    domain_size=(domain_x, domain_y, domain_z),
                    n_fractures=n_fractures_3d,
                    families=families_3d
                )
                
                dfn_3d_df = pd.DataFrame([f.to_dict() for f in dfn_3d])
                
                # # ACTION: Atribuir famílias aleatórias se coloração por família ativada
                # if st.session_state.color_by_sets and st.session_state.num_sets:
                #     np.random.seed(random_seed_3d)
                #     dfn_3d_df['family'] = np.random.randint(0, st.session_state.num_sets, len(dfn_3d_df))
                
                # Salvar no estado
                st.session_state.dfn_3d = dfn_3d
                st.session_state.dfn_3d_df = dfn_3d_df
                st.session_state.dfn_3d_domain = (domain_x, domain_y, domain_z)
                
                st.divider()
                st.success("✅ DFN 3D gerado com sucesso!")

        # ========== FUNÇÃO DE RENDERIZAÇÃO REATIVA ==========
        def render_current_view():
            """
            ACTION: Renderiza a visualização 3D com base no estado atual.
            Chamada automaticamente quando widgets mudam.
            """
            if 'dfn_3d_df' not in st.session_state or st.session_state.dfn_3d_df is None:
                st.info(" ⚠️ Clique no botão 'Gerar DFN 3D' para visualizar o gráfico.")
                return
            
            viz = FractureVisualizer()
            domain_size = st.session_state.dfn_3d_domain
            
            with st.spinner("Atualizando visualização DFN 3D..."):
                # Chamar plot_dfn_3d com parâmetros do estado
                fig_dfn_3d = viz.plot_dfn_3d(
                    fractures_df=st.session_state.dfn_3d_df,
                    domain_size=domain_size,
                    shape_mode=st.session_state.viz_mode,
                    show_centers=st.session_state.show_centers_3d,
                    show_numbers=st.session_state.show_numbers_3d,
                    color_by_family=st.session_state.get('color_by_family_3d', False),
                    family_col='family'
                )
                
                st.plotly_chart(fig_dfn_3d, width='stretch')
                
                # ========== ESTATÍSTICAS ==========
                dfn_3d_df = st.session_state.dfn_3d_df
                dfn_3d_df['area'] = np.pi * dfn_3d_df['radius']**2

                st.divider()
                st.subheader("📊 Estatísticas do DFN 3D") # Estatísticas gerais

                col1, col2, col3, col4 = st.columns(4)
                volume = domain_size[0] * domain_size[1] * domain_size[2]

                with col1:
                    st.metric("Total de fraturas", len(dfn_3d_df))
                    st.metric("Área total (mm²)", f"{dfn_3d_df['area'].sum():.2f}")
                
                with col2:
                    p32 = dfn_3d_df['area'].sum() / volume
                    st.metric("P32 (mm²/mm³)", f"{p32:.5f}")
                    st.metric("Abertura média (mm)", f"{dfn_3d_df['aperture'].mean():.3f}")
                    #st.metric("Abertura média (mm)", f"{dfn_3d_df['aperture'].mean() * 1000:.2f}")
                
                with col3:
                    porosity_3d = (dfn_3d_df['aperture'] * dfn_3d_df['area']).sum() / volume
                    st.metric("Porosidade 3D (%)", f'{porosity_3d * 100:.3f}')
                    st.metric("Raio médio (mm)", f"{dfn_3d_df['radius'].mean():.2f}")

                with col4:
                    k_estimate = (dfn_3d_df['aperture']**3).mean() / 12 # Permeabilidade estimada (lei cúbica)
                    st.metric("Permeabilidade (mD)", f"{k_estimate * 1e12:.2f}", 
                                 help="Estimativa simplificada de permeabilidade (k = b³/12)")
                    
                    # Intensidade linear P10 equivalente
                    p10_equiv = dfn_3d_df['radius'].sum() * 2 / volume**(1/3)
                    st.metric("P10 equiv. (1/mm)", f"{p10_equiv:.4f}")

                # Estatísticas por família
                if 'family' in dfn_3d_df.columns and st.session_state.get('color_by_family_3d', False):
                    st.divider()
                    st.subheader("📈 Estatísticas por Família")
                    
                    family_stats = []
                    for family_id in sorted(dfn_3d_df['family'].unique()):
                        family_data = dfn_3d_df[dfn_3d_df['family'] == family_id]
                        
                        family_stats.append({
                            'Família': f'Set {family_id + 1}',
                            'N° Fraturas': len(family_data),
                            'Percentual (%)': f"{len(family_data)/len(dfn_3d_df)*100:.1f}",
                            'Raio Médio (mm)': f"{family_data['radius'].mean():.2f}",
                            'Dip Médio (°)': f"{family_data['dip'].mean():.1f}",
                            'Dip Dir Médio (°)': f"{family_data['dip_direction'].mean():.1f}",
                            'Área Total (mm²)': f"{family_data['area'].sum():.2f}"
                        })
                    
                    stats_df = pd.DataFrame(family_stats)
                    st.dataframe(stats_df, hide_index=True, width='stretch')
                    
                    # Gráfico de distribuição por família
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=[s['Família'] for s in family_stats],
                            values=[int(s['N° Fraturas']) for s in family_stats],
                            marker=dict(colors=['#E74C3C', '#3498DB', '#2ECC71', '#F39C12'][:len(family_stats)])
                        )])
                        fig_pie.update_layout(
                            title="Distribuição de Fraturas por Família",
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
                        x=dfn_3d_df['aperture'],
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
        st.info("📋 Por favor, complete as análises anteriores primeiro")


# Tab 6: Exportar
with tab6:
    st.header("ð💾 Exportação de Resultados")
    
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
                st.success("✅ Sessão restaurada!")
                st.rerun()
    else:
        st.info("📁 Por favor, carregue os dados primeiro")

# Rodapé com referências
st.markdown("""
---
### 📚 Referências Científicas

- **Marrett, R.** (1996). Aggregate properties of fracture populations. *Journal of Structural Geology*, 18(2-3), 169-178.
- **Ortega, O.J., Marrett, R.A., & Laubach, S.E.** (2006). A scale-independent approach to fracture intensity and average spacing measurement. *AAPG Bulletin*, 90(2), 193-208.

⚠️ **Observações importantes:**
- A área da imagem (FRAMFRAT) é crucial para normalização correta das densidades
- O comprimento da scanline é fundamental para cálculo de P10
- Comparações entre fontes requerem limiar comum de tamanho (Ortega et al., 2006)
""")

        












# import re
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# from pathlib import Path
# import json
# from datetime import datetime

# # Importar mÃ³dulos customizados
# from modules.io_fractures import FractureDataLoader
# from modules.powerlaw_fits import PowerLawFitter
# from modules.intensity_spacing import IntensitySpacingAnalyzer
# from modules.dfn_generator import DFNGenerator
# from modules.visualizations import FractureVisualizer
# from modules.results_exporter import ResultsExporter

# # ConfiguraÃ§Ã£o da pÃ¡gina
# st.set_page_config(
#     page_title="AnÃ¡lise de Fraturas - Marrett & Ortega",
#     page_icon="ðŸ”¬",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # CSS customizado
# st.markdown("""
# <style>
#     .main {padding: 0rem 1rem;}
#     .stTabs [data-baseweb="tab-list"] {gap: 2px;}
#     .stTabs [data-baseweb="tab"] {padding: 10px 20px;}
# </style>
# """, unsafe_allow_html=True)

# # TÃ­tulo e descriÃ§Ã£o
# st.title("ðŸ”¬ Sistema de AnÃ¡lise de Fraturas")
# st.markdown("""
# **AnÃ¡lise integrada de fraturas** baseada em Marrett (1996) e Ortega et al. (2006)
# - Lei de potÃªncia para distribuiÃ§Ãµes de tamanho
# - Intensidade e espaÃ§amento size-cognizant
# - GeraÃ§Ã£o de DFN estocÃ¡stica
# """)

# # Inicializar session state
# if 'data_loaded' not in st.session_state:
#     st.session_state.data_loaded = False
# if 'framfrat_data' not in st.session_state:
#     st.session_state.framfrat_data = None
# if 'scanline_data' not in st.session_state:
#     st.session_state.scanline_data = None
# if 'analysis_results' not in st.session_state:
#     st.session_state.analysis_results = {}
# if 'l_min_framfrat' not in st.session_state:
#     st.session_state.l_min_framfrat = 0.001
# if 'b_min_framfrat' not in st.session_state:
#     st.session_state.b_min_framfrat = 0.0001
# if 'l_min_scanline' not in st.session_state:
#     st.session_state.l_min_scanline = 0.001
# if 'b_min_scanline' not in st.session_state:
#     st.session_state.b_min_scanline = 0.0001

# # Sidebar simplificado
# with st.sidebar:
#     st.header("â„¹ï¸ InformaÃ§Ãµes")
    
#     st.markdown("""
#     ### ðŸ’¡ Guia de Uso
    
#     **1. Dados** ðŸ“Š
#     - Selecione o tipo de anÃ¡lise
#     - Configure parÃ¢metros e filtros
#     - FaÃ§a upload do arquivo
    
#     **2. Ajustes** ðŸ“ˆ
#     - Escolha o mÃ©todo (OLS ou MLE)
#     - Visualize os ajustes das leis de potÃªncia
    
#     **3. Intensidade** ðŸ“
#     - Analise P10 e espaÃ§amento
#     - Compare diferentes fontes
    
#     **4. DFN** ðŸ—ºï¸
#     - Gere redes 2D e 3D
#     - Configure parÃ¢metros estocÃ¡sticos
    
#     **5. Exportar** ðŸ’¾
#     - Baixe resultados e relatÃ³rios
#     """)
    
#     st.divider()
    
#     st.markdown("""
#     ### ðŸ“š ReferÃªncias
#     - Marrett (1996)
#     - Ortega et al. (2006)
#     """)
    
#     st.divider()
    
#     st.markdown("""
#     ### ðŸ”§ VersÃ£o
#     **v1.0** - Sistema de AnÃ¡lise de Fraturas
#     """)

# # Ãrea principal - Abas
# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#     "ðŸ“Š Dados", 
#     "ðŸ“ˆ Ajustes", 
#     "ðŸ“ Intensidade & EspaÃ§amento",
#     "ðŸ—ºï¸ DFN 2D", 
#     "ðŸŽ² DFN 3D", 
#     "ðŸ’¾ Exportar"
# ])

# # Tab 1: Upload de Dados
# with tab1:
#     st.header("ðŸ“ Upload de Dados")
    
#     # SeleÃ§Ã£o do tipo de anÃ¡lise
#     st.subheader("ðŸ“Š Tipo de AnÃ¡lise")
#     analysis_type = st.radio(
#         "Selecione o tipo de dados que deseja analisar:",
#         options=["FRAMFRAT", "Scanline"],
#         index=None,
#         horizontal=True,
#         help="FRAMFRAT: AnÃ¡lise de imagens 2D | Scanline: AnÃ¡lise linear 1D"
#     )
    
#     if analysis_type is None:
#         st.info("ðŸ‘† Por favor, selecione o tipo de anÃ¡lise para continuar")
    
#     elif analysis_type == "FRAMFRAT":
#         st.divider()
        
#         # Indicador de status com opÃ§Ã£o de limpar
#         if st.session_state.framfrat_data is not None:
#             col_status1, col_status2 = st.columns([3, 1])
#             with col_status1:
#                 st.success("âœ… Dados FRAMFRAT jÃ¡ processados na memÃ³ria")
#             with col_status2:
#                 if st.button("ðŸ—‘ï¸ Limpar", key="clear_framfrat", help="Limpar dados processados"):
#                     st.session_state.framfrat_data = None
#                     st.session_state.data_loaded = False
#                     st.session_state.analysis_results = {}
#                     st.rerun()
        
#         #DADOS FRAMFRAT
#         st.subheader("FRAMFRAT (.xlsx)")
        
#         col1, col2 = st.columns([1, 1], gap="large")
        
#         with col1:
#             st.markdown("##### ðŸ“¤ Upload de Arquivo")
#             uploaded_framfrat = st.file_uploader(
#                 "Arquivo FRAMFRAT (.xlsx)",
#                 type=['xlsx', 'xls'],
#                 help="Arquivo Excel com colunas: cumprimento, abertura, orientaÃ§Ã£o (opcional), x, y",
#                 key="framfrat_upload"
#             )
            
#             if uploaded_framfrat:
#                 st.success("âœ… Arquivo carregado!")
            
#             st.markdown("##### âš™ï¸ ParÃ¢metros da Imagem")
#             image_area = st.number_input(
#                 "Ãrea da imagem (mÂ²)",
#                 min_value=0.01,
#                 value=1.0,
#                 step=0.01,
#                 help="Ãrea real representada pela imagem analisada",
#                 key="img_area"
#             )
            
#             pixel_per_m = st.number_input(
#                 "ResoluÃ§Ã£o/Escala (pixels/m)",
#                 min_value=1.0,
#                 value=100.0,
#                 step=1.0,
#                 help="NÃºmero de pixels por metro na imagem",
#                 key="pixel_scale"
#             )

#             # BotÃ£o de processar
#             process_framfrat = st.button(
#                 "ðŸš€ Processar Dados FRAMFRAT",
#                 type="primary",
#                 #width='stretch',
#                 disabled=not uploaded_framfrat,
#                 help="Clique para processar os dados carregados",
#                 key="btn_process_framfrat",
#             )
        
#         # Processar dados quando botÃ£o Ã© clicado
#         with col2:
#             #if not process_framfrat and not st.session_state.framfrat_data:
#             # if (not process_framfrat) and (st.session_state.framfrat_data is None):

#             #     st.info("""
#             #     ### ðŸ“‹ InstruÃ§Ãµes
                
#             #     1. **FaÃ§a upload** do arquivo FRAMFRAT (.xlsx)
#             #     2. **Configure** os parÃ¢metros da imagem
#             #     3. **Ajuste** os filtros se necessÃ¡rio
#             #     4. **Clique** em "Processar Dados"
                
#             #     Os dados serÃ£o carregados e validados.
#             #     """)
                        
#             st.markdown("##### ðŸ” Filtros de Dados")
#             l_min = st.number_input(
#                 "Comprimento mÃ­nimo (m)", 
#                 min_value=0.0, 
#                 value=0.001, 
#                 step=0.001,
#                 format="%.3f",
#                 help="Filtrar fraturas menores que este valor",
#                 key="l_min_framfrat"
#             )
            
#             b_min = st.number_input(
#                 "Abertura mÃ­nima (m)", 
#                 min_value=0.0, 
#                 value=0.0001, 
#                 step=0.0001,
#                 format="%.4f",
#                 help="Filtrar fraturas com abertura menor que este valor",
#                 key="b_min_framfrat"
#             )
            
#             if process_framfrat and uploaded_framfrat:
#                 with st.spinner("Processando dados FRAMFRAT..."):
#                     try:
#                         loader = FractureDataLoader()
#                         framfrat_data = loader.load_framfrat(
#                             uploaded_framfrat,
#                             image_area,
#                             pixel_per_m
#                         )
#                         st.session_state.framfrat_data = framfrat_data
#                         st.session_state.data_loaded = True
#                         st.session_state.analysis_type = "FRAMFRAT"
#                         # Salvar parÃ¢metros no session state

#                         # st.session_state.l_min_framfrat = l_min
#                         # st.session_state.b_min_framfrat = b_min
                        
#                         st.success("âœ… Dados processados com sucesso!")
                                            
#                     except Exception as e:
#                         st.error(f"âŒ Erro ao processar FRAMFRAT: {str(e)}")

#         # Preview dos dados (sÃ³ mostra se dados foram processados)
#         if st.session_state.framfrat_data is not None:
#             framfrat_data = st.session_state.framfrat_data
            
#             st.divider()
#             st.markdown("")
#             st.success(f"### âœ… {len(framfrat_data)} fraturas processadas")            
            
#             with st.expander("ðŸ“‹ Preview dos dados FRAMFRAT", expanded=True):
#                 # Mostrar primeiras linhas
#                 preview_df = framfrat_data[['ID_Fratura', 'ID_Segmento', 'length', 'aperture']].head(5).copy()

#                 # Detectar se devemos exibir ID_Segmento (nÃ£o-nulo e nÃ£o string vazia)
#                 show_segmento = (
#                     "ID_Segmento" in preview_df.columns and
#                     preview_df["ID_Segmento"].replace(r"^\s*$", pd.NA, regex=True).notna().any()
#                 )
                
#                 # Criar DataFrame para display com unidades corretas
#                 display_df = pd.DataFrame({
#                     'ID_Fratura': preview_df['ID_Fratura'],
#                     'Comprimento (m)': preview_df['length'].round(4),
#                     'Abertura (mm)': (preview_df['aperture'] * 1000).round(2)
#                 })
                
#                 # Se houver 'ID_Segmento' vÃ¡lido, insere e reordena para ficar como 2Âª coluna
#                 if show_segmento:
#                     display_df["ID_Segmento"] = preview_df["ID_Segmento"]
#                     desired_order = ["ID_Fratura", "ID_Segmento", "Comprimento (m)", "Abertura (mm)"]
#                     display_df = display_df.reindex(columns=desired_order)

#                 st.dataframe(display_df, hide_index=True)
                
#                 st.divider()
                
#                 # EstatÃ­sticas
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric(
#                         "Total de Fraturas", 
#                         len(framfrat_data)
#                     )
#                 with col2:
#                     st.metric(
#                         "Compr. mÃ©dio", 
#                         f"{framfrat_data['length'].mean():.3f} m"
#                     )
#                 with col3:
#                     st.metric(
#                         "Abertura mÃ©dia", 
#                         f"{framfrat_data['aperture'].mean()*1000:.2f} mm"
#                     )
                
#                 # EstatÃ­sticas adicionais
#                 st.divider()
#                 st.write("ðŸ“Š **EstatÃ­sticas Detalhadas:**")
                
#                 stats_df = pd.DataFrame({
#                     'MÃ©trica': ['MÃ­nimo', 'MÃ¡ximo', 'Mediana', 'Desvio PadrÃ£o'],
#                     'Comprimento (m)': [
#                         f"{framfrat_data['length'].min():.4f}",
#                         f"{framfrat_data['length'].max():.4f}",
#                         f"{framfrat_data['length'].median():.4f}",
#                         f"{framfrat_data['length'].std():.4f}"
#                     ],
#                     'Abertura (mm)': [
#                         f"{framfrat_data['aperture'].min()*1000:.3f}",
#                         f"{framfrat_data['aperture'].max()*1000:.3f}",
#                         f"{framfrat_data['aperture'].median()*1000:.3f}",
#                         f"{framfrat_data['aperture'].std()*1000:.3f}"
#                     ]
#                 })
#                 st.table(stats_df)
                    
    
#     elif analysis_type == "Scanline":
#         st.divider()
        
#         # Indicador de status com opÃ§Ã£o de limpar
#         if st.session_state.scanline_data is not None:
#             col_status1, col_status2 = st.columns([3, 1])
#             with col_status1:
#                 st.success("âœ… Dados Scanline jÃ¡ processados na memÃ³ria")
#             with col_status2:
#                 if st.button("ðŸ—‘ï¸ Limpar", key="clear_scanline", help="Limpar dados processados"):
#                     st.session_state.scanline_data = None
#                     st.session_state.data_loaded = False
#                     st.session_state.analysis_results = {}
#                     st.rerun()
        
#         #DADOS SCANLINE
#         st.subheader("ðŸ“ AnÃ¡lise Scanline (Linear 1D)")
        
#         col1, col2 = st.columns([1, 1], gap='large')

#         with col1:
#             st.markdown("##### ðŸ“¤ Upload de Arquivo")
#             uploaded_scanline = st.file_uploader(
#                 "Arquivo Scanline (.txt/.csv)",
#                 type=['txt', 'csv'],
#                 help="Arquivo com posiÃ§Ãµes e aberturas das fraturas",
#                 key="scanline_upload"
#             )
            
#             if uploaded_scanline:
#                 st.success("âœ… Arquivo carregado!")
            
#             st.markdown("##### âš™ï¸ ParÃ¢metros da Scanline")
#             scanline_length = st.number_input(
#                 "Comprimento da scanline (m)",
#                 min_value=0.1,
#                 value=10.0,
#                 step=0.1,
#                 help="Comprimento total da linha de amostragem",
#                 key="scan_length"
#             )
            
#             scanline_azimuth = st.number_input(
#                 "Azimute da linha (Â°)", 
#                 min_value=0, 
#                 max_value=360, 
#                 value=0,
#                 help="OrientaÃ§Ã£o da scanline",
#                 key="scan_azimuth"
#             )
            
#             # BotÃ£o de processar
#             process_scanline = st.button(
#                 "ðŸš€ Processar Dados Scanline",
#                 type="primary",
#                 #width='stretch',
#                 disabled=not uploaded_scanline,
#                 help="Clique para processar os dados carregados",
#                 key="btn_process_scanline"
#             )
        
#         # Processar dados quando botÃ£o Ã© clicado
#         with col2:
#             #if not process_scanline and not st.session_state.scanline_data:
#             # if (not process_scanline) and (st.session_state.scanline_data is None):


#             #     st.info("""
#             #     ### ðŸ“‹ InstruÃ§Ãµes
                
#             #     1. **FaÃ§a upload** do arquivo Scanline (.txt/.csv)
#             #     2. **Configure** os parÃ¢metros da linha
#             #     3. **Ajuste** os filtros se necessÃ¡rio
#             #     4. **Clique** em "Processar Dados"
                
#             #     Os dados serÃ£o carregados e validados.
#             #     """)

#             st.markdown("##### ðŸ” Filtros de Dados")
#             l_min_scan = st.number_input(
#                 "EspaÃ§amento mÃ­nimo (m)", 
#                 min_value=0.0, 
#                 value=0.001, 
#                 step=0.001,
#                 format="%.3f",
#                 help="Filtrar fraturas com espaÃ§amento menor que este valor",
#                 key="l_min_scanline"
#             )
            
#             b_min_scan = st.number_input(
#                 "Abertura mÃ­nima (m)", 
#                 min_value=0.0, 
#                 value=0.0001, 
#                 step=0.0001,
#                 format="%.4f",
#                 help="Filtrar fraturas com abertura menor que este valor",
#                 key="b_min_scanline"
#             )
            
#             if process_scanline and uploaded_scanline:
#                 with st.spinner("Processando dados Scanline..."):
#                     try:
#                         loader = FractureDataLoader()
#                         scanline_data = loader.load_scanline(
#                             uploaded_scanline,
#                             scanline_length
#                         )
#                         st.session_state.scanline_data = scanline_data
#                         st.session_state.data_loaded = True
#                         st.session_state.analysis_type = "Scanline"
#                         # Salvar parÃ¢metros no session state
#                         # st.session_state.l_min_scanline = l_min_scan
#                         # st.session_state.b_min_scanline = b_min_scan
                        
#                         st.success("âœ… Dados processados com sucesso!")

#                     except Exception as e:
#                         st.error(f"âŒ Erro ao processar Scanline: {str(e)}")
    
#         # Preview dos dados (sÃ³ mostra se dados foram processados)
#         if st.session_state.scanline_data is not None:
#             scanline_data = st.session_state.scanline_data
            
#             st.divider()
#             st.success(f"### âœ… {len(scanline_data)} fraturas processadas")
            
#             # Preview dos dados
#             with st.expander("ðŸ“‹ Preview dos dados Scanline"):
#                 st.dataframe(scanline_data.head(10))
                
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("Fraturas", len(scanline_data))
#                 with col2:
#                     st.metric("EspaÃ§amento mÃ©dio (m)", f"{scanline_data['length'].mean():.3f}")
#                 with col3:
#                     st.metric("Abertura mÃ©dia (mm)", f"{scanline_data['aperture'].mean()*1000:.2f}")
    
#     # SeÃ§Ã£o de comparaÃ§Ã£o (aparece apenas se ambos os dados forem carregados)
#     st.divider()
    
#     if st.checkbox("ðŸ”„ Modo de ComparaÃ§Ã£o", help="Carregue ambos os tipos de dados para comparar"):
#         st.subheader("ðŸ“Š Carregar Dados para ComparaÃ§Ã£o")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.write("**FRAMFRAT**")
#             if st.session_state.framfrat_data is not None:
#                 st.success("âœ… Dados FRAMFRAT processados")
#                 st.metric("Fraturas", len(st.session_state.framfrat_data))
#             else:
#                 st.info("ðŸ‘† Selecione FRAMFRAT acima e processe os dados")
        
#         with col2:
#             st.write("**Scanline**")
#             if st.session_state.scanline_data is not None:
#                 st.success("âœ… Dados Scanline processados")
#                 st.metric("Fraturas", len(st.session_state.scanline_data))
#             else:
#                 st.info("ðŸ‘† Selecione Scanline acima e processe os dados")
        
#         # Verificar se ambos estÃ£o carregados
#         if st.session_state.framfrat_data is not None and st.session_state.scanline_data is not None:
#             st.session_state.comparison_mode = True
#             st.success("âœ… Modo de comparaÃ§Ã£o ativado! VÃ¡ para a aba 'Intensidade & EspaÃ§amento' para anÃ¡lise comparativa")
#         else:
#             st.warning("âš ï¸ Processe ambos os tipos de dados para ativar o modo de comparaÃ§Ã£o")

# # Tab 2: Ajustes de Lei de PotÃªncia
# with tab2:
#     st.header("ðŸ“ˆ Lei de PotÃªncia")
    
#     if st.session_state.data_loaded:
#         # Seletor de mÃ©todo de ajuste
#         st.subheader("âš™ï¸ ConfiguraÃ§Ã£o de Ajuste para Lei de PotÃªncia")
        
#         col_config1, col_config2 = st.columns([0.4, 1])
        
#         with col_config1:
#             fit_method = st.selectbox(
#                 "Selecione o MÃ©todo de ajuste", 
#                 [None, "OLS", "MLE"],
#                 format_func=lambda x: "Selecione um mÃ©todo" if x is None else f"{x} ({'log-log' if x == 'OLS' else 'Clauset et al.'})",
#                 help="OLS: MÃ­nimos quadrados ordinÃ¡rios em escala log-log\nMLE: MÃ¡xima verossimilhanÃ§a (Clauset et al. 2009)"
#             )
        
#         with col_config2:
#             if fit_method:
#                 st.markdown("")
#                 st.info(f"âœ“ MÃ©todo: **{fit_method}**")
        
#         if fit_method is None:
#             st.warning("âš ï¸ Por favor, selecione um mÃ©todo de ajuste para continuar")
#         else:
#             st.divider()
            
#             fitter = PowerLawFitter()
#             viz = FractureVisualizer()
            
#             # Obter valores de filtro
#             if 'l_min_framfrat' in st.session_state:
#                 l_min = st.session_state.l_min_framfrat
#             else:
#                 l_min = 0.001
                
#             if 'b_min_framfrat' in st.session_state:
#                 b_min = st.session_state.b_min_framfrat
#             else:
#                 b_min = 0.0001
            
#             # Ajustar leis de potÃªncia
#             results = {}
            
#             col1, col2, col3 = st.columns(3)
            
#             # Comprimento
#             with col1:
#                 st.subheader("Comprimento (l)")
#                 if st.session_state.framfrat_data is not None:
#                     l_fit = fitter.fit_power_law(
#                         st.session_state.framfrat_data['length'].values,
#                         l_min,
#                         method=fit_method
#                     )
#                     results['length_fit'] = l_fit
                    
#                     fig_l = viz.plot_power_law_fit(
#                         st.session_state.framfrat_data['length'].values,
#                         l_fit
#                     )
#                     st.plotly_chart(fig_l, width='stretch')
                    
#                     # Mostrar mÃ©tricas apropriadas baseadas no mÃ©todo
#                     if fit_method == "OLS":
#                         st.info(f"""
#                         **ParÃ¢metros ajustados:**
#                         - Expoente (e): {l_fit['exponent']:.3f}
#                         - Coeficiente (h): {l_fit['coefficient']:.2e}
#                         - RÂ²: {l_fit['r_squared']:.3f}
#                         - p-valor: {l_fit['p_value']:.4f}
#                         """)
#                     else:  # MLE
#                         st.info(f"""
#                         **ParÃ¢metros ajustados:**
#                         - Expoente (Î±): {l_fit['exponent']:.3f}
#                         - Coeficiente: {l_fit['coefficient']:.2e}
#                         - EstatÃ­stica KS: {l_fit['ks_statistic']:.3f}
#                         - Erro padrÃ£o: {l_fit['sigma']:.3f}
#                         """)
            
#             # Abertura
#             with col2:
#                 st.subheader("Abertura (b)")
#                 if st.session_state.framfrat_data is not None:
#                     b_fit = fitter.fit_power_law(
#                         st.session_state.framfrat_data['aperture'].values,
#                         b_min,
#                         method=fit_method
#                     )
#                     results['aperture_fit'] = b_fit
                    
#                     fig_b = viz.plot_power_law_fit(
#                         st.session_state.framfrat_data['aperture'].values,
#                         b_fit
#                     )
#                     st.plotly_chart(fig_b, width='stretch')
                    
#                     # Mostrar mÃ©tricas apropriadas
#                     if fit_method == "OLS":
#                         st.info(f"""
#                         **ParÃ¢metros ajustados:**
#                         - Expoente (c): {b_fit['exponent']:.3f}
#                         - Coeficiente (a): {b_fit['coefficient']:.2e}
#                         - RÂ²: {b_fit['r_squared']:.3f}
#                         - p-valor: {b_fit['p_value']:.4f}
#                         """)
#                     else:  # MLE
#                         st.info(f"""
#                         **ParÃ¢metros ajustados:**
#                         - Expoente (Î±): {b_fit['exponent']:.3f}
#                         - Coeficiente: {b_fit['coefficient']:.2e}
#                         - EstatÃ­stica KS: {b_fit['ks_statistic']:.3f}
#                         - Erro padrÃ£o: {b_fit['sigma']:.3f}
#                         """)
            
#             # RelaÃ§Ã£o b-l
#             with col3:
#                 st.subheader("RelaÃ§Ã£o b-l")
#                 if st.session_state.framfrat_data is not None:
#                     bl_fit = fitter.fit_aperture_length_relation(
#                         st.session_state.framfrat_data['aperture'].values,
#                         st.session_state.framfrat_data['length'].values
#                     )
#                     results['bl_relation'] = bl_fit
                    
#                     fig_bl = viz.plot_aperture_length_relation(
#                         st.session_state.framfrat_data['aperture'].values,
#                         st.session_state.framfrat_data['length'].values,
#                         bl_fit
#                     )
#                     st.plotly_chart(fig_bl, width='stretch')
                    
#                     st.info(f"""
#                     **RelaÃ§Ã£o b = gÂ·l^m:**
#                     - Expoente (m): {bl_fit['m']:.3f}
#                     - Coeficiente (g): {bl_fit['g']:.2e}
#                     - RÂ²: {bl_fit['r_squared']:.3f}
#                     - p-valor: {bl_fit['p_value']:.4f}
#                     """)
            
#             # Salvar resultados
#             st.session_state.analysis_results = results
#     else:
#         st.info("ðŸ“ Por favor, carregue os dados primeiro na aba 'Dados'")

# # Tab 3: Intensidade e EspaÃ§amento
# with tab3:
#     st.header("ðŸ“ AnÃ¡lise de Intensidade e EspaÃ§amento")
    
#     if st.session_state.data_loaded:
#         analyzer = IntensitySpacingAnalyzer()
#         viz = FractureVisualizer()
        
#         # Obter parÃ¢metros do session_state
#         if st.session_state.framfrat_data is not None:
#             image_area = st.session_state.framfrat_data.attrs.get('area', 1.0)
#             l_min = st.session_state.get('l_min_framfrat', 0.001)
#         else:
#             l_min = st.session_state.get('l_min_scanline', 0.001)
            
#         if st.session_state.scanline_data is not None:
#             scanline_length = st.session_state.scanline_data.attrs.get('scanline_length', 10.0)
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("Intensidade P10 (Size-Cognizant)")
            
#             # Calcular intensidades para diferentes limiares
#             if st.session_state.framfrat_data is not None:
#                 max_length_f = st.session_state.framfrat_data['length'].max()
#             else:
#                 max_length_f = 1.0
                
#             if st.session_state.scanline_data is not None:
#                 max_length_s = st.session_state.scanline_data['length'].max()
#             else:
#                 max_length_s = 1.0
            
#             thresholds = np.logspace(
#                 np.log10(l_min), 
#                 np.log10(max(max_length_f, max_length_s)), 
#                 50
#             )
            
#             intensities_framfrat = []
#             intensities_scanline = []
            
#             for threshold in thresholds:
#                 if st.session_state.framfrat_data is not None:
#                     p10_f = analyzer.calculate_p10(
#                         st.session_state.framfrat_data,
#                         threshold,
#                         image_area
#                     )
#                     intensities_framfrat.append(p10_f)
                
#                 if st.session_state.scanline_data is not None:
#                     p10_s = analyzer.calculate_p10_scanline(
#                         st.session_state.scanline_data,
#                         threshold,
#                         scanline_length
#                     )
#                     intensities_scanline.append(p10_s)
            
#             # Plotar curva de intensidade
#             fig_intensity = go.Figure()
            
#             if intensities_framfrat:
#                 fig_intensity.add_trace(go.Scatter(
#                     x=thresholds,
#                     y=intensities_framfrat,
#                     mode='lines',
#                     name='FRAMFRAT',
#                     line=dict(color='blue', width=2)
#                 ))
            
#             if intensities_scanline:
#                 fig_intensity.add_trace(go.Scatter(
#                     x=thresholds,
#                     y=intensities_scanline,
#                     mode='lines',
#                     name='Scanline',
#                     line=dict(color='red', width=2)
#                 ))
            
#             fig_intensity.update_layout(
#                 title="Intensidade vs Limiar de Tamanho",
#                 xaxis_title="Limiar de comprimento (m)",
#                 yaxis_title="P10 (fraturas/m)",
#                 xaxis_type="log",
#                 yaxis_type="log",
#                 hovermode='x unified'
#             )
            
#             st.plotly_chart(fig_intensity, width='stretch')
        
#         with col2:
#             st.subheader("EspaÃ§amento MÃ©dio")
            
#             # Calcular espaÃ§amentos
#             spacings_framfrat = [1/i if i > 0 else np.nan for i in intensities_framfrat]
#             spacings_scanline = [1/i if i > 0 else np.nan for i in intensities_scanline]
            
#             # Plotar curva de espaÃ§amento
#             fig_spacing = go.Figure()
            
#             if spacings_framfrat:
#                 fig_spacing.add_trace(go.Scatter(
#                     x=thresholds,
#                     y=spacings_framfrat,
#                     mode='lines',
#                     name='FRAMFRAT',
#                     line=dict(color='blue', width=2)
#                 ))
            
#             if spacings_scanline:
#                 fig_spacing.add_trace(go.Scatter(
#                     x=thresholds,
#                     y=spacings_scanline,
#                     mode='lines',
#                     name='Scanline',
#                     line=dict(color='red', width=2)
#                 ))
            
#             fig_spacing.update_layout(
#                 title="EspaÃ§amento vs Limiar de Tamanho",
#                 xaxis_title="Limiar de comprimento (m)",
#                 yaxis_title="EspaÃ§amento mÃ©dio (m)",
#                 xaxis_type="log",
#                 yaxis_type="log",
#                 hovermode='x unified'
#             )
            
#             st.plotly_chart(fig_spacing, width='stretch')
        
#         # ComparaÃ§Ã£o normalizada
#         st.divider()
#         st.subheader("ðŸ“Š ComparaÃ§Ã£o Normalizada")
        
#         # Obter o mÃ¡ximo apropriado
#         if st.session_state.framfrat_data is not None:
#             max_for_slider = st.session_state.framfrat_data['length'].quantile(0.5)
#         elif st.session_state.scanline_data is not None:
#             max_for_slider = st.session_state.scanline_data['length'].quantile(0.5)
#         else:
#             max_for_slider = 1.0
        
#         # Selecionar limiar comum
#         common_threshold = st.slider(
#             "Limiar comum de tamanho (m)",
#             min_value=float(l_min),
#             max_value=float(max_for_slider),
#             value=float(min(l_min * 10, max_for_slider)),
#             format="%.4f"
#         )
        
#         col1, col2, col3 = st.columns(3)
        
#         if st.session_state.framfrat_data is not None:
#             p10_f_common = analyzer.calculate_p10(
#                 st.session_state.framfrat_data,
#                 common_threshold,
#                 image_area
#             )
#             with col1:
#                 st.metric(
#                     "P10 FRAMFRAT",
#                     f"{p10_f_common:.3f} fraturas/m",
#                     f"EspaÃ§amento: {1/p10_f_common:.3f} m"
#                 )
        
#         if st.session_state.scanline_data is not None:
#             p10_s_common = analyzer.calculate_p10_scanline(
#                 st.session_state.scanline_data,
#                 common_threshold,
#                 scanline_length
#             )
#             with col2:
#                 st.metric(
#                     "P10 Scanline",
#                     f"{p10_s_common:.3f} fraturas/m",
#                     f"EspaÃ§amento: {1/p10_s_common:.3f} m"
#                 )
        
#         if st.session_state.framfrat_data is not None and st.session_state.scanline_data is not None:
#             ratio = p10_f_common / p10_s_common
#             with col3:
#                 st.metric(
#                     "RazÃ£o FRAMFRAT/Scanline",
#                     f"{ratio:.2f}",
#                     "Fator de intensificaÃ§Ã£o" if ratio > 1 else "Fator de reduÃ§Ã£o"
#                 )
#     else:
#         st.info("ðŸ“ Por favor, carregue os dados primeiro")

# # Tab 4: DFN 2D
# with tab4:
#     st.header("ðŸ—ºï¸ GeraÃ§Ã£o de DFN 2D")
    
#     if st.session_state.data_loaded and st.session_state.analysis_results:
#         # Obter Ã¡rea da imagem
#         if st.session_state.framfrat_data is not None:
#             image_area = st.session_state.framfrat_data.attrs.get('area', 1.0)
#             l_min = st.session_state.get('l_min_framfrat', 0.001)
#         else:
#             image_area = 1.0
#             l_min = 0.001
        
#         col1, col2 = st.columns([1, 2])
        
#         with col1:
#             st.subheader("ConfiguraÃ§Ãµes DFN 2D")
            
#             # Semente aleatÃ³ria
#             random_seed_2d = st.number_input(
#                 "ðŸŽ² Semente aleatÃ³ria", 
#                 min_value=0, 
#                 value=42,
#                 help="Para reprodutibilidade da geraÃ§Ã£o",
#                 key="seed_2d"
#             )
            
#             st.divider()
            
#             # DomÃ­nio
#             domain_width = st.number_input(
#                 "Largura do domÃ­nio (m)",
#                 min_value=0.1,
#                 value=float(np.sqrt(image_area)),
#                 step=0.1
#             )
            
#             domain_height = st.number_input(
#                 "Altura do domÃ­nio (m)",
#                 min_value=0.1,
#                 value=float(np.sqrt(image_area)),
#                 step=0.1
#             )
            
#             # NÃºmero de fraturas
#             n_fractures = st.number_input(
#                 "NÃºmero de fraturas",
#                 min_value=10,
#                 value=100,
#                 step=10,
#                 help="Baseado na intensidade P10"
#             )
            
#             # Usar parÃ¢metros ajustados
#             use_fitted = st.checkbox(
#                 "Usar parÃ¢metros ajustados",
#                 value=True,
#                 help="Usa os parÃ¢metros das leis de potÃªncia ajustadas"
#             )
            
#             # Controles de VisualizaÃ§Ã£o
#             st.divider()
#             st.subheader("Controles de VisualizaÃ§Ã£o")
            
#             fracture_shape_2d = st.selectbox(
#                 "Formato da Fratura",
#                 options=['lines', 'rectangles'],
#                 format_func=lambda x: {'lines': 'Linhas', 'rectangles': 'RetÃ¢ngulos'}.get(x, x),
#                 help="Escolha como representar as fraturas 2D. 'Discos' nÃ£o se aplica a DFN 2D."
#             )
            
#             show_centers_2d = st.checkbox(
#                 "Mostrar Centros das Fraturas",
#                 value=False,
#                 help="Exibe o ponto central de cada fratura com uma cor de destaque."
#             )
            
#             show_numbers_2d = st.checkbox(
#                 "Mostrar NumeraÃ§Ã£o das Fraturas",
#                 value=False,
#                 help="Exibe o nÃºmero de contagem prÃ³ximo ao centro de cada fratura."
#             )
            
#             # BotÃ£o de gerar
#             generate_2d = st.button(
#                 "ðŸŽ² Gerar DFN 2D",
#                 type="primary",
#                 width='stretch'
#             )
        
#         with col2:
#             if generate_2d:
#                 with st.spinner("Gerando DFN 2D..."):
#                     # Usar a semente especÃ­fica desta aba
#                     generator = DFNGenerator(random_seed_2d)
#                     viz = FractureVisualizer()
                    
#                     # Preparar parÃ¢metros
#                     if use_fitted and 'length_fit' in st.session_state.analysis_results:
#                         params = {
#                             'exponent': st.session_state.analysis_results['length_fit']['exponent'],
#                             'x_min': l_min,
#                             'coefficient': st.session_state.analysis_results['length_fit']['coefficient'],
#                         }
                        
#                         # Adicionar parÃ¢metros de abertura se disponÃ­veis
#                         if 'bl_relation' in st.session_state.analysis_results:
#                             params['g'] = st.session_state.analysis_results['bl_relation']['g']
#                             params['m'] = st.session_state.analysis_results['bl_relation']['m']
                        
#                         # Adicionar orientaÃ§Ã£o se disponÃ­vel
#                         if 'orientation' in st.session_state.framfrat_data.columns:
#                             orientations = st.session_state.framfrat_data['orientation'].values
#                             params['orientation_mean'] = np.mean(orientations)
#                             params['orientation_std'] = np.std(orientations)
#                     else:
#                         params = {
#                             'exponent': 2.0,
#                             'x_min': 0.01,
#                             'coefficient': 100
#                         }
                    
#                     # Gerar DFN
#                     dfn_2d = generator.generate_2d_dfn(
#                         params=params,
#                         domain_size=(domain_width, domain_height),
#                         n_fractures=n_fractures
#                     )
                    
#                            # Visualizar DFN
#                     fig_dfn = viz.plot_dfn_2d(
#                     dfn_2d,
#                     (domain_width, domain_height),
#                     fracture_shape=fracture_shape_2d,
#                     show_centers=show_centers_2d,
#                     show_numbers=show_numbers_2d
#                     )
                    
#                     st.plotly_chart(fig_dfn, width='stretch')
                    
#                     # Converter lista de fraturas para DataFrame para estatÃ­sticas
#                     dfn_df = pd.DataFrame([f.to_dict() for f in dfn_2d])
                    
#                     # EstatÃ­sticas do DFN
#                     st.divider()
#                     col1, col2, col3 = st.columns(3)
                    
#                     with col1:
#                         st.metric("Total de fraturas", len(dfn_2d))
#                         st.metric("Comprimento total (m)", f"{dfn_df['length'].sum():.2f}")
                    
#                     with col2:
#                         st.metric("Comprimento mÃ©dio (m)", f"{dfn_df['length'].mean():.3f}")
#                         st.metric("Abertura mÃ©dia (m)", f"{dfn_df['aperture'].mean():.4f}")
                    
#                     with col3:
#                         st.metric("P21 (m/mÂ²)", f"{dfn_df['length'].sum() / (domain_width * domain_height):.3f}")
#                         porosity = (dfn_df['aperture'] * dfn_df['length']).sum() / (domain_width * domain_height)
#                         st.metric("Porosidade (%)", f"{porosity * 100:.3f}")
                    
#                     # Salvar DFN gerado
#                     st.session_state.dfn_2d = dfn_2d
#     else:
#         st.info("ðŸ“ Por favor, complete as anÃ¡lises anteriores primeiro")



# # Tab 5: DFN 3D
# with tab5:
#     st.header("ðŸŽ² GeraÃ§Ã£o de DFN 3D")
    
#     if st.session_state.data_loaded and st.session_state.analysis_results:
#         # Obter l_min
#         l_min = st.session_state.get('l_min_framfrat', 0.001)
        
#         st.subheader("ConfiguraÃ§Ãµes DFN 3D")

#         col1, col2, col3 = st.columns(3) # DOMÃNIO 3D
#         domain_x = col1.number_input("DimensÃ£o X (m)", min_value=10.0, value=st.session_state.get('dfn_3d_domain', [100.0, 100.0, 20.0])[0], step=1.0)
#         domain_y = col2.number_input("DimensÃ£o Y (m)", min_value=10.0, value=st.session_state.get('dfn_3d_domain', [100.0, 100.0, 20.0])[1], step=1.0)    
#         domain_z = col3.number_input("DimensÃ£o Z (m)", min_value=5.0, value=st.session_state.get('dfn_3d_domain', [100.0, 100.0, 20.0])[2], step=1.0)

#         col_L, col_R = st.columns([1, 1], gap='large')

#         with col_L:
#             # OrientaÃ§Ã£o preferencial
#             st.divider()
#             st.write("**OrientaÃ§Ã£o Preferencial**")
#             col_left, col_mid= st.columns([1, 1], gap='large')
#             dip_mean = col_left.slider("Dip mÃ©dio (Â°)", min_value=0, max_value=90, value=45)
#             dip_dir_mean = col_mid.slider("Dip Direction mÃ©dio (Â°)", min_value=0, max_value=360, value=90)

#         with col_R:
#             st.divider()
#             st.write("**Mais configuraÃ§Ãµes**")
#             col_left, col_mid = st.columns([1, 1], gap='medium')

#             # Semente aleatÃ³ria
#             random_seed_3d = col_left.number_input(
#                 "ðŸŽ² Semente aleatÃ³ria", 
#                 min_value=0, 
#                 value=42,
#                 help="Para reprodutibilidade da geraÃ§Ã£o",
#                 key="seed_3d"
#             )
                                
#             # NÃºmero de fraturas
#             n_fractures_3d = col_mid.number_input("NÃºmero de fraturas 3D", min_value=10, value=200, step=10)
        
#         st.divider()
        
#         # ========== CONTROLES DE VISUALIZAÃ‡ÃƒO ==========
#         st.subheader('ðŸŽ›ï¸ Controles de VisualizaÃ§Ã£o')
        
#         # Inicializar estado se nÃ£o existir
#         if 'viz_mode' not in st.session_state:
#             st.session_state.viz_mode = 'ellipsoids'
#         if 'show_centers_3d' not in st.session_state:
#             st.session_state.show_centers_3d = False
#         if 'show_numbers_3d' not in st.session_state:
#             st.session_state.show_numbers_3d = False
#         if 'color_by_sets' not in st.session_state:
#             st.session_state.color_by_sets = False
#         if 'num_sets' not in st.session_state:
#             st.session_state.num_sets = None
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.write("**Tipo de visualizaÃ§Ã£o das fraturas**")
            
#             # ACTION: Radio buttons para escolher modo de visualizaÃ§Ã£o
#             viz_options = {
#                 'lines': 'ðŸ“ˆ Linhas',
#                 'rectangles': 'â¬œ RetÃ¢ngulos', 
#                 'ellipsoids': 'â­• ElipsÃ³ides'
#             }
            
#             viz_mode = st.radio(
#                 "Tipo de visualizaÃ§Ã£o",
#                 options=list(viz_options.keys()),
#                 format_func=lambda x: viz_options[x],
#                 index=list(viz_options.keys()).index(st.session_state.viz_mode),
#                 key='viz_mode_radio',
#                 label_visibility='collapsed'
#             )
            
#             # ACTION: Atualizar estado quando mudar
#             if viz_mode != st.session_state.viz_mode:
#                 st.session_state.viz_mode = viz_mode
        
#         with col2:
#             # ACTION: Checkbox para numeraÃ§Ã£o
#             show_numbers = st.checkbox(
#                 'ðŸ”¢ NumeraÃ§Ã£o das Fraturas',
#                 value=st.session_state.show_numbers_3d,
#                 help='Numerar as fraturas',
#                 key='show_numbers_checkbox'
#             )
#             if show_numbers != st.session_state.show_numbers_3d:
#                 st.session_state.show_numbers_3d = show_numbers
            
#             # ACTION: Checkbox para centros
#             show_centers = st.checkbox(
#                 'ðŸŽ¯ Centros das Fraturas',
#                 value=st.session_state.show_centers_3d,
#                 help='Mostrar os centros das fraturas',
#                 key='show_centers_checkbox'
#             )
#             if show_centers != st.session_state.show_centers_3d:
#                 st.session_state.show_centers_3d = show_centers
        
#         with col3:
#             # ACTION: Selectbox para nÃºmero de famÃ­lias
#             num_sets = st.selectbox(
#                 'NÃºmero de sets',
#                 options=[None, 1, 2, 3, 4],
#                 index=0,
#                 format_func=lambda x: 'NÃºmero de famÃ­lias' if x is None else str(x),
#                 help='NÃºmero de famÃ­lias das fraturas.',
#                 key='num_sets_select'
#             )
            
#             # ACTION: Ativar coloraÃ§Ã£o por famÃ­lia
#             if num_sets is not None:
#                 st.session_state.color_by_sets = True
#                 st.session_state.num_sets = num_sets
#             else:
#                 st.session_state.color_by_sets = False
#                 st.session_state.num_sets = None

#         st.markdown("")
#         st.markdown("")
        
#         # BotÃ£o de gerar
#         col_esq, col_dir = st.columns([1, 4], gap='large')
        
#         generate_3d = col_esq.button("ðŸŽ² Gerar DFN 3D", type="primary", key='btn_generate_3d')
        
#         # ========== LÃ“GICA DE GERAÃ‡ÃƒO ==========
#         if generate_3d:
#             with st.spinner("Gerando DFN 3D..."):
#                 generator = DFNGenerator(random_seed_3d)
                
#                 # Preparar parÃ¢metros
#                 if 'length_fit' in st.session_state.analysis_results:
#                     params_3d = {
#                         'exponent': st.session_state.analysis_results['length_fit']['exponent'],
#                         'x_min': l_min,
#                         'coefficient': st.session_state.analysis_results['length_fit']['coefficient'],
#                         'dip_mean': dip_mean,
#                         'dip_std': 10,
#                         'dip_dir_mean': dip_dir_mean,
#                         'dip_dir_std': 20
#                     }
                    
#                     if 'bl_relation' in st.session_state.analysis_results:
#                         params_3d['g'] = st.session_state.analysis_results['bl_relation']['g']
#                         params_3d['m'] = st.session_state.analysis_results['bl_relation']['m']
#                 else:
#                     params_3d = {
#                         'exponent': 2.0,
#                         'x_min': 0.01,
#                         'coefficient': 100,
#                         'dip_mean': dip_mean,
#                         'dip_dir_mean': dip_dir_mean
#                     }
                
#                 # Gerar DFN 3D
#                 dfn_3d = generator.generate_3d_dfn(
#                     params=params_3d,
#                     domain_size=(domain_x, domain_y, domain_z),
#                     n_fractures=n_fractures_3d
#                 )
                
#                 # Converter para DataFrame e adicionar famÃ­lia se necessÃ¡rio
#                 dfn_3d_df = pd.DataFrame([f.to_dict() for f in dfn_3d])
                
#                 # ACTION: Atribuir famÃ­lias aleatÃ³rias se coloraÃ§Ã£o por famÃ­lia ativada
#                 if st.session_state.color_by_sets and st.session_state.num_sets:
#                     np.random.seed(random_seed_3d)
#                     dfn_3d_df['family'] = np.random.randint(0, st.session_state.num_sets, len(dfn_3d_df))
                
#                 # Salvar no estado
#                 st.session_state.dfn_3d = dfn_3d
#                 st.session_state.dfn_3d_df = dfn_3d_df
#                 st.session_state.dfn_3d_domain = (domain_x, domain_y, domain_z)
                
#                 st.divider()
#                 st.success("âœ… DFN 3D gerado com sucesso!")

#         # ========== FUNÃ‡ÃƒO DE RENDERIZAÃ‡ÃƒO REATIVA ==========
#         def render_current_view():
#             """
#             ACTION: Renderiza a visualizaÃ§Ã£o 3D com base no estado atual.
#             Chamada automaticamente quando widgets mudam.
#             """
#             if 'dfn_3d_df' not in st.session_state or st.session_state.dfn_3d_df is None:
#                 st.info("âš ï¸ Clique no botÃ£o 'Gerar DFN 3D' para visualizar o grÃ¡fico.")
#                 return
            
#             viz = FractureVisualizer()
#             domain_size = st.session_state.dfn_3d_domain
            
#             with st.spinner("Atualizando visualizaÃ§Ã£o DFN 3D..."):
#                 # ACTION: Chamar plot_dfn_3d com parÃ¢metros do estado
#                 fig_dfn_3d = viz.plot_dfn_3d(
#                     fractures_df=st.session_state.dfn_3d_df,
#                     domain_size=domain_size,
#                     shape_mode=st.session_state.viz_mode,
#                     show_centers=st.session_state.show_centers_3d,
#                     show_numbers=st.session_state.show_numbers_3d,
#                     color_by_family=st.session_state.color_by_sets,
#                     family_col='family'
#                 )
                
#                 st.plotly_chart(fig_dfn_3d, width='stretch')
                
#                 # EstatÃ­sticas
#                 dfn_3d_df = st.session_state.dfn_3d_df
#                 dfn_3d_df['area'] = np.pi * dfn_3d_df['radius']**2

#                 st.divider()
#                 col1, col2, col3 = st.columns(3)
#                 volume = domain_size[0] * domain_size[1] * domain_size[2]

#                 with col1:
#                     st.metric("Total de fraturas", len(dfn_3d_df))
#                     st.metric("Ãrea total (mÂ²)", f"{dfn_3d_df['area'].sum():.2f}")
                
#                 with col2:
#                     st.metric("P32 (mÂ²/mÂ³)", f"{dfn_3d_df['area'].sum() / volume:.3f}")
#                     st.metric("Abertura mÃ©dia (mm)", f"{dfn_3d_df['aperture'].mean() * 1000:.2f}")
                
#                 with col3:
#                     porosity_3d = (dfn_3d_df['aperture'] * dfn_3d_df['area']).sum() / volume
#                     st.metric("Porosidade 3D (%)", f'{porosity_3d * 100:.3f}')
#                     k_estimate = (dfn_3d_df['aperture']**3).mean() / 12
#                     st.metric("Permeabilidade (mD)", f"{k_estimate * 1e12:.2f}", 
#                                 help="Estimativa simplificada de permeabilidade (k = bÂ³/12)")
        
#         # ACTION: Renderizar visualizaÃ§Ã£o (reativo aos widgets)
#         render_current_view()
            
#     else:
#         st.info("ðŸ“‹ Por favor, complete as anÃ¡lises anteriores primeiro")


# # Tab 6: Exportar
# with tab6:
#     st.header("ðŸ’¾ ExportaÃ§Ã£o de Resultados")
    
#     if st.session_state.data_loaded:
#         exporter = ResultsExporter()
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.subheader("ðŸ“Š Dados Processados")
            
#             # Exportar dados tratados
#             if st.button("ðŸ“¥ Exportar Dados Tratados (CSV)"):
#                 if st.session_state.framfrat_data is not None:
#                     csv_data = exporter.export_to_csv(st.session_state.framfrat_data)
#                     st.download_button(
#                         label="Download FRAMFRAT CSV",
#                         data=csv_data,
#                         file_name=f"framfrat_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                         mime="text/csv"
#                     )
                
#                 if st.session_state.scanline_data is not None:
#                     csv_scanline = exporter.export_to_csv(st.session_state.scanline_data)
#                     st.download_button(
#                         label="Download Scanline CSV",
#                         data=csv_scanline,
#                         file_name=f"scanline_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
#                         mime="text/csv"
#                     )
            
#             # Exportar parÃ¢metros ajustados
#             if st.button("ðŸ“Š Exportar ParÃ¢metros (JSON)"):
#                 if st.session_state.analysis_results:
#                     json_params = exporter.export_parameters(st.session_state.analysis_results)
#                     st.download_button(
#                         label="Download ParÃ¢metros JSON",
#                         data=json_params,
#                         file_name=f"parameters_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                         mime="application/json"
#                     )
        
#         with col2:
#             st.subheader("ðŸ—ºï¸ Modelos DFN")
            
#             # Exportar DFN 2D
#             if hasattr(st.session_state, 'dfn_2d'):
#                 if st.button("ðŸ“¥ Exportar DFN 2D (GeoJSON)"):
#                     geojson_data = exporter.export_dfn_2d_geojson(st.session_state.dfn_2d)
#                     st.download_button(
#                         label="Download DFN 2D GeoJSON",
#                         data=geojson_data,
#                         file_name=f"dfn_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.geojson",
#                         mime="application/geo+json"
#                     )
            
#             # Exportar DFN 3D
#             if hasattr(st.session_state, 'dfn_3d'):
#                 if st.button("ðŸ“¥ Exportar DFN 3D (VTK)"):
#                     vtk_data = exporter.export_dfn_3d_vtk(st.session_state.dfn_3d)
#                     st.download_button(
#                         label="Download DFN 3D VTK",
#                         data=vtk_data,
#                         file_name=f"dfn_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vtk",
#                         mime="application/x-vtk"
#                     )
        
#         # RelatÃ³rio completo
#         st.divider()
#         st.subheader("ðŸ“„ RelatÃ³rio Completo")
        
#         if st.button("ðŸ“‹ Gerar RelatÃ³rio Completo (Excel)", type="primary"):
#             with st.spinner("Gerando relatÃ³rio..."):
#                 # Coletar metadados
#                 metadata = {}
                
#                 if st.session_state.framfrat_data is not None:
#                     metadata['image_area'] = st.session_state.framfrat_data.attrs.get('area', 1.0)
#                     metadata['pixel_scale'] = st.session_state.framfrat_data.attrs.get('scale', 100.0)
#                     metadata['l_min'] = st.session_state.get('l_min_framfrat', 0.001)
#                     metadata['b_min'] = st.session_state.get('b_min_framfrat', 0.0001)
                
#                 if st.session_state.scanline_data is not None:
#                     metadata['scanline_length'] = st.session_state.scanline_data.attrs.get('scanline_length', 10.0)
#                     metadata['l_min_scan'] = st.session_state.get('l_min_scanline', 0.001)
#                     metadata['b_min_scan'] = st.session_state.get('b_min_scanline', 0.0001)
                
#                 excel_data = exporter.generate_full_report(
#                     st.session_state.framfrat_data,
#                     st.session_state.scanline_data,
#                     st.session_state.analysis_results,
#                     metadata
#                 )
                
#                 st.download_button(
#                     label="ðŸ“¥ Download RelatÃ³rio Excel",
#                     data=excel_data,
#                     file_name=f"fracture_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
#                     mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                 )
        
#         # Salvar/Carregar sessÃ£o
#         st.divider()
#         st.subheader("ðŸ’¼ Gerenciar SessÃ£o")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             if st.button("ðŸ’¾ Salvar SessÃ£o"):
#                 session_data = exporter.save_session(st.session_state)
#                 st.download_button(
#                     label="Download SessÃ£o",
#                     data=session_data,
#                     file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
#                     mime="application/json"
#                 )
        
#         with col2:
#             uploaded_session = st.file_uploader("Carregar SessÃ£o", type=['json'], key="session_upload")
#             if uploaded_session and st.button("ðŸ“‚ Restaurar SessÃ£o"):
#                 exporter.load_session(uploaded_session, st.session_state)
#                 st.success("âœ… SessÃ£o restaurada!")
#                 st.rerun()
#     else:
#         st.info("ðŸ“ Por favor, carregue os dados primeiro")

# # RodapÃ© com referÃªncias
# st.markdown("""
# ---
# ### ðŸ“š ReferÃªncias CientÃ­ficas

# - **Marrett, R.** (1996). Aggregate properties of fracture populations. *Journal of Structural Geology*, 18(2-3), 169-178.
# - **Ortega, O.J., Marrett, R.A., & Laubach, S.E.** (2006). A scale-independent approach to fracture intensity and average spacing measurement. *AAPG Bulletin*, 90(2), 193-208.

# âš ï¸ **ObservaÃ§Ãµes importantes:**
# - A Ã¡rea da imagem (FRAMFRAT) Ã© crucial para normalizaÃ§Ã£o correta das densidades
# - O comprimento da scanline Ã© fundamental para cÃ¡lculo de P10
# - ComparaÃ§Ãµes entre fontes requerem limiar comum de tamanho (Ortega et al., 2006)
# """)