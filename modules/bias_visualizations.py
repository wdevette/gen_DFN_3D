# modules/bias_visualizations.py
"""
Visualizações diagnósticas para correções de vieses
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional

def plot_correction_pipeline(results_dict: Dict) -> go.Figure:
    """
    Visualiza o pipeline completo de correções
    
    Mostra o efeito cumulativo de cada correção aplicada
    """
    corrections = results_dict.get('corrections', [])
    
    if not corrections:
        return None
    
    # Extrair dados
    steps = []
    intensities = []
    factors = []
    
    # Primeiro ponto: dados brutos
    first_corr = corrections[0]
    steps.append("Raw Data")
    intensities.append(first_corr.intensity_observed)
    factors.append(1.0)
    
    # Adicionar cada correção
    for corr in corrections:
        steps.append(corr.method.split('(')[0].strip())
        intensities.append(corr.intensity_corrected)
        factors.append(corr.correction_factor)
    
    # Criar figura com subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Intensidade ao Longo do Pipeline', 'Fatores de Correção'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Plot 1: Intensidade
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=intensities,
            mode='lines+markers',
            name='Intensidade',
            line=dict(color='#2c7be5', width=3),
            marker=dict(size=10, color='#2c7be5'),
            hovertemplate='<b>%{x}</b><br>Intensidade: %{y:.3f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot 2: Fatores
    fig.add_trace(
        go.Bar(
            x=steps[1:],  # Excluir "Raw Data"
            y=factors[1:],
            name='Fator',
            marker=dict(
                color=factors[1:],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Fator", y=0.2, len=0.4)
            ),
            hovertemplate='<b>%{x}</b><br>Fator: %{y:.2f}×<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Adicionar linha horizontal em 1.0 no segundo plot
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                  row=2, col=1, annotation_text="Sem correção")
    
    # Layout
    fig.update_xaxes(title_text="Etapa do Pipeline", row=2, col=1)
    fig.update_yaxes(title_text="Intensidade", row=1, col=1)
    fig.update_yaxes(title_text="Fator de Correção", row=2, col=1)
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Pipeline de Correções de Vieses",
        title_x=0.5,
        hovermode='x unified'
    )
    
    return fig


def plot_terzaghi_weights(terzaghi_result) -> go.Figure:
    """
    Visualiza distribuição de pesos de Terzaghi
    """
    if 'data_corrected' not in terzaghi_result.validation:
        return None
    
    data = terzaghi_result.validation['data_corrected']
    
    # Criar figura
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Distribuição de Ângulos θ', 'Distribuição de Pesos'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}]]
    )
    
    # Plot 1: Pesos vs Ângulos
    fig.add_trace(
        go.Scatter(
            x=data['theta_deg'],
            y=data['terzaghi_weight'],
            mode='markers',
            marker=dict(
                size=8,
                color=data['terzaghi_weight'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Peso", x=0.45)
            ),
            name='Fraturas',
            hovertemplate='<b>θ: %{x:.1f}°</b><br>Peso: %{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Plot 2: Histograma de pesos
    fig.add_trace(
        go.Histogram(
            x=data['terzaghi_weight'],
            nbinsx=30,
            marker=dict(color='#2c7be5'),
            name='Frequência',
            hovertemplate='Peso: %{x:.2f}<br>Contagem: %{y}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Layout
    fig.update_xaxes(title_text="Ângulo θ (graus)", row=1, col=1)
    fig.update_yaxes(title_text="Peso de Terzaghi", row=1, col=1)
    fig.update_xaxes(title_text="Peso de Terzaghi", row=1, col=2)
    fig.update_yaxes(title_text="Frequência", row=1, col=2)
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Análise de Pesos de Terzaghi (1965)",
        title_x=0.5
    )
    
    return fig


def plot_dl_scaling_validation(dl_result) -> go.Figure:
    """
    Visualiza resultados da validação D-L Scaling
    """
    if 'data_filtered' not in dl_result.validation:
        return None
    
    data_valid = dl_result.validation['data_filtered']
    
    if 'data_removed' in dl_result.validation:
        data_removed = dl_result.validation['data_removed']
    else:
        data_removed = pd.DataFrame()
    
    # Criar figura
    fig = go.Figure()
    
    # Dados válidos
    if len(data_valid) > 0:
        fig.add_trace(
            go.Scatter(
                x=data_valid['length_m'] if 'length_m' in data_valid.columns else data_valid.index,
                y=data_valid['aperture_m'] if 'aperture_m' in data_valid.columns else data_valid['aperture'],
                mode='markers',
                marker=dict(size=8, color='green', symbol='circle'),
                name='Dados Válidos',
                hovertemplate='<b>Válido</b><br>L: %{x:.3f} m<br>b: %{y:.4f} m<extra></extra>'
            )
        )
    
    # Dados removidos
    if len(data_removed) > 0:
        fig.add_trace(
            go.Scatter(
                x=data_removed['length_m'] if 'length_m' in data_removed.columns else data_removed.index,
                y=data_removed['aperture_m'] if 'aperture_m' in data_removed.columns else data_removed['aperture'],
                mode='markers',
                marker=dict(size=8, color='red', symbol='x'),
                name='Dados Removidos',
                hovertemplate='<b>Removido</b><br>L: %{x:.3f} m<br>b: %{y:.4f} m<extra></extra>'
            )
        )
    
    # Linha teórica D-L: b = A × L^0.5
    A = dl_result.parameters['A_constant']
    L_range = np.logspace(np.log10(0.01), np.log10(10), 100)
    b_expected = A * np.sqrt(L_range)
    
    fig.add_trace(
        go.Scatter(
            x=L_range,
            y=b_expected,
            mode='lines',
            line=dict(color='blue', width=2, dash='dash'),
            name=f'D-L Teórico (A={A})',
            hovertemplate='<b>Teórico</b><br>L: %{x:.3f} m<br>b: %{y:.4f} m<extra></extra>'
        )
    )
    
    # Bounds de tolerância
    tolerance = dl_result.parameters['tolerance']
    fig.add_trace(
        go.Scatter(
            x=L_range,
            y=b_expected * tolerance,
            mode='lines',
            line=dict(color='orange', width=1, dash='dot'),
            name=f'Limite Superior ({tolerance}×)',
            hoverinfo='skip'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=L_range,
            y=b_expected / tolerance,
            mode='lines',
            line=dict(color='orange', width=1, dash='dot'),
            name=f'Limite Inferior (1/{tolerance}×)',
            fill='tonexty',
            fillcolor='rgba(255, 165, 0, 0.1)',
            hoverinfo='skip'
        )
    )
    
    # Layout
    fig.update_xaxes(type="log", title_text="Comprimento L (m)")
    fig.update_yaxes(type="log", title_text="Abertura b (m)")
    
    fig.update_layout(
        height=500,
        title_text=f"Validação D-L Scaling ({dl_result.validation['removal_percent']:.1f}% removido)",
        title_x=0.5,
        hovermode='closest',
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig


def plot_powerlaw_fit(lengths: np.ndarray, alpha: float, l_min: float, l_max: float) -> go.Figure:
    """
    Visualiza ajuste power-law com dados observados
    """
    # Filtrar dados
    lengths_clean = lengths[(lengths >= l_min) & (lengths <= l_max)]
    
    # Distribuição cumulativa complementar
    sorted_lengths = np.sort(lengths_clean)[::-1]
    n = len(sorted_lengths)
    cumulative = np.arange(1, n + 1)
    
    # Linha teórica
    L_theory = np.logspace(np.log10(l_min), np.log10(l_max), 100)
    N_theory = n * (L_theory / l_min)**(-alpha)
    
    # Criar figura
    fig = go.Figure()
    
    # Dados observados
    fig.add_trace(
        go.Scatter(
            x=sorted_lengths,
            y=cumulative,
            mode='markers',
            marker=dict(size=6, color='#2c7be5'),
            name='Dados Observados',
            hovertemplate='<b>Observado</b><br>L: %{x:.3f} m<br>N(≥L): %{y}<extra></extra>'
        )
    )
    
    # Ajuste teórico
    fig.add_trace(
        go.Scatter(
            x=L_theory,
            y=N_theory,
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Power-law: α={alpha:.2f}',
            hovertemplate='<b>Teórico</b><br>L: %{x:.3f} m<br>N(≥L): %{y:.1f}<extra></extra>'
        )
    )
    
    # Marcadores de l_min e l_max
    fig.add_vline(x=l_min, line_dash="dash", line_color="green",
                  annotation_text=f"l_min={l_min:.3f}m")
    fig.add_vline(x=l_max, line_dash="dash", line_color="orange",
                  annotation_text=f"l_max={l_max:.3f}m")
    
    # Layout
    fig.update_xaxes(type="log", title_text="Comprimento L (m)")
    fig.update_yaxes(type="log", title_text="N(L ≥ x)")
    
    fig.update_layout(
        height=500,
        title_text=f"Ajuste Power-Law: N(L) ∝ L^(-{alpha:.2f})",
        title_x=0.5,
        hovermode='closest',
        legend=dict(x=0.02, y=0.02)
    )
    
    return fig


def plot_marrett_factor_sensitivity(alpha_range: np.ndarray, l_min: float, l_max: float) -> go.Figure:
    """
    Visualiza sensibilidade do fator de Marrett a α
    """
    factors = []
    
    for alpha in alpha_range:
        if np.abs(alpha - 1.0) < 1e-6:
            f = np.log(l_max / l_min)
        else:
            num = l_max**(1 - alpha) - l_min**(1 - alpha)
            den = (1 - alpha) * l_max**(1 - alpha)
            f = num / den
        factors.append(f)
    
    # Criar figura
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=alpha_range,
            y=factors,
            mode='lines',
            line=dict(color='#2c7be5', width=3),
            fill='tozeroy',
            fillcolor='rgba(44, 123, 229, 0.2)',
            hovertemplate='<b>α: %{x:.2f}</b><br>Fator Marrett: %{y:.2f}×<extra></extra>'
        )
    )
    
    # Região aceitável (α entre 1.2 e 2.8)
    fig.add_vrect(
        x0=1.2, x1=2.8,
        fillcolor="green", opacity=0.1,
        layer="below", line_width=0,
        annotation_text="Range Aceitável",
        annotation_position="top left"
    )
    
    # Layout
    fig.update_xaxes(title_text="Expoente α")
    fig.update_yaxes(title_text="Fator de Correção de Marrett")
    
    fig.update_layout(
        height=400,
        title_text=f"Sensibilidade do Fator de Marrett (l_min={l_min:.3f}m, l_max={l_max:.3f}m)",
        title_x=0.5,
        hovermode='x'
    )
    
    return fig


def display_correction_summary(results_dict: Dict):
    """
    Exibe resumo formatado das correções aplicadas
    """
    st.markdown("### 📊 Resumo das Correções Aplicadas")
    
    if 'summary' not in results_dict:
        st.warning("Nenhuma correção aplicada ainda")
        return
    
    summary = results_dict['summary']
    corrections = results_dict.get('corrections', [])

    
    # Criar colunas para métricas
    cols = st.columns(len(corrections) + 1)
    
    # Primeira coluna: Dados brutos
    with cols[0]:
        st.metric(
            "📥 Dados Brutos",
            f"{corrections[0].intensity_observed:.2f}".replace(".", ",") if corrections else "N/A",
            help="Intensidade observada sem correções"
        )
    
    # Colunas subsequentes: Cada correção
    for i, corr in enumerate(corrections):
        with cols[i + 1]:
            increase = (corr.correction_factor - 1) * 100
            st.metric(
                f"🔧 {corr.method.split('(')[0].strip()}",
                f"{corr.intensity_corrected:.2f}".replace(".", ","),
                f"+{increase:.1f}%".replace(".", ","),
                delta_color="normal",
                help=f"Fator: {corr.correction_factor:.2f}×".replace(".", ",")
            )
    
    # Resumo final
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "📈 Aumento Total",
            f"{summary.get('total_increase_factor', 0):.2f}×".replace(".", ","),
            f"+{summary.get('total_increase_percent', 0):.1f}%".replace(".", ",")
        )
    
    with col2:
        if 'P10_final' in summary:
            st.metric("P10 Final", f"{summary['P10_final']:.2f} frat/m".replace(".", ","))
        elif 'P21_final' in summary:
            st.metric("P21 Final", f"{summary['P21_final']:.2f} m/m²".replace(".", ","))
    
    with col3:
        if 'P30_estimated' in summary:
            st.metric("P30 Estimado", f"{summary['P30_estimated']:.2f} frat/m³".replace(".", ","))
    
    # Detalhes de cada correção
    with st.expander("📋 Detalhes das Correções"):
        for i, corr in enumerate(corrections):
            st.markdown(f"**{i+1}. {corr.method}**")
            st.text(corr.summary())
            st.markdown("")


def plot_before_after_comparison(data_before: pd.DataFrame, 
                                 data_after: pd.DataFrame,
                                 length_col: str = 'length_m') -> go.Figure:
    """
    Compara distribuições antes e depois das correções
    """
    fig = go.Figure()
    
    # Antes
    fig.add_trace(
        go.Histogram(
            x=data_before[length_col],
            name='Antes',
            opacity=0.7,
            marker=dict(color='red'),
            hovertemplate='<b>Antes</b><br>L: %{x:.3f} m<br>Count: %{y}<extra></extra>'
        )
    )
    
    # Depois
    fig.add_trace(
        go.Histogram(
            x=data_after[length_col],
            name='Depois',
            opacity=0.7,
            marker=dict(color='green'),
            hovertemplate='<b>Depois</b><br>L: %{x:.3f} m<br>Count: %{y}<extra></extra>'
        )
    )
    
    # Layout
    fig.update_xaxes(title_text="Comprimento (m)")
    fig.update_yaxes(title_text="Frequência")
    
    fig.update_layout(
        barmode='overlay',
        height=400,
        title_text="Distribuição de Comprimentos: Antes vs Depois",
        title_x=0.5,
        legend=dict(x=0.8, y=0.95)
    )
    
    return fig
