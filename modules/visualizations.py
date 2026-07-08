"""
visualizations.py - VERSÃO CORRIGIDA v2

CORREÇÕES IMPLEMENTADAS:
1. CORRIGIDO subsampling que perdia a coluna 'family'
2. Removida opção "retângulos" - mantidas apenas "linhas" e "elipsóides"
3. CORRIGIDAS cores por família - cada família com cor distinta
4. CORRIGIDO modo "ellipsoids" que mostrava linhas
5. ADICIONADO hover detalhado com todas as informações da fratura
6. AUMENTADO tamanho dos pontos de centro (2 → 8)
7. OTIMIZAÇÃO: Um trace por família para performance
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class FractureVisualizer:
    """Visualizador de fraturas e análises - VERSÃO CORRIGIDA v2"""
    
    def __init__(self, style: str = 'scientific'):
        self.style = style
        
        # Paleta de cores DISTINTAS para famílias (mais vibrantes)
        self.family_colors = [
            '#E74C3C',  # Vermelho
            '#3498DB',  # Azul
            '#2ECC71',  # Verde
            '#F39C12',  # Laranja
            '#9B59B6',  # Roxo
            '#1ABC9C',  # Turquesa
            '#E91E63',  # Rosa
            '#00BCD4',  # Ciano
            '#FF5722',  # Laranja escuro
            '#607D8B',  # Cinza azulado
        ]
        
        self.colors = px.colors.qualitative.Set2
        
        if style == 'scientific':
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
                sns.set_palette("husl")
            except:
                pass
    
    def plot_power_law_fit(self, data: np.ndarray, fit_params: Dict) -> go.Figure:
        """Plota ajuste de lei de potência"""
        sorted_data = np.sort(data)[::-1]
        n = len(sorted_data)
        cumulative = np.arange(1, n + 1)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=sorted_data,
            y=cumulative,
            mode='markers',
            name='Dados observados',
            marker=dict(size=6, color='rgba(31,119,180,1)', symbol='circle'),
           # marker=dict(size=6, symbol='circle'),
            hovertemplate='Tamanho: %{x:.3f}<br>N(≥x): %{y}<extra></extra>'
        ))
        
        x_fit = np.logspace(
            np.log10(fit_params['x_min']),
            np.log10(sorted_data[0]),
            100
        )
        y_fit = fit_params['coefficient'] * x_fit**(-fit_params['exponent'])
        
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name=f"Ajuste: N = {fit_params['coefficient']:.1f} × x^(-{fit_params['exponent']:.2f})",
            line=dict(color='rgba(214,39,40,1)', width=2),
            hovertemplate='Tamanho: %{x:.3f}<br>N(≥x): %{y:.1f}<extra></extra>'
        ))
        
        fig.add_vline(
            x=fit_params['x_min'],
            line_dash="dash",
            line_color="rgba(44,160,44,1)",
            annotation_text=f"x_min = {fit_params['x_min']:.3f}"
        )
        
        fig.update_layout(
            title={'text': 'Distribuição Power-Law de Tamanhos', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title='Tamanho (m)',
            yaxis_title='N(≥x) - Número cumulativo',
            xaxis_type='log',
            yaxis_type='log',
            showlegend=True,
            legend=dict(x=0.6, y=0.95),
            template=None,
            paper_bgcolor='rgba(0,0,0,0)',
            #plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='rgba(200,200,200,0.9)', gridcolor='rgba(150,150,150,0.2)'),
            yaxis=dict(color='rgba(200,200,200,0.9)', gridcolor='rgba(150,150,150,0.2)'),
            hovermode='x unified'
        )
        
        if 'r_squared' in fit_params:
            annotation_text = f"R² = {fit_params['r_squared']:.3f}"
        else:
            annotation_text = f"KS = {fit_params.get('ks_statistic', 0):.3f}"
        
        fig.add_annotation(
            x=0.95, y=0.05,
            xref="paper", yref="paper",
            text=annotation_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    def plot_dfn_2d(self, fractures: List, domain_size: tuple, 
                    show_centers: bool = False, 
                    show_numbers: bool = False,
                    color_by_family: bool = False) -> go.Figure:
        """Visualiza DFN 2D com suporte a coloração por família"""
        fig = go.Figure()
        
        width, height = domain_size
        plotted_families = set()
        
        for i, frac in enumerate(fractures):
            if color_by_family and hasattr(frac, 'family'):
                color = self.family_colors[frac.family % len(self.family_colors)]
                family_label = f"Fam. {frac.family + 1}"
            else:
                color = '#34495E'
                family_label = ""
            
            show_legend = False
            if color_by_family and family_label and family_label not in plotted_families:
                show_legend = True
                plotted_families.add(family_label)
            
            fig.add_trace(go.Scatter(
                x=[frac.x1, frac.x2],
                y=[frac.y1, frac.y2],
                mode='lines',
                line=dict(color=color, width=max(2, frac.aperture * 0.5)),
                name=family_label if family_label else None,
                showlegend=show_legend,
                legendgroup=family_label,
                hovertemplate=(
                    f'<b>Fratura {i+1}</b><br>'
                    f'{family_label}<br>' if family_label else '' +
                    f'Comprimento: {frac.length:.3f} m<br>'
                    f'Abertura: {frac.aperture*1000:.2f} mm<br>'
                    f'Orientação: {frac.orientation:.1f}°<extra></extra>'
                )
            ))
            
            if show_centers:
                cx = (frac.x1 + frac.x2) / 2
                cy = (frac.y1 + frac.y2) / 2
                fig.add_trace(go.Scatter(
                    x=[cx], y=[cy],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='circle'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=width, y1=height,
            line=dict(color="red", width=2, dash="dash"),
            fillcolor="rgba(255,255,255,0)"
        )
        
        fig.update_layout(
            title='Rede de Fraturas Discretas 2D',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            xaxis=dict(scaleanchor="y", scaleratio=1, range=[-width*0.1, width*1.1]),
            yaxis=dict(range=[-height*0.1, height*1.1]),
            template=None,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode='closest'
        )
        
        return fig
    
    def plot_dfn_3d(
        self,
        fractures_df: pd.DataFrame,
        domain_size: tuple,
        shape_mode: str = 'lines',
        show_centers: bool = False,
        show_numbers: bool = False,
        color_by_family: bool = True,
        family_col: str = 'family',
        figure=None,
        n_fractures_theoretical: int = None,
        family_weights: dict = None
    ):
        """
        Visualiza DFN 3D - VERSÃO CORRIGIDA v3
        
        CORREÇÕES v3:
        - NOVO: Suporta n_fractures_theoretical para mostrar número teórico na legenda
        - NOVO: family_weights para calcular distribuição teórica por família
        - CORRIGIDO: Subsampling agora preserva a coluna 'family'
        - Cores distintas por família funcionando corretamente
        """
        
        if figure is None:
            fig = go.Figure()
        else:
            fig = figure
        
        width, height, depth = domain_size
        
        # ================================================================
        # SUBSAMPLING (máximo para visualização)
        # ================================================================
        MAX_FRACTURES_VIZ = 5000
        n_total = len(fractures_df)
        subsampled = False
        
        # Usar número teórico se fornecido
        n_theoretical = n_fractures_theoretical if n_fractures_theoretical else n_total
        
        # Garantir que a coluna family existe
        if family_col not in fractures_df.columns:
            fractures_df = fractures_df.copy()
            fractures_df[family_col] = 0
            print(f"⚠️ Coluna '{family_col}' não encontrada. Usando família 0 para todas.")
        
        # Debug: verificar distribuição antes do subsampling
        print(f"Distribuição de famílias ANTES do subsampling:")
        family_counts_before = fractures_df[family_col].value_counts().sort_index()
        for fam_id, count in family_counts_before.items():
            print(f"  Família {fam_id}: {count} fraturas")
        
        if n_total > MAX_FRACTURES_VIZ:
            subsampled = True
            frac_ratio = MAX_FRACTURES_VIZ / n_total
            
            # CORREÇÃO: Preservar a coluna family durante o subsampling
            # Método: Amostragem estratificada por família
            sampled_dfs = []
            
            for family_id in fractures_df[family_col].unique():
                family_mask = fractures_df[family_col] == family_id
                family_df = fractures_df[family_mask]
                n_sample = max(1, int(len(family_df) * frac_ratio))
                
                if len(family_df) <= n_sample:
                    sampled_dfs.append(family_df)
                else:
                    sampled_df = family_df.sample(n=n_sample, random_state=42)
                    sampled_dfs.append(sampled_df)
            
            fractures_df = pd.concat(sampled_dfs, ignore_index=True)
            
            print(f"Subsampling: {n_total} → {len(fractures_df)} fraturas")
        
        n_viz = len(fractures_df)
        
        # Debug: verificar distribuição DEPOIS do subsampling
        print(f"Distribuição de famílias DEPOIS do subsampling:")
        family_counts_after = fractures_df[family_col].value_counts().sort_index()
        for fam_id, count in family_counts_after.items():
            print(f"  Família {fam_id}: {count} fraturas")
        
        # ================================================================
        # PROCESSAMENTO VETORIZADO
        # ================================================================
        centers = np.array(fractures_df['center'].tolist())
        normals = np.array(fractures_df['normal'].tolist())
        radii = fractures_df['radius'].values
        apertures = fractures_df['aperture'].values
        dips = fractures_df['dip'].values
        dip_dirs = fractures_df['dip_direction'].values
        
        # Calcular vetores v1 e v2 para todas as fraturas
        z_axis = np.array([0, 0, 1])
        x_axis = np.array([1, 0, 0])
        
        vertical_mask = np.abs(normals[:, 2]) >= 0.99
        
        v1 = np.zeros_like(normals)
        v1[~vertical_mask] = np.cross(normals[~vertical_mask], z_axis)
        v1[vertical_mask] = np.cross(normals[vertical_mask], x_axis)
        
        v1_norms = np.linalg.norm(v1, axis=1, keepdims=True)
        v1_norms[v1_norms == 0] = 1
        v1 = v1 / v1_norms
        
        v2 = np.cross(normals, v1)
        
        # Extremos das linhas
        p1 = centers - radii[:, np.newaxis] * v1
        p2 = centers + radii[:, np.newaxis] * v1
        
        families = sorted(fractures_df[family_col].unique())
        print(f"Famílias a renderizar: {families}")
        
        # ================================================================
        # RENDERIZAR POR FAMÍLIA COM CORES DISTINTAS
        # ================================================================
        
        if shape_mode == 'lines':
            # ========== MODO LINHAS (otimizado) ==========
            for family_id in families:
                mask = fractures_df[family_col].values == family_id
                n_fam = mask.sum()
                
                if n_fam == 0:
                    print(f"  Família {family_id}: 0 fraturas (pulando)")
                    continue
                
                print(f"  Renderizando Família {family_id}: {n_fam} fraturas")
                
                p1_fam = p1[mask]
                p2_fam = p2[mask]
                centers_fam = centers[mask]
                radii_fam = radii[mask]
                apertures_fam = apertures[mask]
                dips_fam = dips[mask]
                dip_dirs_fam = dip_dirs[mask]
                
                # Criar arrays de coordenadas com NaN como separador
                x_coords = np.empty(n_fam * 3)
                y_coords = np.empty(n_fam * 3)
                z_coords = np.empty(n_fam * 3)
                
                x_coords[0::3] = p1_fam[:, 0]
                x_coords[1::3] = p2_fam[:, 0]
                x_coords[2::3] = np.nan
                
                y_coords[0::3] = p1_fam[:, 1]
                y_coords[1::3] = p2_fam[:, 1]
                y_coords[2::3] = np.nan
                
                z_coords[0::3] = p1_fam[:, 2]
                z_coords[1::3] = p2_fam[:, 2]
                z_coords[2::3] = np.nan
                
                # Criar texto de hover para cada fratura
                hover_texts = []
                for i in range(n_fam):
                    hover_texts.extend([
                        f"<b>Família {int(family_id)+1}</b><br>"
                        f"<b>Centro:</b> ({centers_fam[i,0]:.2f}, {centers_fam[i,1]:.2f}, {centers_fam[i,2]:.2f}) m<br>"
                        f"<b>Raio:</b> {radii_fam[i]:.3f} m<br>"
                        f"<b>Comprimento:</b> {2*radii_fam[i]:.3f} m<br>"
                        f"<b>Abertura:</b> {apertures_fam[i]*1000:.3f} mm<br>"
                        f"<b>Dip:</b> {dips_fam[i]:.1f}°<br>"
                        f"<b>Dip Dir:</b> {dip_dirs_fam[i]:.1f}°",
                        "",
                        ""
                    ])
                
                color = self.family_colors[int(family_id) % len(self.family_colors)]
                
                # Calcular número teórico de fraturas para esta família
                if family_weights and family_id in family_weights:
                    n_theoretical_fam = int(n_theoretical * family_weights[family_id])
                else:
                    n_theoretical_fam = int(n_theoretical * (n_fam / n_total)) if n_total > 0 else n_fam
                
                fig.add_trace(go.Scatter3d(
                    x=x_coords,
                    y=y_coords,
                    z=z_coords,
                    mode='lines',
                    line=dict(color=color, width=3),
                    name=f'Família {int(family_id)+1} ({n_theoretical_fam})',
                    showlegend=True,
                    text=hover_texts,
                    hoverinfo='text',
                    connectgaps=False
                ))
        
        elif shape_mode == 'ellipsoids':
            # ========== MODO ELIPSÓIDES (DISCOS) - CORRIGIDO ==========
            n_circle_points = 16  # Pontos para formar o círculo
            
            for family_id in families:
                mask = fractures_df[family_col].values == family_id
                indices = np.where(mask)[0]
                n_fam = len(indices)
                
                if n_fam == 0:
                    print(f"  Família {family_id}: 0 fraturas (pulando)")
                    continue
                
                print(f"  Renderizando Família {family_id} (elipsóides): {n_fam} fraturas")
                
                color = self.family_colors[int(family_id) % len(self.family_colors)]
                
                # Calcular número teórico de fraturas para esta família
                if family_weights and family_id in family_weights:
                    n_theoretical_fam = int(n_theoretical * family_weights[family_id])
                else:
                    n_theoretical_fam = int(n_theoretical * (n_fam / n_total)) if n_total > 0 else n_fam
                
                # Limitar número de discos para performance
                max_disks = min(n_fam, 500)
                if n_fam > max_disks:
                    indices = np.random.choice(indices, max_disks, replace=False)
                
                # Para cada fratura, criar disco
                for idx_count, idx in enumerate(indices):
                    center = centers[idx]
                    radius = radii[idx]
                    v1_i = v1[idx]
                    v2_i = v2[idx]
                    
                    # Gerar pontos do círculo
                    theta = np.linspace(0, 2*np.pi, n_circle_points, endpoint=False)
                    circle_x = center[0] + radius * (np.cos(theta) * v1_i[0] + np.sin(theta) * v2_i[0])
                    circle_y = center[1] + radius * (np.cos(theta) * v1_i[1] + np.sin(theta) * v2_i[1])
                    circle_z = center[2] + radius * (np.cos(theta) * v1_i[2] + np.sin(theta) * v2_i[2])
                    
                    # Adicionar centro para formar mesh
                    all_x = np.concatenate([[center[0]], circle_x])
                    all_y = np.concatenate([[center[1]], circle_y])
                    all_z = np.concatenate([[center[2]], circle_z])
                    
                    # Criar triângulos em leque
                    i_idx = [0] * n_circle_points
                    j_idx = list(range(1, n_circle_points + 1))
                    k_idx = list(range(2, n_circle_points + 1)) + [1]
                    
                    hover_text = (
                        f"<b>Família {int(family_id)+1}</b><br>"
                        f"<b>Centro:</b> ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}) m<br>"
                        f"<b>Raio:</b> {radius:.3f} m<br>"
                        f"<b>Área:</b> {np.pi * radius**2:.3f} m²<br>"
                        f"<b>Abertura:</b> {apertures[idx]*1000:.3f} mm<br>"
                        f"<b>Dip:</b> {dips[idx]:.1f}°<br>"
                        f"<b>Dip Dir:</b> {dip_dirs[idx]:.1f}°"
                    )
                    
                    fig.add_trace(go.Mesh3d(
                        x=all_x, y=all_y, z=all_z,
                        i=i_idx, j=j_idx, k=k_idx,
                        opacity=0.6,
                        color=color,
                        showscale=False,
                        hovertext=hover_text,
                        hoverinfo='text',
                        showlegend=(idx_count == 0),
                        name=f'Família {int(family_id)+1} ({n_theoretical_fam})' if idx_count == 0 else None
                    ))
        
        # ================================================================
        # CENTROS DAS FRATURAS (com tamanho AUMENTADO)
        # ================================================================
        if show_centers:
            for family_id in families:
                mask = fractures_df[family_col].values == family_id
                centers_fam = centers[mask]
                n_fam = mask.sum()
                
                if n_fam == 0:
                    continue
                
                color = self.family_colors[int(family_id) % len(self.family_colors)]
                
                radii_fam = radii[mask]
                apertures_fam = apertures[mask]
                dips_fam = dips[mask]
                dip_dirs_fam = dip_dirs[mask]
                
                hover_texts = [
                    f"<b>Centro - Família {int(family_id)+1}</b><br>"
                    f"<b>Posição:</b> ({centers_fam[i,0]:.2f}, {centers_fam[i,1]:.2f}, {centers_fam[i,2]:.2f}) m<br>"
                    f"<b>Raio:</b> {radii_fam[i]:.3f} m<br>"
                    f"<b>Abertura:</b> {apertures_fam[i]*1000:.3f} mm<br>"
                    f"<b>Dip:</b> {dips_fam[i]:.1f}°<br>"
                    f"<b>Dip Dir:</b> {dip_dirs_fam[i]:.1f}°"
                    for i in range(n_fam)
                ]
                
                fig.add_trace(go.Scatter3d(
                    x=centers_fam[:, 0],
                    y=centers_fam[:, 1],
                    z=centers_fam[:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,  # AUMENTADO de 2 para 8
                        color=color,
                        opacity=0.9,
                        symbol='circle',
                        line=dict(width=1, color='white')
                    ),
                    name=f'Centros Fam {int(family_id)+1}',
                    showlegend=True,
                    text=hover_texts,
                    hoverinfo='text'
                ))
        
        # ================================================================
        # CAIXA DO DOMÍNIO
        # ================================================================
        vertices = [
            [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
            [0, 0, depth], [width, 0, depth], [width, height, depth], [0, height, depth]
        ]
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        edge_x, edge_y, edge_z = [], [], []
        for e in edges:
            edge_x.extend([vertices[e[0]][0], vertices[e[1]][0], None])
            edge_y.extend([vertices[e[0]][1], vertices[e[1]][1], None])
            edge_z.extend([vertices[e[0]][2], vertices[e[1]][2], None])
        
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='red', width=3),
            name='Domínio',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # ================================================================
        # LAYOUT
        # ================================================================
        mode_label = 'Linhas' if shape_mode == 'lines' else 'Elipsóides'
        title_text = f'DFN 3D - {n_theoretical} fraturas ({mode_label})'
        if subsampled:
            title_text += f' (visualizando {n_viz})'
        
        fig.update_layout(
            title=title_text,
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            template=None,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=700
        )
        
        return fig
    
    def plot_aperture_length_relation(self, apertures: np.ndarray, 
                                     lengths: np.ndarray, 
                                     fit_params: Dict) -> go.Figure:
        """Plota relação abertura-comprimento com ajuste"""
        mask = (apertures > 0) & (lengths > 0)
        b_valid = apertures[mask]
        l_valid = lengths[mask]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=l_valid,
            y=b_valid,
            mode='markers',
            name='Dados observados',
            marker=dict(size=6, color='blue', symbol='circle', opacity=0.6),
            hovertemplate='Comprimento: %{x:.3f} m<br>Abertura: %{y:.4f} m<extra></extra>'
        ))
        
        l_fit = np.logspace(np.log10(l_valid.min()), np.log10(l_valid.max()), 100)
        b_fit = fit_params['g'] * l_fit ** fit_params['m']
        
        fig.add_trace(go.Scatter(
            x=l_fit,
            y=b_fit,
            mode='lines',
            name=f"b = {fit_params['g']:.2e} × l^{fit_params['m']:.2f}",
            line=dict(color='red', width=2),
            hovertemplate='Comprimento: %{x:.3f} m<br>Abertura ajustada: %{y:.4f} m<extra></extra>'
        ))
        
        fig.update_layout(
            title={'text': 'Relação Abertura-Comprimento (b-l)', 'x': 0.5, 'xanchor': 'center'},
            xaxis_title='Comprimento (m)',
            yaxis_title='Abertura (m)',
            xaxis_type='log',
            yaxis_type='log',
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='closest',
            template=None,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        fig.add_annotation(
            x=0.95, y=0.05,
            xref="paper", yref="paper",
            text=f"R² = {fit_params['r_squared']:.3f}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )
        
        return fig
    
    def plot_rose_diagram(self, orientations: np.ndarray, bins: int = 36) -> go.Figure:
        """Cria diagrama de roseta para orientações"""
        counts, bin_edges = np.histogram(orientations, bins=bins, range=(0, 360))
        theta = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig = go.Figure(go.Barpolar(
            r=counts,
            theta=theta,
            width=360/bins,
            marker_color='blue',
            marker_line_color='black',
            marker_line_width=1,
            opacity=0.8,
            hovertemplate='Direção: %{theta}°<br>Frequência: %{r}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Diagrama de Roseta - Orientações',
            polar=dict(
                radialaxis=dict(visible=True, showticklabels=True, tickfont_size=10),
                angularaxis=dict(visible=True, direction="clockwise", rotation=90, tickmode='linear', tick0=0, dtick=30)
            ),
            showlegend=False,
            template=None,
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig



