import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class FractureVisualizer:
    """Visualizador de fraturas e análises"""
    
    def __init__(self, style: str = 'scientific'):
        self.style = style
        self.colors = px.colors.qualitative.Set2
        
        # Configurar estilo matplotlib
        if style == 'scientific':
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("husl")
    
    def plot_power_law_fit(self, data: np.ndarray, fit_params: Dict) -> go.Figure:
        """
        Plota ajuste de lei de potência
        
        Args:
            data: Dados originais
            fit_params: Parâmetros do ajuste
        
        Returns:
            Figura Plotly
        """
        # Calcular distribuição cumulativa
        sorted_data = np.sort(data)[::-1]
        n = len(sorted_data)
        cumulative = np.arange(1, n + 1)
        
        # Criar figura
        fig = go.Figure()
        
        # Dados observados
        fig.add_trace(go.Scatter(
            x=sorted_data,
            y=cumulative,
            mode='markers',
            name='Dados observados',
            marker=dict(size=6, color='blue', symbol='circle'),
            hovertemplate='Tamanho: %{x:.3f}<br>N(≥x): %{y}<extra></extra>'
        ))
        
        # Ajuste
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
            line=dict(color='red', width=2, dash='solid'),
            hovertemplate='Tamanho: %{x:.3f}<br>N(≥x): %{y:.1f}<extra></extra>'
        ))
        
        # Linha x_min
        fig.add_vline(
            x=fit_params['x_min'],
            line_dash="dash",
            line_color="green",
            annotation_text=f"x_min = {fit_params['x_min']:.3f}"
        )
        
        # Layout log-log
        fig.update_layout(
            title={
                'text': 'Distribuição Power-Law de Tamanhos',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Tamanho (m)',
            yaxis_title='N(≥x) - Número cumulativo',
            xaxis_type='log',
            yaxis_type='log',
            showlegend=True,
            legend=dict(x=0.6, y=0.95),
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Adicionar anotação com estatísticas
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
                    fracture_shape: str = 'lines', 
                    show_centers: bool = False, 
                    show_numbers: bool = False,
                    color_by_family: bool = False) -> go.Figure:
        """
         Visualiza DFN 2D com suporte a coloração por família
        ALTERAÇÃO: Dimensões em mm, cores por família
    
        Args:
            fractures: Lista de Fracture2D
            domain_size: (largura, altura) em mm
            fracture_shape: 'lines' ou 'rectangles'
            show_centers: Mostrar centros das fraturas
            show_numbers: Mostrar numeração
            color_by_family: Colorir por família/set
        
        Returns:
            Figura Plotly
        """
        fig = go.Figure()

        fig = go.Figure()
    
        # Paleta de cores para famílias (tons vibrantes)
        family_colors = [
            '#E74C3C',  # Vermelho
            '#3498DB',  # Azul
            '#2ECC71',  # Verde
            '#F39C12',  # Laranja
            '#9B59B6',  # Roxo
            '#1ABC9C',  # Turquesa
        ]
        
        # Cores padrão mais vibrantes para visualização 2D
        default_colors = ['#34495E', '#16A085', '#E67E22', '#8E44AD']

        # Adicionar fraturas
        for i, frac in enumerate(fractures):
            # Determinar cor
            if color_by_family and hasattr(frac, 'family'):
                color = family_colors[frac.family % len(family_colors)]
                family_label = f"Fam. {frac.family + 1}"
            else:
                color = default_colors[i % len(default_colors)]
                family_label = ""

            # 1. Visualização da Fratura (Linha/Retângulo)
            if fracture_shape == 'lines':
                # Linha da fratura
                fig.add_trace(go.Scatter(
                    x=[frac.x1, frac.x2],
                    y=[frac.y1, frac.y2],
                    mode='lines',
                    line=dict(
                        color=color,
                        width=max(2, frac.aperture * 0.5)  # Escala para mm
                    ),
                    name=family_label if family_label and i < 3 else None,
                    showlegend=(color_by_family and i < len(set(f.family for f in fractures if hasattr(f, 'family')))),
                    legendgroup=family_label,
                    hovertemplate=(
                        f'Fratura {i+1}<br>'
                        f'{family_label}<br>' if family_label else '' +
                        f'Comprimento: {frac.length:.2f} mm<br>'
                        f'Abertura: {frac.aperture:.2f} mm<br>'
                        f'Orientação: {frac.orientation:.1f}°<extra></extra>'
                    )
                ))
                
            elif fracture_shape == 'rectangles':
                dx = frac.x2 - frac.x1
                dy = frac.y2 - frac.y1
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    nx = -dy / length
                    ny = dx / length
                    
                    half_aperture = frac.aperture / 2.0
                    
                    x_rect = [
                        frac.x1 + nx * half_aperture,
                        frac.x2 + nx * half_aperture,
                        frac.x2 - nx * half_aperture,
                        frac.x1 - nx * half_aperture,
                        frac.x1 + nx * half_aperture
                    ]
                    y_rect = [
                        frac.y1 + ny * half_aperture,
                        frac.y2 + ny * half_aperture,
                        frac.y2 - ny * half_aperture,
                        frac.y1 - ny * half_aperture,
                        frac.y1 + ny * half_aperture
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=x_rect,
                        y=y_rect,
                        mode='lines',
                        fill='toself',
                        fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.6])}',
                        line=dict(color=color, width=2),
                        name=family_label if family_label and i < 3 else None,
                        showlegend=(color_by_family and i < len(set(f.family for f in fractures if hasattr(f, 'family')))),
                        legendgroup=family_label,
                        hovertemplate=(
                            f'Fratura {i+1}<br>'
                            f'{family_label}<br>' if family_label else '' +
                            f'Comprimento: {frac.length:.2f} mm<br>'
                            f'Abertura: {frac.aperture:.2f} mm<br>'
                            f'Orientação: {frac.orientation:.1f}°<extra></extra>'
                        )
                    ))
            
            # 2. Centros e numeração
            if show_centers or show_numbers:
                # Calcular o centro da fratura (ponto médio)
                center_x = (frac.x1 + frac.x2) / 2.0
                center_y = (frac.y1 + frac.y2) / 2.0
                
                if show_centers:
                    fig.add_trace(go.Scatter(
                        x=[center_x],
                        y=[center_y],
                        mode='markers',
                        marker=dict(
                            size=6, #8,
                            color= 'white', #'magenta', # Cor viva
                            symbol='circle', #'circle-open'
                            line=dict(color=color, width=2)
                        ),
                        #name=f'Centro Fratura {i+1}',
                        showlegend=False,
                        hovertemplate=f'Centro Fratura {i+1}<extra></extra>'
                    ))
                
                
                # if show_numbers:
                #     fig.add_annotation(
                #         x=center_x,
                #         y=center_y,
                #         text=str(i + 1),
                #         showarrow=False,
                #         font=dict(
                #             size=12,
                #             color="red" # Cor viva para o número
                #         ),
                #         xshift=5, # Deslocamento para não ficar exatamente no centro
                #         yshift=5
                #     )
                if show_numbers:
                    fig.add_annotation(
                        x=center_x,
                        y=center_y,
                        text=str(i + 1),
                        showarrow=False,
                        font=dict(size=10, color='white', family='Arial Black'),
                        bgcolor=color,
                        opacity=0.8,
                        borderpad=2
                    )
        
         # Borda do domínio
        # width, height = domain_size
        # fig.add_shape(
        #     type="rect",
        #     x0=0, y0=0, x1=width, y1=height,
        #     line=dict(color="red", width=2, dash="dash"),
        #     fillcolor="rgba(255,255,255,0)"
        # )
        width, height = domain_size
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=width, y1=height,
            line=dict(color="#2C3E50", width=3, dash="dash"),
            fillcolor="rgba(255,255,255,0)"
        )

        
        # Layout
        # fig.update_layout(
        #     title=f'Rede de Fraturas Discretas 2D ({len(fractures)} fraturas)',
        #     xaxis_title='X (m)',
        #     yaxis_title='Y (m)',
        #     xaxis=dict(
        #         scaleanchor="y",
        #         scaleratio=1,
        #         range=[-width*0.1, width*1.1]
        #     ),
        #     yaxis=dict(range=[-height*0.1, height*1.1]),
        #     template='plotly_white',
        #     showlegend=False,
        #     hovermode='closest'
        # )
            # Layout
        fig.update_layout(
            title=f'Rede de Fraturas Discretas 2D ({len(fractures)} fraturas)',
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                range=[-width*0.05, width*1.05],
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                range=[-height*0.05, height*1.05],
                showgrid=True,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white',
            template='plotly_white',
            hovermode='closest',
            height=600
        )
        
        return fig
    
   

    def plot_dfn_3d(
        self,
        fractures_df: pd.DataFrame,
        domain_size: tuple,
        shape_mode: str = 'ellipsoids',
        show_centers: bool = False,
        show_numbers: bool = False,
        color_by_family: bool = False,
        family_col: str = 'family',
        figure=None
    ) -> go.Figure:
        """
        Visualiza DFN 3D com múltiplos modos de renderização.
        
        Args:
            fractures_df: DataFrame com colunas ['center', 'normal', 'radius', 'aperture', 'dip', 'dip_direction']
                        Opcionalmente 'family' para coloração por conjunto
            domain_size: (largura, altura, profundidade) do domínio
            shape_mode: 'lines' | 'rectangles' | 'ellipsoids'
            show_centers: Mostrar marcadores nos centros
            show_numbers: Mostrar numeração das fraturas
            color_by_family: Colorir por família (requer coluna family_col)
            family_col: Nome da coluna de família no DataFrame
            figure: Figure existente para adicionar traces (opcional)
        
        Returns:
            Figura Plotly 3D atualizada
        """
        if figure is None:
            fig = go.Figure()
        else:
            fig = figure
        
        width, height, depth = domain_size
        
        # Definir paleta de cores para famílias
        family_colors = px.colors.qualitative.Set2
        
        # Verificar se coloração por família está ativa e coluna existe
        use_family_color = color_by_family and family_col in fractures_df.columns
        
        # ========== RENDERIZAR FRATURAS ==========
        for idx, row in fractures_df.iterrows():
            center = np.array(row['center'])
            normal = np.array(row['normal'])
            radius = float(row['radius'])  # Este é o raio real do disco
            
            # Determinar cor
            if use_family_color:
                family_id = int(row[family_col])
                color = family_colors[family_id % len(family_colors)]
            else:
                color = f'rgb({50+idx*5 % 200}, {100+idx*7 % 200}, {150+idx*3 % 200})'
            
            # ========== MODO: LINHAS ==========
            if shape_mode == 'lines':
                # Calcular vetores base no plano da fratura
                if abs(normal[2]) < 0.99:
                    v1 = np.cross(normal, [0, 0, 1])
                else:
                    v1 = np.cross(normal, [1, 0, 0])
                v1 = v1 / np.linalg.norm(v1)
                
                # Extremos da linha atravessando o disco
                # CORREÇÃO: usar diâmetro completo (2 * radius)
                p1 = center - radius * v1
                p2 = center + radius * v1
                
                # CORREÇÃO: typo no índice - era p2[1], deveria ser p2[0]
                fig.add_trace(go.Scatter3d(
                    x=[p1[0], p2[0]],  # ✓ CORRIGIDO
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode='lines',
                    line=dict(color=color, width=6),
                    
                    showlegend=False,
                    hovertemplate=(
                        f'Fratura {idx+1}<br>'
                        f'Raio: {radius:.3f} m<br>'
                        f'Comprimento: {2*radius:.3f} m<br>'
                        f'Abertura: {row["aperture"]*1000:.2f} mm<br>'
                        f'Dip: {row["dip"]:.1f}°<br>'
                        f'Dip Dir: {row["dip_direction"]:.1f}°<extra></extra>'
                    )
                ))
            
            # ========== MODO: RETÂNGULOS ==========
            elif shape_mode == 'rectangles':
                # Gerar base ortonormal no plano da fratura
                if abs(normal[2]) < 0.99:
                    v1 = np.cross(normal, [0, 0, 1])
                else:
                    v1 = np.cross(normal, [1, 0, 0])
                v1 = v1 / np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                
                # CORREÇÃO: usar o raio completo para o retângulo
                # Dimensões do retângulo proporcional ao raio
                a = radius * 10.0  # Comprimento total
                b = radius * 0.7  # Largura (70% do comprimento)
                
                # Vértices do retângulo centrado
                vertices = [
                    center + a*v1 + b*v2,  # Superior direito
                    center - a*v1 + b*v2,  # Superior esquerdo
                    center - a*v1 - b*v2,  # Inferior esquerdo
                    center + a*v1 - b*v2   # Inferior direito
                ]
                
                # Coordenadas para Mesh3d
                x = [v[0] for v in vertices]
                y = [v[1] for v in vertices]
                z = [v[2] for v in vertices]
                
                # Dois triângulos formando o retângulo
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    i=[0, 0],  # Vértices iniciais dos triângulos
                    j=[1, 2],  # Segundos vértices
                    k=[2, 3],  # Terceiros vértices
                    opacity=0.7,
                    color=color,
                    showscale=False,
                    hovertemplate=(
                        f'Fratura {idx+1}<br>'
                        f'Raio: {radius:.3f} m<br>'
                        f'Dimensões: {2*a:.2f}m × {2*b:.2f}m<br>'
                        f'Abertura: {row["aperture"]*1000:.2f} mm<br>'
                        f'Dip: {row["dip"]:.1f}°<br>'
                        f'Dip Dir: {row["dip_direction"]:.1f}°<extra></extra>'
                    )
                ))
                
                # Adicionar contorno do retângulo para melhor visualização
                x_border = x + [x[0]]  # Fechar o loop
                y_border = y + [y[0]]
                z_border = z + [z[0]]
                
                fig.add_trace(go.Scatter3d(
                    x=x_border,
                    y=y_border,
                    z=z_border,
                    mode='lines',
                    line=dict(color='black', width=4),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # ========== MODO: ELIPSÓIDES ==========
            elif shape_mode == 'ellipsoids':
                # CORREÇÃO: aumentar resolução e usar raio correto
                n_points = 30  # Mais pontos para melhor visualização
                theta = np.linspace(0, 2*np.pi, n_points)
                
                # Base ortonormal no plano
                if abs(normal[2]) < 0.99:
                    v1 = np.cross(normal, [0, 0, 1])
                else:
                    v1 = np.cross(normal, [1, 0, 0])
                v1 = v1 / np.linalg.norm(v1)
                v2 = np.cross(normal, v1)
                
                # Gerar pontos do círculo no plano da fratura
                circle_points = []
                for t in theta:
                    point = center + radius * (np.cos(t) * v1 + np.sin(t) * v2)
                    circle_points.append(point)
                
                circle_points = np.array(circle_points)
                
                # Adicionar centro para formar triângulos (leque)
                x_disk = np.concatenate([[center[0]], circle_points[:, 0]])
                y_disk = np.concatenate([[center[1]], circle_points[:, 1]])
                z_disk = np.concatenate([[center[2]], circle_points[:, 2]])
                
                # Criar triângulos em leque a partir do centro
                n = len(theta)
                i_indices = [0] * n
                j_indices = list(range(1, n+1))
                k_indices = list(range(2, n+2))
                k_indices[-1] = 1  # Fechar o círculo
                
                fig.add_trace(go.Mesh3d(
                    x=x_disk,
                    y=y_disk,
                    z=z_disk,
                    i=i_indices,
                    j=j_indices,
                    k=k_indices,
                    opacity=0.7,
                    color=color,
                    showscale=False,
                    hovertemplate=(
                        f'Fratura {idx+1}<br>'
                        f'Raio: {radius:.3f} m<br>'
                        f'Área: {np.pi * radius**2:.3f} m²<br>'
                        f'Abertura: {row["aperture"]*1000:.2f} mm<br>'
                        f'Dip: {row["dip"]:.1f}°<br>'
                        f'Dip Dir: {row["dip_direction"]:.1f}°<extra></extra>'
                    )
                ))
                
                # Adicionar contorno do disco
                fig.add_trace(go.Scatter3d(
                    x=np.append(circle_points[:, 0], circle_points[0, 0]),
                    y=np.append(circle_points[:, 1], circle_points[0, 1]),
                    z=np.append(circle_points[:, 2], circle_points[0, 2]),
                    mode='lines',
                    line=dict(color='black', width=4),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # ========== CENTROS DAS FRATURAS ==========
            if show_centers:
                fig.add_trace(go.Scatter3d(
                    x=[center[0]],
                    y=[center[1]],
                    z=[center[2]],
                    mode='markers',
                    marker=dict(size=4, color='magenta', symbol='circle'),
                    name=f'Centro {idx+1}',
                    showlegend=False,
                    hovertemplate=f'Centro Fratura {idx+1}<extra></extra>'
                ))
            
            # ========== NUMERAÇÃO DAS FRATURAS ==========
            if show_numbers:
                fig.add_trace(go.Scatter3d(
                    x=[center[0]],
                    y=[center[1]],
                    z=[center[2]],
                    mode='text',
                    text=[str(idx + 1)],
                    textposition='top center',
                    textfont=dict(size=10, color='red', family='Arial Black'),
                    showlegend=False,
                    hovertemplate=f'Fratura {idx+1}<extra></extra>'
                ))
        
        # ========== CAIXA DO DOMÍNIO ==========
        vertices = [
            [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
            [0, 0, depth], [width, 0, depth], [width, height, depth], [0, height, depth]
        ]
        
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Base
            [4, 5], [5, 6], [6, 7], [7, 4],  # Topo
            [0, 4], [1, 5], [2, 6], [3, 7]   # Verticais
        ]
        
        for edge in edges:
            v1, v2 = edge
            fig.add_trace(go.Scatter3d(
                x=[vertices[v1][0], vertices[v2][0]],
                y=[vertices[v1][1], vertices[v2][1]],
                z=[vertices[v1][2], vertices[v2][2]],
                mode='lines',
                line=dict(color='black', width=4),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # ========== LAYOUT ==========
        mode_labels = {
            'lines': 'Linhas',
            'rectangles': 'Retângulos',
            'ellipsoids': 'Elipsóides'
        }
        
        fig.update_layout(
            title=f'Rede de Fraturas Discretas 3D ({len(fractures_df)} fraturas) - Modo: {mode_labels.get(shape_mode, shape_mode)}',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            showlegend=False,
            template='plotly_white',
            height=700
        )
        
        return fig


    def plot_rose_diagram(self, orientations: np.ndarray, bins: int = 36) -> go.Figure:
        """
        Cria diagrama de roseta para orientações
        
        Args:
            orientations: Array de orientações em graus
            bins: Número de bins
        
        Returns:
            Figura Plotly polar
        """
        # Calcular histograma
        counts, bin_edges = np.histogram(orientations, bins=bins, range=(0, 360))
        
        # Centro dos bins
        theta = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Criar figura polar
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
                radialaxis=dict(
                    visible=True,
                    showticklabels=True,
                    tickfont_size=10
                ),
                angularaxis=dict(
                    visible=True,
                    direction="clockwise",
                    rotation=90,
                    tickmode='linear',
                    tick0=0,
                    dtick=30
                )
            ),
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_stereonet(self, dips: np.ndarray, dip_directions: np.ndarray) -> go.Figure:
        """
        Cria estereograma para orientações 3D
        
        Args:
            dips: Ângulos de mergulho em graus
            dip_directions: Direções de mergulho em graus
        
        Returns:
            Figura Plotly
        """
        # Converter para projeção estereográfica
        dips_rad = np.radians(dips)
        dirs_rad = np.radians(dip_directions)
        
        # Projeção de Schmidt (equal area)
        r = np.sqrt(2) * np.sin(dips_rad / 2)
        x = r * np.sin(dirs_rad)
        y = r * np.cos(dirs_rad)
        
        # Criar figura
        fig = go.Figure()
        
        # Adicionar pontos
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            marker=dict(
                size=8,
                color=dips,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Dip (°)')
            ),
            hovertemplate='Dip: %{marker.color:.1f}°<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
        ))
        
        # Adicionar círculo unitário
        theta = np.linspace(0, 2*np.pi, 100)
        fig.add_trace(go.Scatter(
            x=np.cos(theta),
            y=np.sin(theta),
            mode='lines',
            line=dict(color='black', width=4),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Layout
        fig.update_layout(
            title='Estereograma (Projeção de Schmidt)',
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                range=[-1.1, 1.1],
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                range=[-1.1, 1.1],
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    

    def plot_aperture_length_relation(self, apertures: np.ndarray, 
                                 lengths: np.ndarray, 
                                 fit_params: Dict) -> go.Figure:
        """
        Plota relação abertura-comprimento com ajuste
        
        Args:
            apertures: Array de aberturas
            lengths: Array de comprimentos
            fit_params: Parâmetros do ajuste (m, g)
        
        Returns:
            Figura Plotly
        """
        # Filtrar valores válidos
        mask = (apertures > 0) & (lengths > 0)
        b_valid = apertures[mask]
        l_valid = lengths[mask]
        
        # Criar figura
        fig = go.Figure()
        
        # Dados observados
        fig.add_trace(go.Scatter(
            x=l_valid,
            y=b_valid,
            mode='markers',
            name='Dados observados',
            marker=dict(
                size=6,
                color='blue',
                symbol='circle',
                opacity=0.6
            ),
            hovertemplate='Comprimento: %{x:.3f} m<br>Abertura: %{y:.4f} m<extra></extra>'
        ))
        
        # Ajuste power-law: b = g * l^m
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
        
        # Se houver ajuste robusto
        if 'm_robust' in fit_params:
            b_fit_robust = fit_params['g_robust'] * l_fit ** fit_params['m_robust']
            fig.add_trace(go.Scatter(
                x=l_fit,
                y=b_fit_robust,
                mode='lines',
                name=f"Robusto: b = {fit_params['g_robust']:.2e} × l^{fit_params['m_robust']:.2f}",
                line=dict(color='green', width=2, dash='dash'),
                hovertemplate='Comprimento: %{x:.3f} m<br>Abertura robusta: %{y:.4f} m<extra></extra>'
            ))
        
        # Layout log-log
        fig.update_layout(
            title={
                'text': 'Relação Abertura-Comprimento (b-l)',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Comprimento (m)',
            yaxis_title='Abertura (m)',
            xaxis_type='log',
            yaxis_type='log',
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='closest',
            template='plotly_white'
        )
        
        # Adicionar anotação com R²
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

    def plot_intensity_comparison(self, thresholds: np.ndarray,
                                p10_framfrat: np.ndarray,
                                p10_scanline: np.ndarray) -> go.Figure:
        """
        Plota comparação de intensidades P10 vs threshold
        
        Args:
            thresholds: Array de limiares
            p10_framfrat: Intensidades FRAMFRAT
            p10_scanline: Intensidades scanline
        
        Returns:
            Figura Plotly
        """
        fig = go.Figure()
        
        # P10 FRAMFRAT
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=p10_framfrat,
            mode='lines+markers',
            name='FRAMFRAT',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Threshold: %{x:.3f} m<br>P10: %{y:.2f} fraturas/m<extra></extra>'
        ))
        
        # P10 Scanline
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=p10_scanline,
            mode='lines+markers',
            name='Scanline',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            hovertemplate='Threshold: %{x:.3f} m<br>P10: %{y:.2f} fraturas/m<extra></extra>'
        ))
        
        # Layout
        fig.update_layout(
            title='Comparação de Intensidades Size-Cognizant',
            xaxis_title='Limiar de tamanho (m)',
            yaxis_title='P10 (fraturas/m)',
            xaxis_type='log',
            yaxis_type='log',
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

    def plot_spacing_comparison(self, thresholds: np.ndarray,
                            spacing_framfrat: np.ndarray,
                            spacing_scanline: np.ndarray) -> go.Figure:
        """
        Plota comparação de espaçamentos médios
        
        Args:
            thresholds: Array de limiares
            spacing_framfrat: Espaçamentos FRAMFRAT
            spacing_scanline: Espaçamentos scanline
        
        Returns:
            Figura Plotly
        """
        fig = go.Figure()
        
        # Espaçamento FRAMFRAT
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=spacing_framfrat,
            mode='lines+markers',
            name='FRAMFRAT',
            line=dict(color='blue', width=2),
            marker=dict(size=8),
            hovertemplate='Threshold: %{x:.3f} m<br>Espaçamento: %{y:.2f} m<extra></extra>'
        ))
        
        # Espaçamento Scanline
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=spacing_scanline,
            mode='lines+markers',
            name='Scanline',
            line=dict(color='red', width=2),
            marker=dict(size=8),
            hovertemplate='Threshold: %{x:.3f} m<br>Espaçamento: %{y:.2f} m<extra></extra>'
        ))
        
        # Layout
        fig.update_layout(
            title='Comparação de Espaçamentos Médios',
            xaxis_title='Limiar de tamanho (m)',
            yaxis_title='Espaçamento médio (m)',
            xaxis_type='log',
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

    def plot_histogram_comparison(self, data1: np.ndarray, data2: np.ndarray,
                                label1: str = "FRAMFRAT", 
                                label2: str = "Scanline",
                                variable: str = "Comprimento") -> go.Figure:
        """
        Plota histogramas comparativos
        
        Args:
            data1: Primeiro conjunto de dados
            data2: Segundo conjunto de dados
            label1: Rótulo do primeiro conjunto
            label2: Rótulo do segundo conjunto
            variable: Nome da variável
        
        Returns:
            Figura Plotly
        """
        fig = go.Figure()
        
        # Histograma 1
        fig.add_trace(go.Histogram(
            x=data1,
            name=label1,
            opacity=0.6,
            marker_color='blue',
            xbins=dict(
                start=np.log10(min(data1.min(), data2.min())),
                end=np.log10(max(data1.max(), data2.max())),
                size=0.1
            ),
            histnorm='probability density',
            hovertemplate=f'{variable}: %{{x:.3f}}<br>Densidade: %{{y:.3f}}<extra></extra>'
        ))
        
        # Histograma 2
        fig.add_trace(go.Histogram(
            x=data2,
            name=label2,
            opacity=0.6,
            marker_color='red',
            xbins=dict(
                start=np.log10(min(data1.min(), data2.min())),
                end=np.log10(max(data1.max(), data2.max())),
                size=0.1
            ),
            histnorm='probability density',
            hovertemplate=f'{variable}: %{{x:.3f}}<br>Densidade: %{{y:.3f}}<extra></extra>'
        ))
        
        # Layout
        fig.update_layout(
            title=f'Distribuição de {variable}',
            xaxis_title=f'{variable} (m)',
            yaxis_title='Densidade de probabilidade',
            xaxis_type='log',
            barmode='overlay',
            showlegend=True,
            legend=dict(x=0.7, y=0.98),
            template='plotly_white'
        )
        
        return fig

    def plot_cumulative_comparison(self, data1: np.ndarray, data2: np.ndarray,
                                label1: str = "FRAMFRAT",
                                label2: str = "Scanline") -> go.Figure:
        """
        Plota distribuições cumulativas comparativas
        
        Args:
            data1: Primeiro conjunto de dados
            data2: Segundo conjunto de dados
            label1: Rótulo do primeiro conjunto
            label2: Rótulo do segundo conjunto
        
        Returns:
            Figura Plotly
        """
        fig = go.Figure()
        
        # Calcular distribuições cumulativas
        sorted1 = np.sort(data1)[::-1]
        sorted2 = np.sort(data2)[::-1]
        
        cum1 = np.arange(1, len(sorted1) + 1)
        cum2 = np.arange(1, len(sorted2) + 1)
        
        # Normalizar pelo total
        cum1_norm = cum1 / len(sorted1)
        cum2_norm = cum2 / len(sorted2)
        
        # Plotar
        fig.add_trace(go.Scatter(
            x=sorted1,
            y=cum1_norm,
            mode='lines',
            name=label1,
            line=dict(color='blue', width=2),
            hovertemplate='Tamanho: %{x:.3f}<br>P(X≥x): %{y:.3f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=sorted2,
            y=cum2_norm,
            mode='lines',
            name=label2,
            line=dict(color='red', width=2),
            hovertemplate='Tamanho: %{x:.3f}<br>P(X≥x): %{y:.3f}<extra></extra>'
        ))
        
        # Layout
        fig.update_layout(
            title='Distribuição Cumulativa Complementar',
            xaxis_title='Tamanho (m)',
            yaxis_title='P(X ≥ x)',
            xaxis_type='log',
            yaxis_type='log',
            showlegend=True,
            legend=dict(x=0.7, y=0.98),
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig

    def plot_connectivity_matrix(self, connectivity: np.ndarray) -> go.Figure:
        """
        Plota matriz de conectividade
        
        Args:
            connectivity: Matriz de adjacência
        
        Returns:
            Figura Plotly
        """
        fig = go.Figure(data=go.Heatmap(
            z=connectivity,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Conectado'),
            hovertemplate='Fratura %{y} - Fratura %{x}: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Matriz de Conectividade',
            xaxis_title='Índice da fratura',
            yaxis_title='Índice da fratura',
            template='plotly_white'
        )
        
        return fig

    def plot_3d_surface_density(self, x: np.ndarray, y: np.ndarray, 
                            density: np.ndarray) -> go.Figure:
        """
        Plota superfície 3D de densidade de fraturas
        
        Args:
            x: Coordenadas X
            y: Coordenadas Y
            density: Valores de densidade
        
        Returns:
            Figura Plotly 3D
        """
        fig = go.Figure(data=[go.Surface(
            x=x,
            y=y,
            z=density,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Densidade (fraturas/m²)'),
            hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Densidade: %{z:.3f}<extra></extra>'
        )])
        
        fig.update_layout(
            title='Densidade Espacial de Fraturas',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Densidade (fraturas/m²)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            template='plotly_white'
        )
        
        return fig










# import plotly.graph_objects as go
# import plotly.express as px
# import numpy as np
# import pandas as pd
# from typing import List, Dict, Optional
# import matplotlib.pyplot as plt
# import seaborn as sns

# class FractureVisualizer:
#     """Visualizador de fraturas e anÃ¡lises"""
    
#     def __init__(self, style: str = 'scientific'):
#         self.style = style
#         self.colors = px.colors.qualitative.Set2
        
#         # Configurar estilo matplotlib
#         if style == 'scientific':
#             plt.style.use('seaborn-v0_8-darkgrid')
#             sns.set_palette("husl")
    
#     def plot_power_law_fit(self, data: np.ndarray, fit_params: Dict) -> go.Figure:
#         """
#         Plota ajuste de lei de potÃªncia
        
#         Args:
#             data: Dados originais
#             fit_params: ParÃ¢metros do ajuste
        
#         Returns:
#             Figura Plotly
#         """
#         # Calcular distribuiÃ§Ã£o cumulativa
#         sorted_data = np.sort(data)[::-1]
#         n = len(sorted_data)
#         cumulative = np.arange(1, n + 1)
        
#         # Criar figura
#         fig = go.Figure()
        
#         # Dados observados
#         fig.add_trace(go.Scatter(
#             x=sorted_data,
#             y=cumulative,
#             mode='markers',
#             name='Dados observados',
#             marker=dict(size=6, color='blue', symbol='circle'),
#             hovertemplate='Tamanho: %{x:.3f}<br>N(â‰¥x): %{y}<extra></extra>'
#         ))
        
#         # Ajuste
#         x_fit = np.logspace(
#             np.log10(fit_params['x_min']),
#             np.log10(sorted_data[0]),
#             100
#         )
#         y_fit = fit_params['coefficient'] * x_fit**(-fit_params['exponent'])
        
#         fig.add_trace(go.Scatter(
#             x=x_fit,
#             y=y_fit,
#             mode='lines',
#             name=f"Ajuste: N = {fit_params['coefficient']:.1f} Ã— x^(-{fit_params['exponent']:.2f})",
#             line=dict(color='red', width=2, dash='solid'),
#             hovertemplate='Tamanho: %{x:.3f}<br>N(â‰¥x): %{y:.1f}<extra></extra>'
#         ))
        
#         # Linha x_min
#         fig.add_vline(
#             x=fit_params['x_min'],
#             line_dash="dash",
#             line_color="green",
#             annotation_text=f"x_min = {fit_params['x_min']:.3f}"
#         )
        
#         # Layout log-log
#         fig.update_layout(
#             title={
#                 'text': 'DistribuiÃ§Ã£o Power-Law de Tamanhos',
#                 'x': 0.5,
#                 'xanchor': 'center'
#             },
#             xaxis_title='Tamanho (m)',
#             yaxis_title='N(â‰¥x) - NÃºmero cumulativo',
#             xaxis_type='log',
#             yaxis_type='log',
#             showlegend=True,
#             legend=dict(x=0.6, y=0.95),
#             hovermode='x unified',
#             template='plotly_white'
#         )
        
#         # Adicionar anotaÃ§Ã£o com estatÃ­sticas
#         if 'r_squared' in fit_params:
#             annotation_text = f"RÂ² = {fit_params['r_squared']:.3f}"
#         else:
#             annotation_text = f"KS = {fit_params.get('ks_statistic', 0):.3f}"
        
#         fig.add_annotation(
#             x=0.95, y=0.05,
#             xref="paper", yref="paper",
#             text=annotation_text,
#             showarrow=False,
#             font=dict(size=12),
#             bgcolor="white",
#             bordercolor="black",
#             borderwidth=1
#         )
        
#         return fig
    
#     def plot_dfn_2d(self, fractures: List, domain_size: tuple, 
#                     fracture_shape: str = 'lines', 
#                     show_centers: bool = False, 
#                     show_numbers: bool = False) -> go.Figure:
#         """
#         Visualiza DFN 2D
        
#         Args:
#             fractures: Lista de Fracture2D
#             domain_size: (largura, altura)
        
#         Returns:
#             Figura Plotly
#         """
#         fig = go.Figure()
        
#         # Adicionar fraturas
#         for i, frac in enumerate(fractures):
#             # 1. VisualizaÃ§Ã£o da Fratura (Linha/RetÃ¢ngulo)
#             if fracture_shape == 'lines':
#                 # Linha da fratura
#                 fig.add_trace(go.Scatter(
#                     x=[frac.x1, frac.x2],
#                     y=[frac.y1, frac.y2],
#                     mode='lines',
#                     line=dict(
#                         color='black',
#                         width=max(1, frac.aperture * 1000)  # Espessura proporcional
#                     ),
#                     showlegend=False,
#                     hovertemplate=(
#                         f'Fratura {i+1}<br>'
#                         f'Comprimento: {frac.length:.3f} m<br>'
#                         f'Abertura: {frac.aperture*1000:.2f} mm<br>'
#                         f'OrientaÃ§Ã£o: {frac.orientation:.1f}Â°<extra></extra>'
#                     )
#                 ))
#             elif fracture_shape == 'rectangles':
#                 # ImplementaÃ§Ã£o simplificada de retÃ¢ngulo (apenas para visualizaÃ§Ã£o)
#                 # O DFN 2D gera linhas, mas o usuÃ¡rio pediu "RetÃ¢ngulos"
#                 # Usaremos a abertura para dar espessura Ã  linha
                
#                 # Calcular o vetor normal Ã  fratura
#                 dx = frac.x2 - frac.x1
#                 dy = frac.y2 - frac.y1
#                 length = np.sqrt(dx**2 + dy**2)
                
#                 if length > 0:
#                     # Vetor unitÃ¡rio perpendicular (normal)
#                     nx = -dy / length
#                     ny = dx / length
                    
#                     half_aperture = frac.aperture / 2.0
                    
#                     # Coordenadas dos 4 cantos do retÃ¢ngulo
#                     x_rect = [
#                         frac.x1 + nx * half_aperture,
#                         frac.x2 + nx * half_aperture,
#                         frac.x2 - nx * half_aperture,
#                         frac.x1 - nx * half_aperture,
#                         frac.x1 + nx * half_aperture # Fechar o polÃ­gono
#                     ]
#                     y_rect = [
#                         frac.y1 + ny * half_aperture,
#                         frac.y2 + ny * half_aperture,
#                         frac.y2 - ny * half_aperture,
#                         frac.y1 - ny * half_aperture,
#                         frac.y1 + ny * half_aperture # Fechar o polÃ­gono
#                     ]
                    
#                     fig.add_trace(go.Scatter(
#                         x=x_rect,
#                         y=y_rect,
#                         mode='lines',
#                         fill='toself',
#                         fillcolor='rgba(0, 0, 0, 0.5)',
#                         line=dict(color='black', width=1),
#                         showlegend=False,
#                         hovertemplate=(
#                             f'Fratura {i+1}<br>'
#                             f'Comprimento: {frac.length:.3f} m<br>'
#                             f'Abertura: {frac.aperture*1000:.2f} mm<br>'
#                             f'OrientaÃ§Ã£o: {frac.orientation:.1f}Â°<extra></extra>'
#                         )
#                     ))
            
#             # 2. VisualizaÃ§Ã£o dos Centros
#             if show_centers or show_numbers:
#                 # Calcular o centro da fratura (ponto mÃ©dio)
#                 center_x = (frac.x1 + frac.x2) / 2.0
#                 center_y = (frac.y1 + frac.y2) / 2.0
                
#                 if show_centers:
#                     fig.add_trace(go.Scatter(
#                         x=[center_x],
#                         y=[center_y],
#                         mode='markers',
#                         marker=dict(
#                             size=8,
#                             color='magenta', # Cor viva
#                             symbol='circle-open'
#                         ),
#                         name=f'Centro Fratura {i+1}',
#                         showlegend=False,
#                         hovertemplate=f'Centro Fratura {i+1}<extra></extra>'
#                     ))
                
#                 # 3. VisualizaÃ§Ã£o da NumeraÃ§Ã£o
#                 if show_numbers:
#                     fig.add_annotation(
#                         x=center_x,
#                         y=center_y,
#                         text=str(i + 1),
#                         showarrow=False,
#                         font=dict(
#                             size=12,
#                             color="red" # Cor viva para o nÃºmero
#                         ),
#                         xshift=5, # Deslocamento para nÃ£o ficar exatamente no centro
#                         yshift=5
#                     )
        
#         # O formato 'Discos' nÃ£o se aplica a DFN 2D gerado por linhas, 
#         # mas a funÃ§Ã£o plot_dfn_3d jÃ¡ usa discos.
#         # A opÃ§Ã£o 'lines' Ã© o padrÃ£o para DFN 2D.
#         # Se o usuÃ¡rio selecionar 'Discos' em 2D, trataremos como 'lines' ou ignoraremos.
#         # Como a funÃ§Ã£o Ã© plot_dfn_2d, vamos focar em linhas e retÃ¢ngulos.
#         # A opÃ§Ã£o 'Discos' serÃ¡ tratada na funÃ§Ã£o plot_dfn_3d se necessÃ¡rio, mas o foco Ã© 2D.
#         # Por enquanto, se for 'discos', faremos o padrÃ£o 'lines' para 2D.
#         if fracture_shape == 'discs':
#             # Adicionar um aviso ou simplesmente usar 'lines'
#             pass
        
#         # Fim da lÃ³gica de visualizaÃ§Ã£o
        
#         # Adicionar bordas do domÃ­nio
#         width, height = domain_size
#         fig.add_shape(
#             type="rect",
#             x0=0, y0=0, x1=width, y1=height,
#             line=dict(color="red", width=2, dash="dash"),
#             fillcolor="rgba(255,255,255,0)"
#         )
        
#         # Layout
#         fig.update_layout(
# 	            title=f'Rede de Fraturas Discretas 2D ({len(fractures)} fraturas)',
#             xaxis_title='X (m)',
#             yaxis_title='Y (m)',
#             xaxis=dict(
#                 scaleanchor="y",
#                 scaleratio=1,
#                 range=[-width*0.1, width*1.1]
#             ),
#             yaxis=dict(range=[-height*0.1, height*1.1]),
#             template='plotly_white',
#             showlegend=False,
#             hovermode='closest'
#         )
        
#         return fig
    
   

#     def plot_dfn_3d(
#         self,
#         fractures_df: pd.DataFrame,
#         domain_size: tuple,
#         shape_mode: str = 'ellipsoids',
#         show_centers: bool = False,
#         show_numbers: bool = False,
#         color_by_family: bool = False,
#         family_col: str = 'family',
#         figure=None
#     ) -> go.Figure:
#         """
#         Visualiza DFN 3D com mÃºltiplos modos de renderizaÃ§Ã£o.
        
#         Args:
#             fractures_df: DataFrame com colunas ['center', 'normal', 'radius', 'aperture', 'dip', 'dip_direction']
#                         Opcionalmente 'family' para coloraÃ§Ã£o por conjunto
#             domain_size: (largura, altura, profundidade) do domÃ­nio
#             shape_mode: 'lines' | 'rectangles' | 'ellipsoids'
#             show_centers: Mostrar marcadores nos centros
#             show_numbers: Mostrar numeraÃ§Ã£o das fraturas
#             color_by_family: Colorir por famÃ­lia (requer coluna family_col)
#             family_col: Nome da coluna de famÃ­lia no DataFrame
#             figure: Figure existente para adicionar traces (opcional)
        
#         Returns:
#             Figura Plotly 3D atualizada
#         """
#         if figure is None:
#             fig = go.Figure()
#         else:
#             fig = figure
        
#         width, height, depth = domain_size
        
#         # Definir paleta de cores para famÃ­lias
#         family_colors = px.colors.qualitative.Set2
        
#         # Verificar se coloraÃ§Ã£o por famÃ­lia estÃ¡ ativa e coluna existe
#         use_family_color = color_by_family and family_col in fractures_df.columns
        
#         # ========== RENDERIZAR FRATURAS ==========
#         for idx, row in fractures_df.iterrows():
#             center = np.array(row['center'])
#             normal = np.array(row['normal'])
#             radius = float(row['radius'])  # Este Ã© o raio real do disco
            
#             # Determinar cor
#             if use_family_color:
#                 family_id = int(row[family_col])
#                 color = family_colors[family_id % len(family_colors)]
#             else:
#                 color = f'rgb({50+idx*5 % 200}, {100+idx*7 % 200}, {150+idx*3 % 200})'
            
#             # ========== MODO: LINHAS ==========
#             if shape_mode == 'lines':
#                 # Calcular vetores base no plano da fratura
#                 if abs(normal[2]) < 0.99:
#                     v1 = np.cross(normal, [0, 0, 1])
#                 else:
#                     v1 = np.cross(normal, [1, 0, 0])
#                 v1 = v1 / np.linalg.norm(v1)
                
#                 # Extremos da linha atravessando o disco
#                 # CORREÃ‡ÃƒO: usar diÃ¢metro completo (2 * radius)
#                 p1 = center - radius * v1
#                 p2 = center + radius * v1
                
#                 # CORREÃ‡ÃƒO: typo no Ã­ndice - era p2[1], deveria ser p2[0]
#                 fig.add_trace(go.Scatter3d(
#                     x=[p1[0], p2[0]],  # âœ… CORRIGIDO
#                     y=[p1[1], p2[1]],
#                     z=[p1[2], p2[2]],
#                     mode='lines',
#                     line=dict(color=color, width=6),
#                     showlegend=False,
#                     hovertemplate=(
#                         f'Fratura {idx+1}<br>'
#                         f'Raio: {radius:.3f} m<br>'
#                         f'Comprimento: {2*radius:.3f} m<br>'
#                         f'Abertura: {row["aperture"]*1000:.2f} mm<br>'
#                         f'Dip: {row["dip"]:.1f}Â°<br>'
#                         f'Dip Dir: {row["dip_direction"]:.1f}Â°<extra></extra>'
#                     )
#                 ))
            
#             # ========== MODO: RETÃ‚NGULOS ==========
#             elif shape_mode == 'rectangles':
#                 # Gerar base ortonormal no plano da fratura
#                 if abs(normal[2]) < 0.99:
#                     v1 = np.cross(normal, [0, 0, 1])
#                 else:
#                     v1 = np.cross(normal, [1, 0, 0])
#                 v1 = v1 / np.linalg.norm(v1)
#                 v2 = np.cross(normal, v1)
                
#                 # CORREÃ‡ÃƒO: usar o raio completo para o retÃ¢ngulo
#                 # DimensÃµes do retÃ¢ngulo proporcional ao raio
#                 a = radius * 10.0  # Comprimento total
#                 b = radius * 0.7  # Largura (70% do comprimento)
                
#                 # VÃ©rtices do retÃ¢ngulo centrado
#                 vertices = [
#                     center + a*v1 + b*v2,  # Superior direito
#                     center - a*v1 + b*v2,  # Superior esquerdo
#                     center - a*v1 - b*v2,  # Inferior esquerdo
#                     center + a*v1 - b*v2   # Inferior direito
#                 ]
                
#                 # Coordenadas para Mesh3d
#                 x = [v[0] for v in vertices]
#                 y = [v[1] for v in vertices]
#                 z = [v[2] for v in vertices]
                
#                 # Dois triÃ¢ngulos formando o retÃ¢ngulo
#                 fig.add_trace(go.Mesh3d(
#                     x=x, y=y, z=z,
#                     i=[0, 0],  # VÃ©rtices iniciais dos triÃ¢ngulos
#                     j=[1, 2],  # Segundos vÃ©rtices
#                     k=[2, 3],  # Terceiros vÃ©rtices
#                     opacity=0.7,
#                     color=color,
#                     showscale=False,
#                     hovertemplate=(
#                         f'Fratura {idx+1}<br>'
#                         f'Raio: {radius:.3f} m<br>'
#                         f'DimensÃµes: {2*a:.2f}m Ã— {2*b:.2f}m<br>'
#                         f'Abertura: {row["aperture"]*1000:.2f} mm<br>'
#                         f'Dip: {row["dip"]:.1f}Â°<br>'
#                         f'Dip Dir: {row["dip_direction"]:.1f}Â°<extra></extra>'
#                     )
#                 ))
                
#                 # Adicionar contorno do retÃ¢ngulo para melhor visualizaÃ§Ã£o
#                 x_border = x + [x[0]]  # Fechar o loop
#                 y_border = y + [y[0]]
#                 z_border = z + [z[0]]
                
#                 fig.add_trace(go.Scatter3d(
#                     x=x_border,
#                     y=y_border,
#                     z=z_border,
#                     mode='lines',
#                     line=dict(color='black', width=4),
#                     showlegend=False,
#                     hoverinfo='skip'
#                 ))
            
#             # ========== MODO: ELIPSÃ“IDES ==========
#             elif shape_mode == 'ellipsoids':
#                 # CORREÃ‡ÃƒO: aumentar resoluÃ§Ã£o e usar raio correto
#                 n_points = 30  # Mais pontos para melhor visualizaÃ§Ã£o
#                 theta = np.linspace(0, 2*np.pi, n_points)
                
#                 # Base ortonormal no plano
#                 if abs(normal[2]) < 0.99:
#                     v1 = np.cross(normal, [0, 0, 1])
#                 else:
#                     v1 = np.cross(normal, [1, 0, 0])
#                 v1 = v1 / np.linalg.norm(v1)
#                 v2 = np.cross(normal, v1)
                
#                 # Gerar pontos do cÃ­rculo no plano da fratura
#                 circle_points = []
#                 for t in theta:
#                     point = center + radius * (np.cos(t) * v1 + np.sin(t) * v2)
#                     circle_points.append(point)
                
#                 circle_points = np.array(circle_points)
                
#                 # Adicionar centro para formar triÃ¢ngulos (leque)
#                 x_disk = np.concatenate([[center[0]], circle_points[:, 0]])
#                 y_disk = np.concatenate([[center[1]], circle_points[:, 1]])
#                 z_disk = np.concatenate([[center[2]], circle_points[:, 2]])
                
#                 # Criar triÃ¢ngulos em leque a partir do centro
#                 n = len(theta)
#                 i_indices = [0] * n
#                 j_indices = list(range(1, n+1))
#                 k_indices = list(range(2, n+2))
#                 k_indices[-1] = 1  # Fechar o cÃ­rculo
                
#                 fig.add_trace(go.Mesh3d(
#                     x=x_disk,
#                     y=y_disk,
#                     z=z_disk,
#                     i=i_indices,
#                     j=j_indices,
#                     k=k_indices,
#                     opacity=0.7,
#                     color=color,
#                     showscale=False,
#                     hovertemplate=(
#                         f'Fratura {idx+1}<br>'
#                         f'Raio: {radius:.3f} m<br>'
#                         f'Ãrea: {np.pi * radius**2:.3f} mÂ²<br>'
#                         f'Abertura: {row["aperture"]*1000:.2f} mm<br>'
#                         f'Dip: {row["dip"]:.1f}Â°<br>'
#                         f'Dip Dir: {row["dip_direction"]:.1f}Â°<extra></extra>'
#                     )
#                 ))
                
#                 # Adicionar contorno do disco
#                 fig.add_trace(go.Scatter3d(
#                     x=np.append(circle_points[:, 0], circle_points[0, 0]),
#                     y=np.append(circle_points[:, 1], circle_points[0, 1]),
#                     z=np.append(circle_points[:, 2], circle_points[0, 2]),
#                     mode='lines',
#                     line=dict(color='black', width=4),
#                     showlegend=False,
#                     hoverinfo='skip'
#                 ))
            
#             # ========== CENTROS DAS FRATURAS ==========
#             if show_centers:
#                 fig.add_trace(go.Scatter3d(
#                     x=[center[0]],
#                     y=[center[1]],
#                     z=[center[2]],
#                     mode='markers',
#                     marker=dict(size=4, color='magenta', symbol='circle'),
#                     name=f'Centro {idx+1}',
#                     showlegend=False,
#                     hovertemplate=f'Centro Fratura {idx+1}<extra></extra>'
#                 ))
            
#             # ========== NUMERAÃ‡ÃƒO DAS FRATURAS ==========
#             if show_numbers:
#                 fig.add_trace(go.Scatter3d(
#                     x=[center[0]],
#                     y=[center[1]],
#                     z=[center[2]],
#                     mode='text',
#                     text=[str(idx + 1)],
#                     textposition='top center',
#                     textfont=dict(size=10, color='red', family='Arial Black'),
#                     showlegend=False,
#                     hovertemplate=f'Fratura {idx+1}<extra></extra>'
#                 ))
        
#         # ========== CAIXA DO DOMÃNIO ==========
#         vertices = [
#             [0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0],
#             [0, 0, depth], [width, 0, depth], [width, height, depth], [0, height, depth]
#         ]
        
#         edges = [
#             [0, 1], [1, 2], [2, 3], [3, 0],  # Base
#             [4, 5], [5, 6], [6, 7], [7, 4],  # Topo
#             [0, 4], [1, 5], [2, 6], [3, 7]   # Verticais
#         ]
        
#         for edge in edges:
#             v1, v2 = edge
#             fig.add_trace(go.Scatter3d(
#                 x=[vertices[v1][0], vertices[v2][0]],
#                 y=[vertices[v1][1], vertices[v2][1]],
#                 z=[vertices[v1][2], vertices[v2][2]],
#                 mode='lines',
#                 line=dict(color='black', width=4),
#                 showlegend=False,
#                 hoverinfo='skip'
#             ))
        
#         # ========== LAYOUT ==========
#         mode_labels = {
#             'lines': 'Linhas',
#             'rectangles': 'RetÃ¢ngulos',
#             'ellipsoids': 'ElipsÃ³ides'
#         }
        
#         fig.update_layout(
#             title=f'Rede de Fraturas Discretas 3D ({len(fractures_df)} fraturas) - Modo: {mode_labels.get(shape_mode, shape_mode)}',
#             scene=dict(
#                 xaxis_title='X (m)',
#                 yaxis_title='Y (m)',
#                 zaxis_title='Z (m)',
#                 aspectmode='data',
#                 camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
#             ),
#             showlegend=False,
#             template='plotly_white',
#             height=700
#         )
        
#         return fig


#     def plot_rose_diagram(self, orientations: np.ndarray, bins: int = 36) -> go.Figure:
#         """
#         Cria diagrama de roseta para orientaÃ§Ãµes
        
#         Args:
#             orientations: Array de orientaÃ§Ãµes em graus
#             bins: NÃºmero de bins
        
#         Returns:
#             Figura Plotly polar
#         """
#         # Calcular histograma
#         counts, bin_edges = np.histogram(orientations, bins=bins, range=(0, 360))
        
#         # Centro dos bins
#         theta = (bin_edges[:-1] + bin_edges[1:]) / 2
        
#         # Criar figura polar
#         fig = go.Figure(go.Barpolar(
#             r=counts,
#             theta=theta,
#             width=360/bins,
#             marker_color='blue',
#             marker_line_color='black',
#             marker_line_width=1,
#             opacity=0.8,
#             hovertemplate='DireÃ§Ã£o: %{theta}Â°<br>FrequÃªncia: %{r}<extra></extra>'
#         ))
        
#         fig.update_layout(
#             title='Diagrama de Roseta - OrientaÃ§Ãµes',
#             polar=dict(
#                 radialaxis=dict(
#                     visible=True,
#                     showticklabels=True,
#                     tickfont_size=10
#                 ),
#                 angularaxis=dict(
#                     visible=True,
#                     direction="clockwise",
#                     rotation=90,
#                     tickmode='linear',
#                     tick0=0,
#                     dtick=30
#                 )
#             ),
#             showlegend=False,
#             template='plotly_white'
#         )
        
#         return fig
    
#     def plot_stereonet(self, dips: np.ndarray, dip_directions: np.ndarray) -> go.Figure:
#         """
#         Cria estereograma para orientaÃ§Ãµes 3D
        
#         Args:
#             dips: Ã‚ngulos de mergulho em graus
#             dip_directions: DireÃ§Ãµes de mergulho em graus
        
#         Returns:
#             Figura Plotly
#         """
#         # Converter para projeÃ§Ã£o estereogrÃ¡fica
#         dips_rad = np.radians(dips)
#         dirs_rad = np.radians(dip_directions)
        
#         # ProjeÃ§Ã£o de Schmidt (equal area)
#         r = np.sqrt(2) * np.sin(dips_rad / 2)
#         x = r * np.sin(dirs_rad)
#         y = r * np.cos(dirs_rad)
        
#         # Criar figura
#         fig = go.Figure()
        
#         # Adicionar pontos
#         fig.add_trace(go.Scatter(
#             x=x, y=y,
#             mode='markers',
#             marker=dict(
#                 size=8,
#                 color=dips,
#                 colorscale='Viridis',
#                 showscale=True,
#                 colorbar=dict(title='Dip (Â°)')
#             ),
#             hovertemplate='Dip: %{marker.color:.1f}Â°<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
#         ))
        
#         # Adicionar cÃ­rculo unitÃ¡rio
#         theta = np.linspace(0, 2*np.pi, 100)
#         fig.add_trace(go.Scatter(
#             x=np.cos(theta),
#             y=np.sin(theta),
#             mode='lines',
#             line=dict(color='black', width=4),
#             showlegend=False,
#             hoverinfo='skip'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title='Estereograma (ProjeÃ§Ã£o de Schmidt)',
#             xaxis=dict(
#                 scaleanchor="y",
#                 scaleratio=1,
#                 range=[-1.1, 1.1],
#                 showgrid=False,
#                 zeroline=False,
#                 visible=False
#             ),
#             yaxis=dict(
#                 range=[-1.1, 1.1],
#                 showgrid=False,
#                 zeroline=False,
#                 visible=False
#             ),
#             template='plotly_white',
#             showlegend=False
#         )
        
#         return fig
    

#     def plot_aperture_length_relation(self, apertures: np.ndarray, 
#                                  lengths: np.ndarray, 
#                                  fit_params: Dict) -> go.Figure:
#         """
#         Plota relaÃ§Ã£o abertura-comprimento com ajuste
        
#         Args:
#             apertures: Array de aberturas
#             lengths: Array de comprimentos
#             fit_params: ParÃ¢metros do ajuste (m, g)
        
#         Returns:
#             Figura Plotly
#         """
#         # Filtrar valores vÃ¡lidos
#         mask = (apertures > 0) & (lengths > 0)
#         b_valid = apertures[mask]
#         l_valid = lengths[mask]
        
#         # Criar figura
#         fig = go.Figure()
        
#         # Dados observados
#         fig.add_trace(go.Scatter(
#             x=l_valid,
#             y=b_valid,
#             mode='markers',
#             name='Dados observados',
#             marker=dict(
#                 size=6,
#                 color='blue',
#                 symbol='circle',
#                 opacity=0.6
#             ),
#             hovertemplate='Comprimento: %{x:.3f} m<br>Abertura: %{y:.4f} m<extra></extra>'
#         ))
        
#         # Ajuste power-law: b = g * l^m
#         l_fit = np.logspace(np.log10(l_valid.min()), np.log10(l_valid.max()), 100)
#         b_fit = fit_params['g'] * l_fit ** fit_params['m']
        
#         fig.add_trace(go.Scatter(
#             x=l_fit,
#             y=b_fit,
#             mode='lines',
#             name=f"b = {fit_params['g']:.2e} Ã— l^{fit_params['m']:.2f}",
#             line=dict(color='red', width=2),
#             hovertemplate='Comprimento: %{x:.3f} m<br>Abertura ajustada: %{y:.4f} m<extra></extra>'
#         ))
        
#         # Se houver ajuste robusto
#         if 'm_robust' in fit_params:
#             b_fit_robust = fit_params['g_robust'] * l_fit ** fit_params['m_robust']
#             fig.add_trace(go.Scatter(
#                 x=l_fit,
#                 y=b_fit_robust,
#                 mode='lines',
#                 name=f"Robusto: b = {fit_params['g_robust']:.2e} Ã— l^{fit_params['m_robust']:.2f}",
#                 line=dict(color='green', width=2, dash='dash'),
#                 hovertemplate='Comprimento: %{x:.3f} m<br>Abertura robusta: %{y:.4f} m<extra></extra>'
#             ))
        
#         # Layout log-log
#         fig.update_layout(
#             title={
#                 'text': 'RelaÃ§Ã£o Abertura-Comprimento (b-l)',
#                 'x': 0.5,
#                 'xanchor': 'center'
#             },
#             xaxis_title='Comprimento (m)',
#             yaxis_title='Abertura (m)',
#             xaxis_type='log',
#             yaxis_type='log',
#             showlegend=True,
#             legend=dict(x=0.02, y=0.98),
#             hovermode='closest',
#             template='plotly_white'
#         )
        
#         # Adicionar anotaÃ§Ã£o com RÂ²
#         fig.add_annotation(
#             x=0.95, y=0.05,
#             xref="paper", yref="paper",
#             text=f"RÂ² = {fit_params['r_squared']:.3f}",
#             showarrow=False,
#             font=dict(size=12),
#             bgcolor="white",
#             bordercolor="black",
#             borderwidth=1
#         )
        
#         return fig

#     def plot_intensity_comparison(self, thresholds: np.ndarray,
#                                 p10_framfrat: np.ndarray,
#                                 p10_scanline: np.ndarray) -> go.Figure:
#         """
#         Plota comparaÃ§Ã£o de intensidades P10 vs threshold
        
#         Args:
#             thresholds: Array de limiares
#             p10_framfrat: Intensidades FRAMFRAT
#             p10_scanline: Intensidades scanline
        
#         Returns:
#             Figura Plotly
#         """
#         fig = go.Figure()
        
#         # P10 FRAMFRAT
#         fig.add_trace(go.Scatter(
#             x=thresholds,
#             y=p10_framfrat,
#             mode='lines+markers',
#             name='FRAMFRAT',
#             line=dict(color='blue', width=2),
#             marker=dict(size=8),
#             hovertemplate='Threshold: %{x:.3f} m<br>P10: %{y:.2f} fraturas/m<extra></extra>'
#         ))
        
#         # P10 Scanline
#         fig.add_trace(go.Scatter(
#             x=thresholds,
#             y=p10_scanline,
#             mode='lines+markers',
#             name='Scanline',
#             line=dict(color='red', width=2),
#             marker=dict(size=8),
#             hovertemplate='Threshold: %{x:.3f} m<br>P10: %{y:.2f} fraturas/m<extra></extra>'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title='ComparaÃ§Ã£o de Intensidades Size-Cognizant',
#             xaxis_title='Limiar de tamanho (m)',
#             yaxis_title='P10 (fraturas/m)',
#             xaxis_type='log',
#             yaxis_type='log',
#             showlegend=True,
#             legend=dict(x=0.02, y=0.98),
#             hovermode='x unified',
#             template='plotly_white'
#         )
        
#         return fig

#     def plot_spacing_comparison(self, thresholds: np.ndarray,
#                             spacing_framfrat: np.ndarray,
#                             spacing_scanline: np.ndarray) -> go.Figure:
#         """
#         Plota comparaÃ§Ã£o de espaÃ§amentos mÃ©dios
        
#         Args:
#             thresholds: Array de limiares
#             spacing_framfrat: EspaÃ§amentos FRAMFRAT
#             spacing_scanline: EspaÃ§amentos scanline
        
#         Returns:
#             Figura Plotly
#         """
#         fig = go.Figure()
        
#         # EspaÃ§amento FRAMFRAT
#         fig.add_trace(go.Scatter(
#             x=thresholds,
#             y=spacing_framfrat,
#             mode='lines+markers',
#             name='FRAMFRAT',
#             line=dict(color='blue', width=2),
#             marker=dict(size=8),
#             hovertemplate='Threshold: %{x:.3f} m<br>EspaÃ§amento: %{y:.2f} m<extra></extra>'
#         ))
        
#         # EspaÃ§amento Scanline
#         fig.add_trace(go.Scatter(
#             x=thresholds,
#             y=spacing_scanline,
#             mode='lines+markers',
#             name='Scanline',
#             line=dict(color='red', width=2),
#             marker=dict(size=8),
#             hovertemplate='Threshold: %{x:.3f} m<br>EspaÃ§amento: %{y:.2f} m<extra></extra>'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title='ComparaÃ§Ã£o de EspaÃ§amentos MÃ©dios',
#             xaxis_title='Limiar de tamanho (m)',
#             yaxis_title='EspaÃ§amento mÃ©dio (m)',
#             xaxis_type='log',
#             showlegend=True,
#             legend=dict(x=0.02, y=0.98),
#             hovermode='x unified',
#             template='plotly_white'
#         )
        
#         return fig

#     def plot_histogram_comparison(self, data1: np.ndarray, data2: np.ndarray,
#                                 label1: str = "FRAMFRAT", 
#                                 label2: str = "Scanline",
#                                 variable: str = "Comprimento") -> go.Figure:
#         """
#         Plota histogramas comparativos
        
#         Args:
#             data1: Primeiro conjunto de dados
#             data2: Segundo conjunto de dados
#             label1: RÃ³tulo do primeiro conjunto
#             label2: RÃ³tulo do segundo conjunto
#             variable: Nome da variÃ¡vel
        
#         Returns:
#             Figura Plotly
#         """
#         fig = go.Figure()
        
#         # Histograma 1
#         fig.add_trace(go.Histogram(
#             x=data1,
#             name=label1,
#             opacity=0.6,
#             marker_color='blue',
#             xbins=dict(
#                 start=np.log10(min(data1.min(), data2.min())),
#                 end=np.log10(max(data1.max(), data2.max())),
#                 size=0.1
#             ),
#             histnorm='probability density',
#             hovertemplate=f'{variable}: %{{x:.3f}}<br>Densidade: %{{y:.3f}}<extra></extra>'
#         ))
        
#         # Histograma 2
#         fig.add_trace(go.Histogram(
#             x=data2,
#             name=label2,
#             opacity=0.6,
#             marker_color='red',
#             xbins=dict(
#                 start=np.log10(min(data1.min(), data2.min())),
#                 end=np.log10(max(data1.max(), data2.max())),
#                 size=0.1
#             ),
#             histnorm='probability density',
#             hovertemplate=f'{variable}: %{{x:.3f}}<br>Densidade: %{{y:.3f}}<extra></extra>'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title=f'DistribuiÃ§Ã£o de {variable}',
#             xaxis_title=f'{variable} (m)',
#             yaxis_title='Densidade de probabilidade',
#             xaxis_type='log',
#             barmode='overlay',
#             showlegend=True,
#             legend=dict(x=0.7, y=0.98),
#             template='plotly_white'
#         )
        
#         return fig

#     def plot_cumulative_comparison(self, data1: np.ndarray, data2: np.ndarray,
#                                 label1: str = "FRAMFRAT",
#                                 label2: str = "Scanline") -> go.Figure:
#         """
#         Plota distribuiÃ§Ãµes cumulativas comparativas
        
#         Args:
#             data1: Primeiro conjunto de dados
#             data2: Segundo conjunto de dados
#             label1: RÃ³tulo do primeiro conjunto
#             label2: RÃ³tulo do segundo conjunto
        
#         Returns:
#             Figura Plotly
#         """
#         fig = go.Figure()
        
#         # Calcular distribuiÃ§Ãµes cumulativas
#         sorted1 = np.sort(data1)[::-1]
#         sorted2 = np.sort(data2)[::-1]
        
#         cum1 = np.arange(1, len(sorted1) + 1)
#         cum2 = np.arange(1, len(sorted2) + 1)
        
#         # Normalizar pelo total
#         cum1_norm = cum1 / len(sorted1)
#         cum2_norm = cum2 / len(sorted2)
        
#         # Plotar
#         fig.add_trace(go.Scatter(
#             x=sorted1,
#             y=cum1_norm,
#             mode='lines',
#             name=label1,
#             line=dict(color='blue', width=2),
#             hovertemplate='Tamanho: %{x:.3f}<br>P(Xâ‰¥x): %{y:.3f}<extra></extra>'
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=sorted2,
#             y=cum2_norm,
#             mode='lines',
#             name=label2,
#             line=dict(color='red', width=2),
#             hovertemplate='Tamanho: %{x:.3f}<br>P(Xâ‰¥x): %{y:.3f}<extra></extra>'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title='DistribuiÃ§Ã£o Cumulativa Complementar',
#             xaxis_title='Tamanho (m)',
#             yaxis_title='P(X â‰¥ x)',
#             xaxis_type='log',
#             yaxis_type='log',
#             showlegend=True,
#             legend=dict(x=0.7, y=0.98),
#             hovermode='x unified',
#             template='plotly_white'
#         )
        
#         return fig

#     def plot_connectivity_matrix(self, connectivity: np.ndarray) -> go.Figure:
#         """
#         Plota matriz de conectividade
        
#         Args:
#             connectivity: Matriz de adjacÃªncia
        
#         Returns:
#             Figura Plotly
#         """
#         fig = go.Figure(data=go.Heatmap(
#             z=connectivity,
#             colorscale='Viridis',
#             showscale=True,
#             colorbar=dict(title='Conectado'),
#             hovertemplate='Fratura %{y} - Fratura %{x}: %{z}<extra></extra>'
#         ))
        
#         fig.update_layout(
#             title='Matriz de Conectividade',
#             xaxis_title='Ãndice da fratura',
#             yaxis_title='Ãndice da fratura',
#             template='plotly_white'
#         )
        
#         return fig

#     def plot_3d_surface_density(self, x: np.ndarray, y: np.ndarray, 
#                             density: np.ndarray) -> go.Figure:
#         """
#         Plota superfÃ­cie 3D de densidade de fraturas
        
#         Args:
#             x: Coordenadas X
#             y: Coordenadas Y
#             density: Valores de densidade
        
#         Returns:
#             Figura Plotly 3D
#         """
#         fig = go.Figure(data=[go.Surface(
#             x=x,
#             y=y,
#             z=density,
#             colorscale='Viridis',
#             showscale=True,
#             colorbar=dict(title='Densidade (fraturas/mÂ²)'),
#             hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Densidade: %{z:.3f}<extra></extra>'
#         )])
        
#         fig.update_layout(
#             title='Densidade Espacial de Fraturas',
#             scene=dict(
#                 xaxis_title='X (m)',
#                 yaxis_title='Y (m)',
#                 zaxis_title='Densidade (fraturas/mÂ²)',
#                 camera=dict(
#                     eye=dict(x=1.5, y=1.5, z=1.5)
#                 )
#             ),
#             template='plotly_white'
#         )
        
#         return fig        