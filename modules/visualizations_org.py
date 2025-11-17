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
                    show_numbers: bool = False) -> go.Figure:
        """
        Visualiza DFN 2D
        
        Args:
            fractures: Lista de Fracture2D
            domain_size: (largura, altura)
        
        Returns:
            Figura Plotly
        """
        fig = go.Figure()
        
        # Adicionar fraturas
        for i, frac in enumerate(fractures):
            # 1. Visualização da Fratura (Linha/Retângulo)
            if fracture_shape == 'lines':
                # Linha da fratura
                fig.add_trace(go.Scatter(
                    x=[frac.x1, frac.x2],
                    y=[frac.y1, frac.y2],
                    mode='lines',
                    line=dict(
                        color='black',
                        width=max(1, frac.aperture * 1000)  # Espessura proporcional
                    ),
                    showlegend=False,
                    hovertemplate=(
                        f'Fratura {i+1}<br>'
                        f'Comprimento: {frac.length:.3f} m<br>'
                        f'Abertura: {frac.aperture*1000:.2f} mm<br>'
                        f'Orientação: {frac.orientation:.1f}°<extra></extra>'
                    )
                ))
            elif fracture_shape == 'rectangles':
                # Implementação simplificada de retângulo (apenas para visualização)
                # O DFN 2D gera linhas, mas o usuário pediu "Retângulos"
                # Usaremos a abertura para dar espessura à linha
                
                # Calcular o vetor normal à fratura
                dx = frac.x2 - frac.x1
                dy = frac.y2 - frac.y1
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Vetor unitário perpendicular (normal)
                    nx = -dy / length
                    ny = dx / length
                    
                    half_aperture = frac.aperture / 2.0
                    
                    # Coordenadas dos 4 cantos do retângulo
                    x_rect = [
                        frac.x1 + nx * half_aperture,
                        frac.x2 + nx * half_aperture,
                        frac.x2 - nx * half_aperture,
                        frac.x1 - nx * half_aperture,
                        frac.x1 + nx * half_aperture # Fechar o polígono
                    ]
                    y_rect = [
                        frac.y1 + ny * half_aperture,
                        frac.y2 + ny * half_aperture,
                        frac.y2 - ny * half_aperture,
                        frac.y1 - ny * half_aperture,
                        frac.y1 + ny * half_aperture # Fechar o polígono
                    ]
                    
                    fig.add_trace(go.Scatter(
                        x=x_rect,
                        y=y_rect,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(0, 0, 0, 0.5)',
                        line=dict(color='black', width=1),
                        showlegend=False,
                        hovertemplate=(
                            f'Fratura {i+1}<br>'
                            f'Comprimento: {frac.length:.3f} m<br>'
                            f'Abertura: {frac.aperture*1000:.2f} mm<br>'
                            f'Orientação: {frac.orientation:.1f}°<extra></extra>'
                        )
                    ))
            
            # 2. Visualização dos Centros
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
                            size=8,
                            color='magenta', # Cor viva
                            symbol='circle-open'
                        ),
                        name=f'Centro Fratura {i+1}',
                        showlegend=False,
                        hovertemplate=f'Centro Fratura {i+1}<extra></extra>'
                    ))
                
                # 3. Visualização da Numeração
                if show_numbers:
                    fig.add_annotation(
                        x=center_x,
                        y=center_y,
                        text=str(i + 1),
                        showarrow=False,
                        font=dict(
                            size=12,
                            color="red" # Cor viva para o número
                        ),
                        xshift=5, # Deslocamento para não ficar exatamente no centro
                        yshift=5
                    )
        
        # O formato 'Discos' não se aplica a DFN 2D gerado por linhas, 
        # mas a função plot_dfn_3d já usa discos.
        # A opção 'lines' é o padrão para DFN 2D.
        # Se o usuário selecionar 'Discos' em 2D, trataremos como 'lines' ou ignoraremos.
        # Como a função é plot_dfn_2d, vamos focar em linhas e retângulos.
        # A opção 'Discos' será tratada na função plot_dfn_3d se necessário, mas o foco é 2D.
        # Por enquanto, se for 'discos', faremos o padrão 'lines' para 2D.
        if fracture_shape == 'discs':
            # Adicionar um aviso ou simplesmente usar 'lines'
            pass
        
        # Fim da lógica de visualização
        
        # Adicionar bordas do domínio
        width, height = domain_size
        fig.add_shape(
            type="rect",
            x0=0, y0=0, x1=width, y1=height,
            line=dict(color="red", width=2, dash="dash"),
            fillcolor="rgba(255,255,255,0)"
        )
        
        # Layout
        fig.update_layout(
	            title=f'Rede de Fraturas Discretas 2D ({len(fractures)} fraturas)',
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            xaxis=dict(
                scaleanchor="y",
                scaleratio=1,
                range=[-width*0.1, width*1.1]
            ),
            yaxis=dict(range=[-height*0.1, height*1.1]),
            template='plotly_white',
            showlegend=False,
            hovermode='closest'
        )
        
        return fig
    
    def plot_dfn_3d(self, fractures: List, domain_size: tuple, 
                    show_centers: bool = False, 
                    show_numbers: bool = False,
                    show_disks: bool = True,
                    show_volume: bool = True,
                    show_plane: bool = False,
                    disk_opacity: float = 0.6,
                    mode: str = 'full',
                    plane_z: float = None) -> go.Figure:
        """
        Visualiza DFN 3D
        
        Args:
            fractures: Lista de Fracture3D
            domain_size: (largura, altura, profundidade)
        
        Returns:
            Figura Plotly 3D
        """
        fig = go.Figure()
        
        # Adicionar cada fratura como um disco
        for i, frac in enumerate(fractures):
            # Criar pontos do disco
            theta = np.linspace(0, 2*np.pi, 20)
            
            # Vetores no plano do disco
            if abs(frac.normal[2]) < 0.99:
                v1 = np.cross(frac.normal, [0, 0, 1])
            else:
                v1 = np.cross(frac.normal, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(frac.normal, v1)
            
            # Pontos do círculo
            x = frac.center[0] + frac.radius * (np.cos(theta)[:, np.newaxis] * v1[0] + 
                                                np.sin(theta)[:, np.newaxis] * v2[0]).flatten()
            y = frac.center[1] + frac.radius * (np.cos(theta)[:, np.newaxis] * v1[1] + 
                                                np.sin(theta)[:, np.newaxis] * v2[1]).flatten()
            z = frac.center[2] + frac.radius * (np.cos(theta)[:, np.newaxis] * v1[2] + 
                                                np.sin(theta)[:, np.newaxis] * v2[2]).flatten()
            
            # 1. Visualização dos Discos (Fraturas)
            if show_disks:
                # Adicionar superfície do disco
                fig.add_trace(go.Mesh3d(
                    x=x, y=y, z=z,
                    opacity=disk_opacity,
                    color=f'rgb({50+i*5 % 200}, {100+i*7 % 200}, {150+i*3 % 200})',
                    hovertemplate=(
                        f'Fratura {i+1}<br>'
                        f'Raio: {frac.radius:.3f} m<br>'
                        f'Abertura: {frac.aperture*1000:.2f} mm<br>'
                        f'Dip: {frac.dip:.1f}°<br>'
                        f'Dip Dir: {frac.dip_direction:.1f}°<extra></extra>'
                    ),
                    showscale=False
                ))
            
            # 2. Visualização dos Centros e Numeração
            if show_centers or show_numbers:
                center_x, center_y, center_z = frac.center
                
                if show_centers:
                    fig.add_trace(go.Scatter3d(
                        x=[center_x],
                        y=[center_y],
                        z=[center_z],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='magenta', # Cor viva
                            symbol='circle'
                        ),
                        name=f'Centro Fratura {i+1}',
                        showlegend=False,
                        hovertemplate=f'Centro Fratura {i+1}<extra></extra>'
                    ))
                
                if show_numbers:
                    # Adicionar o número da fratura
                    fig.add_trace(go.Scatter3d(
                        x=[center_x + 0.01], # Pequeno offset para não sobrepor o centro
                        y=[center_y + 0.01],
                        z=[center_z + 0.01],
                        mode='text',
                        text=[str(i + 1)],
                        textfont=dict(
                            size=12,
                            color="red" # Cor viva para o número
                        ),
                        showlegend=False,
                        hovertemplate=f'Fratura {i+1}<extra></extra>'
                    ))
        
        # Adicionar caixa do domínio
        width, height, depth = domain_size
        
        if show_volume:
            # Vértices da caixa
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
                    line=dict(color='black', width=1),
                    showlegend=False
                ))

        # Adicionar plano de corte
        if show_plane and plane_z is not None:
            fig.add_trace(go.Mesh3d(
                x=[0, width, width, 0],
                y=[0, 0, height, height],
                z=[plane_z, plane_z, plane_z, plane_z],
                color='rgba(0, 255, 0, 0.3)',
                opacity=0.5,
                showscale=False,
                name='Plano de Corte'
            ))
            
        # Layout
        fig.update_layout(
            title=f'Rede de Fraturas Discretas 3D ({len(fractures)} fraturas)',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=False,
            template='plotly_white'
        )
        

        # Layout
        fig.update_layout(
            title=f'Rede de Fraturas Discretas 3D ({len(fractures)} fraturas)',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            template='plotly_white',
            showlegend=False
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
            line=dict(color='black', width=2),
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
#     """Visualizador de fraturas e análises"""
    
#     def __init__(self, style: str = 'scientific'):
#         self.style = style
#         self.colors = px.colors.qualitative.Set2
        
#         # Configurar estilo matplotlib
#         if style == 'scientific':
#             plt.style.use('seaborn-v0_8-darkgrid')
#             sns.set_palette("husl")
    
#     def plot_power_law_fit(self, data: np.ndarray, fit_params: Dict) -> go.Figure:
#         """
#         Plota ajuste de lei de potência
        
#         Args:
#             data: Dados originais
#             fit_params: Parâmetros do ajuste
        
#         Returns:
#             Figura Plotly
#         """
#         # Calcular distribuição cumulativa
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
#             hovertemplate='Tamanho: %{x:.3f}<br>N(≥x): %{y}<extra></extra>'
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
#             name=f"Ajuste: N = {fit_params['coefficient']:.1f} × x^(-{fit_params['exponent']:.2f})",
#             line=dict(color='red', width=2, dash='solid'),
#             hovertemplate='Tamanho: %{x:.3f}<br>N(≥x): %{y:.1f}<extra></extra>'
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
#                 'text': 'Distribuição Power-Law de Tamanhos',
#                 'x': 0.5,
#                 'xanchor': 'center'
#             },
#             xaxis_title='Tamanho (m)',
#             yaxis_title='N(≥x) - Número cumulativo',
#             xaxis_type='log',
#             yaxis_type='log',
#             showlegend=True,
#             legend=dict(x=0.6, y=0.95),
#             hovermode='x unified',
#             template='plotly_white'
#         )
        
#         # Adicionar anotação com estatísticas
#         if 'r_squared' in fit_params:
#             annotation_text = f"R² = {fit_params['r_squared']:.3f}"
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
#             # 1. Visualização da Fratura (Linha/Retângulo)
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
#                         f'Orientação: {frac.orientation:.1f}°<extra></extra>'
#                     )
#                 ))
#             elif fracture_shape == 'rectangles':
#                 # Implementação simplificada de retângulo (apenas para visualização)
#                 # O DFN 2D gera linhas, mas o usuário pediu "Retângulos"
#                 # Usaremos a abertura para dar espessura à linha
                
#                 # Calcular o vetor normal à fratura
#                 dx = frac.x2 - frac.x1
#                 dy = frac.y2 - frac.y1
#                 length = np.sqrt(dx**2 + dy**2)
                
#                 if length > 0:
#                     # Vetor unitário perpendicular (normal)
#                     nx = -dy / length
#                     ny = dx / length
                    
#                     half_aperture = frac.aperture / 2.0
                    
#                     # Coordenadas dos 4 cantos do retângulo
#                     x_rect = [
#                         frac.x1 + nx * half_aperture,
#                         frac.x2 + nx * half_aperture,
#                         frac.x2 - nx * half_aperture,
#                         frac.x1 - nx * half_aperture,
#                         frac.x1 + nx * half_aperture # Fechar o polígono
#                     ]
#                     y_rect = [
#                         frac.y1 + ny * half_aperture,
#                         frac.y2 + ny * half_aperture,
#                         frac.y2 - ny * half_aperture,
#                         frac.y1 - ny * half_aperture,
#                         frac.y1 + ny * half_aperture # Fechar o polígono
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
#                             f'Orientação: {frac.orientation:.1f}°<extra></extra>'
#                         )
#                     ))
            
#             # 2. Visualização dos Centros
#             if show_centers or show_numbers:
#                 # Calcular o centro da fratura (ponto médio)
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
                
#                 # 3. Visualização da Numeração
#                 if show_numbers:
#                     fig.add_annotation(
#                         x=center_x,
#                         y=center_y,
#                         text=str(i + 1),
#                         showarrow=False,
#                         font=dict(
#                             size=12,
#                             color="red" # Cor viva para o número
#                         ),
#                         xshift=5, # Deslocamento para não ficar exatamente no centro
#                         yshift=5
#                     )
        
#         # O formato 'Discos' não se aplica a DFN 2D gerado por linhas, 
#         # mas a função plot_dfn_3d já usa discos.
#         # A opção 'lines' é o padrão para DFN 2D.
#         # Se o usuário selecionar 'Discos' em 2D, trataremos como 'lines' ou ignoraremos.
#         # Como a função é plot_dfn_2d, vamos focar em linhas e retângulos.
#         # A opção 'Discos' será tratada na função plot_dfn_3d se necessário, mas o foco é 2D.
#         # Por enquanto, se for 'discos', faremos o padrão 'lines' para 2D.
#         if fracture_shape == 'discs':
#             # Adicionar um aviso ou simplesmente usar 'lines'
#             pass
        
#         # Fim da lógica de visualização
        
#         # Adicionar bordas do domínio
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
    
#     def plot_dfn_3d(self, fractures: List, domain_size: tuple, 
#                     show_centers: bool = False, 
#                     show_numbers: bool = False) -> go.Figure:
#         """
#         Visualiza DFN 3D
        
#         Args:
#             fractures: Lista de Fracture3D
#             domain_size: (largura, altura, profundidade)
        
#         Returns:
#             Figura Plotly 3D
#         """
#         fig = go.Figure()
        
#         # Adicionar cada fratura como um disco
#         for i, frac in enumerate(fractures):
#             # Criar pontos do disco
#             theta = np.linspace(0, 2*np.pi, 20)
            
#             # Vetores no plano do disco
#             if abs(frac.normal[2]) < 0.99:
#                 v1 = np.cross(frac.normal, [0, 0, 1])
#             else:
#                 v1 = np.cross(frac.normal, [1, 0, 0])
#             v1 = v1 / np.linalg.norm(v1)
#             v2 = np.cross(frac.normal, v1)
            
#             # Pontos do círculo
#             x = frac.center[0] + frac.radius * (np.cos(theta)[:, np.newaxis] * v1[0] + 
#                                                 np.sin(theta)[:, np.newaxis] * v2[0]).flatten()
#             y = frac.center[1] + frac.radius * (np.cos(theta)[:, np.newaxis] * v1[1] + 
#                                                 np.sin(theta)[:, np.newaxis] * v2[1]).flatten()
#             z = frac.center[2] + frac.radius * (np.cos(theta)[:, np.newaxis] * v1[2] + 
#                                                 np.sin(theta)[:, np.newaxis] * v2[2]).flatten()
            
#             # Adicionar superfície do disco
#             fig.add_trace(go.Mesh3d(
#                 x=x, y=y, z=z,
#                 opacity=0.7,
#                 color=f'rgb({50+i*5 % 200}, {100+i*7 % 200}, {150+i*3 % 200})',
#                 hovertemplate=(
#                     f'Fratura {i+1}<br>'
#                     f'Raio: {frac.radius:.3f} m<br>'
#                     f'Abertura: {frac.aperture*1000:.2f} mm<br>'
#                     f'Dip: {frac.dip:.1f}°<br>'
#                     f'Dip Dir: {frac.dip_direction:.1f}°<extra></extra>'
#                 ),
#                 showscale=False
#             ))
            
#             # 2. Visualização dos Centros e Numeração
#             if show_centers or show_numbers:
#                 center_x, center_y, center_z = frac.center
                
#                 if show_centers:
#                     fig.add_trace(go.Scatter3d(
#                         x=[center_x],
#                         y=[center_y],
#                         z=[center_z],
#                         mode='markers',
#                         marker=dict(
#                             size=5,
#                             color='magenta', # Cor viva
#                             symbol='circle'
#                         ),
#                         name=f'Centro Fratura {i+1}',
#                         showlegend=False,
#                         hovertemplate=f'Centro Fratura {i+1}<extra></extra>'
#                     ))
                
#                 if show_numbers:
#                     fig.add_annotation(
#                         x=center_x,
#                         y=center_y,
#                         z=center_z,
#                         text=str(i + 1),
#                         showarrow=False,
#                         font=dict(
#                             size=12,
#                             color="red" # Cor viva para o número
#                         ),
#                         xshift=5, # Deslocamento para não ficar exatamente no centro
#                         yshift=5
#                     )
        
#         # Adicionar caixa do domínio
#         width, height, depth = domain_size
        
#         # Vértices da caixa
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
#                 line=dict(color='red', width=3),
#                 showlegend=False,
#                 hoverinfo='skip'
#             ))
        
#         # Layout
#         fig.update_layout(
#             title='Rede de Fraturas Discretas 3D',
#             scene=dict(
#                 xaxis_title='X (m)',
#                 yaxis_title='Y (m)',
#                 zaxis_title='Z (m)',
#                 aspectmode='data',
#                 camera=dict(
#                     eye=dict(x=1.5, y=1.5, z=1.5)
#                 )
#             ),
#             showlegend=False,
#             template='plotly_white'
#         )
        
#         return fig
    
#     def plot_rose_diagram(self, orientations: np.ndarray, bins: int = 36) -> go.Figure:
#         """
#         Cria diagrama de roseta para orientações
        
#         Args:
#             orientations: Array de orientações em graus
#             bins: Número de bins
        
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
#             hovertemplate='Direção: %{theta}°<br>Frequência: %{r}<extra></extra>'
#         ))
        
#         fig.update_layout(
#             title='Diagrama de Roseta - Orientações',
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
#         Cria estereograma para orientações 3D
        
#         Args:
#             dips: Ângulos de mergulho em graus
#             dip_directions: Direções de mergulho em graus
        
#         Returns:
#             Figura Plotly
#         """
#         # Converter para projeção estereográfica
#         dips_rad = np.radians(dips)
#         dirs_rad = np.radians(dip_directions)
        
#         # Projeção de Schmidt (equal area)
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
#                 colorbar=dict(title='Dip (°)')
#             ),
#             hovertemplate='Dip: %{marker.color:.1f}°<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
#         ))
        
#         # Adicionar círculo unitário
#         theta = np.linspace(0, 2*np.pi, 100)
#         fig.add_trace(go.Scatter(
#             x=np.cos(theta),
#             y=np.sin(theta),
#             mode='lines',
#             line=dict(color='black', width=2),
#             showlegend=False,
#             hoverinfo='skip'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title='Estereograma (Projeção de Schmidt)',
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
#         Plota relação abertura-comprimento com ajuste
        
#         Args:
#             apertures: Array de aberturas
#             lengths: Array de comprimentos
#             fit_params: Parâmetros do ajuste (m, g)
        
#         Returns:
#             Figura Plotly
#         """
#         # Filtrar valores válidos
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
#             name=f"b = {fit_params['g']:.2e} × l^{fit_params['m']:.2f}",
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
#                 name=f"Robusto: b = {fit_params['g_robust']:.2e} × l^{fit_params['m_robust']:.2f}",
#                 line=dict(color='green', width=2, dash='dash'),
#                 hovertemplate='Comprimento: %{x:.3f} m<br>Abertura robusta: %{y:.4f} m<extra></extra>'
#             ))
        
#         # Layout log-log
#         fig.update_layout(
#             title={
#                 'text': 'Relação Abertura-Comprimento (b-l)',
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
        
#         # Adicionar anotação com R²
#         fig.add_annotation(
#             x=0.95, y=0.05,
#             xref="paper", yref="paper",
#             text=f"R² = {fit_params['r_squared']:.3f}",
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
#         Plota comparação de intensidades P10 vs threshold
        
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
#             title='Comparação de Intensidades Size-Cognizant',
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
#         Plota comparação de espaçamentos médios
        
#         Args:
#             thresholds: Array de limiares
#             spacing_framfrat: Espaçamentos FRAMFRAT
#             spacing_scanline: Espaçamentos scanline
        
#         Returns:
#             Figura Plotly
#         """
#         fig = go.Figure()
        
#         # Espaçamento FRAMFRAT
#         fig.add_trace(go.Scatter(
#             x=thresholds,
#             y=spacing_framfrat,
#             mode='lines+markers',
#             name='FRAMFRAT',
#             line=dict(color='blue', width=2),
#             marker=dict(size=8),
#             hovertemplate='Threshold: %{x:.3f} m<br>Espaçamento: %{y:.2f} m<extra></extra>'
#         ))
        
#         # Espaçamento Scanline
#         fig.add_trace(go.Scatter(
#             x=thresholds,
#             y=spacing_scanline,
#             mode='lines+markers',
#             name='Scanline',
#             line=dict(color='red', width=2),
#             marker=dict(size=8),
#             hovertemplate='Threshold: %{x:.3f} m<br>Espaçamento: %{y:.2f} m<extra></extra>'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title='Comparação de Espaçamentos Médios',
#             xaxis_title='Limiar de tamanho (m)',
#             yaxis_title='Espaçamento médio (m)',
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
#             label1: Rótulo do primeiro conjunto
#             label2: Rótulo do segundo conjunto
#             variable: Nome da variável
        
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
#             title=f'Distribuição de {variable}',
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
#         Plota distribuições cumulativas comparativas
        
#         Args:
#             data1: Primeiro conjunto de dados
#             data2: Segundo conjunto de dados
#             label1: Rótulo do primeiro conjunto
#             label2: Rótulo do segundo conjunto
        
#         Returns:
#             Figura Plotly
#         """
#         fig = go.Figure()
        
#         # Calcular distribuições cumulativas
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
#             hovertemplate='Tamanho: %{x:.3f}<br>P(X≥x): %{y:.3f}<extra></extra>'
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=sorted2,
#             y=cum2_norm,
#             mode='lines',
#             name=label2,
#             line=dict(color='red', width=2),
#             hovertemplate='Tamanho: %{x:.3f}<br>P(X≥x): %{y:.3f}<extra></extra>'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title='Distribuição Cumulativa Complementar',
#             xaxis_title='Tamanho (m)',
#             yaxis_title='P(X ≥ x)',
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
#             connectivity: Matriz de adjacência
        
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
#             xaxis_title='Índice da fratura',
#             yaxis_title='Índice da fratura',
#             template='plotly_white'
#         )
        
#         return fig

#     def plot_3d_surface_density(self, x: np.ndarray, y: np.ndarray, 
#                             density: np.ndarray) -> go.Figure:
#         """
#         Plota superfície 3D de densidade de fraturas
        
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
#             colorbar=dict(title='Densidade (fraturas/m²)'),
#             hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Densidade: %{z:.3f}<extra></extra>'
#         )])
        
#         fig.update_layout(
#             title='Densidade Espacial de Fraturas',
#             scene=dict(
#                 xaxis_title='X (m)',
#                 yaxis_title='Y (m)',
#                 zaxis_title='Densidade (fraturas/m²)',
#                 camera=dict(
#                     eye=dict(x=1.5, y=1.5, z=1.5)
#                 )
#             ),
#             template='plotly_white'
#         )
        
#         return fig
        

        
































# import plotly.graph_objects as go
# import plotly.express as px
# import numpy as np
# import pandas as pd
# from typing import List, Dict, Optional
# import matplotlib.pyplot as plt
# import seaborn as sns

# class FractureVisualizer:
#     """Visualizador de fraturas e análises"""
    
#     def __init__(self, style: str = 'scientific'):
#         self.style = style
#         self.colors = px.colors.qualitative.Set2
        
#         # Configurar estilo matplotlib
#         if style == 'scientific':
#             plt.style.use('seaborn-v0_8-darkgrid')
#             sns.set_palette("husl")
    
#     def plot_power_law_fit(self, data: np.ndarray, fit_params: Dict) -> go.Figure:
#         """
#         Plota ajuste de lei de potência
        
#         Args:
#             data: Dados originais
#             fit_params: Parâmetros do ajuste
        
#         Returns:
#             Figura Plotly
#         """
#         # Calcular distribuição cumulativa
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
#             hovertemplate='Tamanho: %{x:.3f}<br>N(≥x): %{y}<extra></extra>'
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
#             name=f"Ajuste: N = {fit_params['coefficient']:.1f} × x^(-{fit_params['exponent']:.2f})",
#             line=dict(color='red', width=2, dash='solid'),
#             hovertemplate='Tamanho: %{x:.3f}<br>N(≥x): %{y:.1f}<extra></extra>'
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
#                 'text': 'Distribuição Power-Law de Tamanhos',
#                 'x': 0.5,
#                 'xanchor': 'center'
#             },
#             xaxis_title='Tamanho (m)',
#             yaxis_title='N(≥x) - Número cumulativo',
#             xaxis_type='log',
#             yaxis_type='log',
#             showlegend=True,
#             legend=dict(x=0.6, y=0.95),
#             hovermode='x unified',
#             template='plotly_white'
#         )
        
#         # Adicionar anotação com estatísticas
#         if 'r_squared' in fit_params:
#             annotation_text = f"R² = {fit_params['r_squared']:.3f}"
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
#             # 1. Visualização da Fratura (Linha/Retângulo)
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
#                         f'Orientação: {frac.orientation:.1f}°<extra></extra>'
#                     )
#                 ))
#             elif fracture_shape == 'rectangles':
#                 # Implementação simplificada de retângulo (apenas para visualização)
#                 # O DFN 2D gera linhas, mas o usuário pediu "Retângulos"
#                 # Usaremos a abertura para dar espessura à linha
                
#                 # Calcular o vetor normal à fratura
#                 dx = frac.x2 - frac.x1
#                 dy = frac.y2 - frac.y1
#                 length = np.sqrt(dx**2 + dy**2)
                
#                 if length > 0:
#                     # Vetor unitário perpendicular (normal)
#                     nx = -dy / length
#                     ny = dx / length
                    
#                     half_aperture = frac.aperture / 2.0
                    
#                     # Coordenadas dos 4 cantos do retângulo
#                     x_rect = [
#                         frac.x1 + nx * half_aperture,
#                         frac.x2 + nx * half_aperture,
#                         frac.x2 - nx * half_aperture,
#                         frac.x1 - nx * half_aperture,
#                         frac.x1 + nx * half_aperture # Fechar o polígono
#                     ]
#                     y_rect = [
#                         frac.y1 + ny * half_aperture,
#                         frac.y2 + ny * half_aperture,
#                         frac.y2 - ny * half_aperture,
#                         frac.y1 - ny * half_aperture,
#                         frac.y1 + ny * half_aperture # Fechar o polígono
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
#                             f'Orientação: {frac.orientation:.1f}°<extra></extra>'
#                         )
#                     ))
            
#             # 2. Visualização dos Centros
#             if show_centers or show_numbers:
#                 # Calcular o centro da fratura (ponto médio)
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
                
#                 # 3. Visualização da Numeração
#                 if show_numbers:
#                     fig.add_annotation(
#                         x=center_x,
#                         y=center_y,
#                         text=str(i + 1),
#                         showarrow=False,
#                         font=dict(
#                             size=12,
#                             color="red" # Cor viva para o número
#                         ),
#                         xshift=5, # Deslocamento para não ficar exatamente no centro
#                         yshift=5
#                     )
        
#         # O formato 'Discos' não se aplica a DFN 2D gerado por linhas, 
#         # mas a função plot_dfn_3d já usa discos.
#         # A opção 'lines' é o padrão para DFN 2D.
#         # Se o usuário selecionar 'Discos' em 2D, trataremos como 'lines' ou ignoraremos.
#         # Como a função é plot_dfn_2d, vamos focar em linhas e retângulos.
#         # A opção 'Discos' será tratada na função plot_dfn_3d se necessário, mas o foco é 2D.
#         # Por enquanto, se for 'discos', faremos o padrão 'lines' para 2D.
#         if fracture_shape == 'discs':
#             # Adicionar um aviso ou simplesmente usar 'lines'
#             pass
        
#         # Fim da lógica de visualização
        
#         # Adicionar bordas do domínio
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
    
#     def plot_dfn_3d(self, fractures: List, domain_size: tuple, 
#                     show_centers: bool = False, 
#                     show_numbers: bool = False) -> go.Figure:
#         """
#         Visualiza DFN 3D
        
#         Args:
#             fractures: Lista de Fracture3D
#             domain_size: (largura, altura, profundidade)
        
#         Returns:
#             Figura Plotly 3D
#         """
#         fig = go.Figure()
        
#         # Adicionar cada fratura como um disco
#         for i, frac in enumerate(fractures):
#             # Criar pontos do disco
#             theta = np.linspace(0, 2*np.pi, 20)
            
#             # Vetores no plano do disco
#             if abs(frac.normal[2]) < 0.99:
#                 v1 = np.cross(frac.normal, [0, 0, 1])
#             else:
#                 v1 = np.cross(frac.normal, [1, 0, 0])
#             v1 = v1 / np.linalg.norm(v1)
#             v2 = np.cross(frac.normal, v1)
            
#             # Pontos do círculo
#             x = frac.center[0] + frac.radius * (np.cos(theta)[:, np.newaxis] * v1[0] + 
#                                                 np.sin(theta)[:, np.newaxis] * v2[0]).flatten()
#             y = frac.center[1] + frac.radius * (np.cos(theta)[:, np.newaxis] * v1[1] + 
#                                                 np.sin(theta)[:, np.newaxis] * v2[1]).flatten()
#             z = frac.center[2] + frac.radius * (np.cos(theta)[:, np.newaxis] * v1[2] + 
#                                                 np.sin(theta)[:, np.newaxis] * v2[2]).flatten()
            
#             # Adicionar superfície do disco
#             fig.add_trace(go.Mesh3d(
#                 x=x, y=y, z=z,
#                 opacity=0.7,
#                 color=f'rgb({50+i*5 % 200}, {100+i*7 % 200}, {150+i*3 % 200})',
#                 hovertemplate=(
#                     f'Fratura {i+1}<br>'
#                     f'Raio: {frac.radius:.3f} m<br>'
#                     f'Abertura: {frac.aperture*1000:.2f} mm<br>'
#                     f'Dip: {frac.dip:.1f}°<br>'
#                     f'Dip Dir: {frac.dip_direction:.1f}°<extra></extra>'
#                 ),
#                 showscale=False
#             ))
            
#             # 2. Visualização dos Centros e Numeração
#             if show_centers or show_numbers:
#                 center_x, center_y, center_z = frac.center
                
#                 if show_centers:
#                     fig.add_trace(go.Scatter3d(
#                         x=[center_x],
#                         y=[center_y],
#                         z=[center_z],
#                         mode='markers',
#                         marker=dict(
#                             size=5,
#                             color='magenta', # Cor viva
#                             symbol='circle'
#                         ),
#                         name=f'Centro Fratura {i+1}',
#                         showlegend=False,
#                         hovertemplate=f'Centro Fratura {i+1}<extra></extra>'
#                     ))
                
#                 if show_numbers:
#                     fig.add_annotation(
#                         x=center_x,
#                         y=center_y,
#                         z=center_z,
#                         text=str(i + 1),
#                         showarrow=False,
#                         font=dict(
#                             size=12,
#                             color="red" # Cor viva para o número
#                         ),
#                         xshift=5, # Deslocamento para não ficar exatamente no centro
#                         yshift=5
#                     )
        
#         # Adicionar caixa do domínio
#         width, height, depth = domain_size
        
#         # Vértices da caixa
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
#                 line=dict(color='red', width=3),
#                 showlegend=False,
#                 hoverinfo='skip'
#             ))
        
#         # Layout
#         fig.update_layout(
#             title='Rede de Fraturas Discretas 3D',
#             scene=dict(
#                 xaxis_title='X (m)',
#                 yaxis_title='Y (m)',
#                 zaxis_title='Z (m)',
#                 aspectmode='data',
#                 camera=dict(
#                     eye=dict(x=1.5, y=1.5, z=1.5)
#                 )
#             ),
#             showlegend=False,
#             template='plotly_white'
#         )
        
#         return fig
    
#     def plot_rose_diagram(self, orientations: np.ndarray, bins: int = 36) -> go.Figure:
#         """
#         Cria diagrama de roseta para orientações
        
#         Args:
#             orientations: Array de orientações em graus
#             bins: Número de bins
        
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
#             hovertemplate='Direção: %{theta}°<br>Frequência: %{r}<extra></extra>'
#         ))
        
#         fig.update_layout(
#             title='Diagrama de Roseta - Orientações',
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
#         Cria estereograma para orientações 3D
        
#         Args:
#             dips: Ângulos de mergulho em graus
#             dip_directions: Direções de mergulho em graus
        
#         Returns:
#             Figura Plotly
#         """
#         # Converter para projeção estereográfica
#         dips_rad = np.radians(dips)
#         dirs_rad = np.radians(dip_directions)
        
#         # Projeção de Schmidt (equal area)
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
#                 colorbar=dict(title='Dip (°)')
#             ),
#             hovertemplate='Dip: %{marker.color:.1f}°<br>X: %{x:.3f}<br>Y: %{y:.3f}<extra></extra>'
#         ))
        
#         # Adicionar círculo unitário
#         theta = np.linspace(0, 2*np.pi, 100)
#         fig.add_trace(go.Scatter(
#             x=np.cos(theta),
#             y=np.sin(theta),
#             mode='lines',
#             line=dict(color='black', width=2),
#             showlegend=False,
#             hoverinfo='skip'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title='Estereograma (Projeção de Schmidt)',
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
#         Plota relação abertura-comprimento com ajuste
        
#         Args:
#             apertures: Array de aberturas
#             lengths: Array de comprimentos
#             fit_params: Parâmetros do ajuste (m, g)
        
#         Returns:
#             Figura Plotly
#         """
#         # Filtrar valores válidos
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
#             name=f"b = {fit_params['g']:.2e} × l^{fit_params['m']:.2f}",
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
#                 name=f"Robusto: b = {fit_params['g_robust']:.2e} × l^{fit_params['m_robust']:.2f}",
#                 line=dict(color='green', width=2, dash='dash'),
#                 hovertemplate='Comprimento: %{x:.3f} m<br>Abertura robusta: %{y:.4f} m<extra></extra>'
#             ))
        
#         # Layout log-log
#         fig.update_layout(
#             title={
#                 'text': 'Relação Abertura-Comprimento (b-l)',
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
        
#         # Adicionar anotação com R²
#         fig.add_annotation(
#             x=0.95, y=0.05,
#             xref="paper", yref="paper",
#             text=f"R² = {fit_params['r_squared']:.3f}",
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
#         Plota comparação de intensidades P10 vs threshold
        
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
#             title='Comparação de Intensidades Size-Cognizant',
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
#         Plota comparação de espaçamentos médios
        
#         Args:
#             thresholds: Array de limiares
#             spacing_framfrat: Espaçamentos FRAMFRAT
#             spacing_scanline: Espaçamentos scanline
        
#         Returns:
#             Figura Plotly
#         """
#         fig = go.Figure()
        
#         # Espaçamento FRAMFRAT
#         fig.add_trace(go.Scatter(
#             x=thresholds,
#             y=spacing_framfrat,
#             mode='lines+markers',
#             name='FRAMFRAT',
#             line=dict(color='blue', width=2),
#             marker=dict(size=8),
#             hovertemplate='Threshold: %{x:.3f} m<br>Espaçamento: %{y:.2f} m<extra></extra>'
#         ))
        
#         # Espaçamento Scanline
#         fig.add_trace(go.Scatter(
#             x=thresholds,
#             y=spacing_scanline,
#             mode='lines+markers',
#             name='Scanline',
#             line=dict(color='red', width=2),
#             marker=dict(size=8),
#             hovertemplate='Threshold: %{x:.3f} m<br>Espaçamento: %{y:.2f} m<extra></extra>'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title='Comparação de Espaçamentos Médios',
#             xaxis_title='Limiar de tamanho (m)',
#             yaxis_title='Espaçamento médio (m)',
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
#             label1: Rótulo do primeiro conjunto
#             label2: Rótulo do segundo conjunto
#             variable: Nome da variável
        
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
#             title=f'Distribuição de {variable}',
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
#         Plota distribuições cumulativas comparativas
        
#         Args:
#             data1: Primeiro conjunto de dados
#             data2: Segundo conjunto de dados
#             label1: Rótulo do primeiro conjunto
#             label2: Rótulo do segundo conjunto
        
#         Returns:
#             Figura Plotly
#         """
#         fig = go.Figure()
        
#         # Calcular distribuições cumulativas
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
#             hovertemplate='Tamanho: %{x:.3f}<br>P(X≥x): %{y:.3f}<extra></extra>'
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=sorted2,
#             y=cum2_norm,
#             mode='lines',
#             name=label2,
#             line=dict(color='red', width=2),
#             hovertemplate='Tamanho: %{x:.3f}<br>P(X≥x): %{y:.3f}<extra></extra>'
#         ))
        
#         # Layout
#         fig.update_layout(
#             title='Distribuição Cumulativa Complementar',
#             xaxis_title='Tamanho (m)',
#             yaxis_title='P(X ≥ x)',
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
#             connectivity: Matriz de adjacência
        
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
#             xaxis_title='Índice da fratura',
#             yaxis_title='Índice da fratura',
#             template='plotly_white'
#         )
        
#         return fig

#     def plot_3d_surface_density(self, x: np.ndarray, y: np.ndarray, 
#                             density: np.ndarray) -> go.Figure:
#         """
#         Plota superfície 3D de densidade de fraturas
        
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
#             colorbar=dict(title='Densidade (fraturas/m²)'),
#             hovertemplate='X: %{x:.2f}<br>Y: %{y:.2f}<br>Densidade: %{z:.3f}<extra></extra>'
#         )])
        
#         fig.update_layout(
#             title='Densidade Espacial de Fraturas',
#             scene=dict(
#                 xaxis_title='X (m)',
#                 yaxis_title='Y (m)',
#                 zaxis_title='Densidade (fraturas/m²)',
#                 camera=dict(
#                     eye=dict(x=1.5, y=1.5, z=1.5)
#                 )
#             ),
#             template='plotly_white'
#         )
        
#         return fig
        

        