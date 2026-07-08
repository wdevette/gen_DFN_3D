import streamlit as st


def force_dark_plotly_layout(fig):
    """
    Força um layout escuro no figure plotly, preservando
    polar.radialaxis.visible e angularaxis.direction/rotation
    e o título/showlegend/height (se já existirem).
    """
    if fig is None:
        return fig

    # captura valores que queremos preservar (se existirem)
    try:
        preserve_radial_visible = fig.layout.polar.radialaxis.visible
    except Exception:
        preserve_radial_visible = True

    try:
        preserve_angular_direction = fig.layout.polar.angularaxis.direction
    except Exception:
        preserve_angular_direction = "clockwise"

    try:
        preserve_angular_rotation = fig.layout.polar.angularaxis.rotation
    except Exception:
        preserve_angular_rotation = 90

    # estilo escuro base (não sobrescreve direction/rotation/visible)
    dark_style = dict(
        paper_bgcolor="#0f1112",
        plot_bgcolor="#0f1112",
        font=dict(size=12, color="white"),
        polar=dict(
            bgcolor="#111316",
            radialaxis=dict(
                tickfont=dict(color="white"),
                gridcolor="#333333",
                linecolor="white",
                visible=preserve_radial_visible
            ),
            angularaxis=dict(
                tickfont=dict(color="white"),
                gridcolor="#333333",
                linecolor="white",
                direction=preserve_angular_direction,
                rotation=preserve_angular_rotation
            )
        ),
        legend=dict(font=dict(color="white"), bgcolor="rgba(0,0,0,0.2)")
    )

    # aplica o estilo (merge simples)
    fig.update_layout(**dark_style)

    # garante título/fontes legíveis (se houver título, mantemos texto e só forçamos cor)
    if fig.layout.title:
        fig.layout.title.font = dict(color="white", size=14)

    return fig



# apresentação de mensagens mais trabalhadas: erro, informações e sucesso
def show_error(message: str,
               bg: str="#341e21",
               border: str ="#A96B73",
               raise_exception: bool = False) -> None: #gerador de alertas com o estilo e cores 
    
    st.markdown(f"""
        <div style='background-color:{bg}; padding:15px; border-radius:5px; border-left:5px solid {border};'>
            <span style='color:#ffffff;'>{message}</span>
        </div>""", unsafe_allow_html=True)


#Função modificada para exibição de notificações de "informações"  
def show_info(message: str,
               bg: str="#132638",
               border: str ="#2c7be5") -> None: #gerador de alertas com o estilo e cores 
        
    st.markdown(f"""
        <div style='background-color:{bg}; padding:15px; border-radius:5px; border-left:5px solid {border};'>
            <span style='color:#ffffff;'>{message}</span>
        </div>""", unsafe_allow_html=True)
    

#Função modificada para exibição de notificações de "sucesso"
def show_success(message: str,
                bg: str= '#1b4332',
                border:str= '#2dc653') -> None: #gerador de alertas com o estilo e cores 
    
    st.markdown(f"""
        <div style='background-color:{bg}; padding:15px; border-radius:5px; border-left:5px solid {border};'>
            <span style='color:#ffffff;'>{message}</span>
        </div>""", unsafe_allow_html=True)
