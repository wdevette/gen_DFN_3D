# 🌐 gen_DFN_3D: Modelagem 3D de Redes de Fraturas Discretas

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxxx-blue.svg)](https://doi.org/)

<p align="center">
  <img src="docs/images/dfn_3d_example.png" alt="DFN 3D Example" width="600"/>
</p>

## 📋 Descrição

**gen_DFN_3D** é uma aplicação web open-source para modelagem estocástica de Redes de Fraturas Discretas (DFN) tridimensionais a partir de dados de afloramento. Desenvolvido como parte de dissertação de mestrado no Programa de Pós-Graduação em Engenharia Civil (PPGEC) da Universidade Federal de Pernambuco (UFPE).

### 🎯 Objetivo

Fornecer uma ferramenta gratuita, acessível e cientificamente rigorosa para caracterização de reservatórios naturalmente fraturados, integrando:

- Processamento de dados de imagens 2D (FRAMFRAT) e 1D (scanline)
- Análise estatística robusta (Power-Law, Fisher)
- Validação mecânica (D-L Scaling)
- Correções de vieses de amostragem
- Geração de modelos DFN  2D e 3D

---

## ✨ Funcionalidades

### 📥 Entrada de Dados
- **FRAMFRAT**: Importação direta de dados processados pelo sistema FRAMFRAT
- **Scanline**: Dados de scanline 1D com orientações e espaçamentos
- **Formatos**: Excel (.xlsx), CSV (.csv), TXT (.txt)

### 📊 Análise Estatística
| Análise | Descrição |
|---------|-----------|
| **Power-Law** | Ajuste OLS e MLE simultâneos com seleção automática |
| **Clustering** | Identificação de famílias via distribuição de Fisher |
| **D-L Scaling** | Validação mecânica baseada em Schultz et al. (2008) |
| **Intensidade** | Cálculo de P10, P20, P21, P30, P32, P33 |

### 🔧 Correções de Vieses
- **Terzaghi (1965)**: Correção de orientação para scanlines
- **Marrett (1996)**: Correção de truncamento de power-law
- **D-L Scaling Validation**: Remoção de fraturas mecanicamente inconsistentes

### 🎲 Geração de DFN
- Modelagem estocástica 2D e 3D
- Distribuição de tamanhos por power-law
- Orientações via distribuição de Fisher
- Visualização interativa 3D

---

## 🚀 Instalação

### Requisitos
- Python 3.9+
- pip ou conda

### Via pip
```bash
# Clonar repositório
git clone https://github.com/wagnerdevete/dfn-pro.git
cd dfn-pro

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar aplicação
streamlit run app.py
```

### Via Docker
```bash
docker pull wagnerdevete/dfn-pro:latest
docker run -p 8501:8501 wagnerdevete/dfn-pro
```

---

## 📖 Uso Rápido

### 1. Carregar Dados
```python
# Estrutura esperada do arquivo Excel/CSV
# | ID_Fratura | Comprimento (mm) | Abertura Média (mm) | Orientação (graus) |
# |------------|------------------|---------------------|-------------------|
# | F_1        | 450.5            | 2.3                 | 145.7             |
```

### 2. Configurar Parâmetros
- **Área da amostra** (m²)
- **Tipo de rocha** (para D-L Scaling)
- **Número de famílias** (automático ou manual)

### 3. Executar Análise
A aplicação executa automaticamente:
1. Detecção de unidades e conversão para metros
2. Análise power-law (OLS + MLE)
3. Identificação de famílias de fraturas
4. Correções de vieses (opcional)
5. Geração do modelo DFN 3D

---

## 📐 Fundamentação Teórica

### Power-Law Distribution
A distribuição de comprimentos de fraturas segue uma lei de potência:

$$N(L \geq l) = C \cdot l^{-\alpha}$$

Onde:
- $N$ = número cumulativo de fraturas
- $l$ = comprimento da fratura
- $\alpha$ = expoente (tipicamente 1.2 - 2.8)
- $C$ = constante de proporcionalidade

### D-L Scaling (Schultz et al., 2008)
Relação entre abertura máxima e comprimento:

$$D_{max} = a \cdot L^n$$

Onde:
- $D_{max}$ = abertura/deslocamento máximo
- $L$ = comprimento da fratura
- $n$ = 0.5 para juntas (modo I)
- $a$ = coeficiente dependente da litologia

| Tipo de Rocha | Coeficiente a |
|---------------|---------------|
| Rochas Vulcânicas | 0.078 |
| Arenito | 0.022 |
| Calcário | 0.025 |
| Granito | 0.035 |
| Folhelho | 0.012 |

### Correção de Marrett (1996)
Fator de correção para truncamento:

$$F = \frac{l_{max}^{1-\alpha} - l_{min}^{1-\alpha}}{(1-\alpha) \cdot l_{max}^{1-\alpha}}$$

---

## 📁 Estrutura do Projeto

```
dfn-pro/
├── app.py                       # Aplicação principal Streamlit
├── func_tools.py                # Funções auxiliares
├── requirements.txt             # Dependências Python
├── Dockerfile                   # Container Docker
├── LICENSE                      # Licença MIT
├── README.md                    # Este arquivo
│
├── assets/
|   ├── fram.fract/svg
|   ├── fram_fractt.png           # icon da web page do gen_DFN_3D
|
├── modules/                      # Módulos Python
│   ├── __init__.py
│   ├── io_fractures.py           # Carregamento de dados
│   ├── powerlaw_analysis.py      # Análise power-law (OLS+MLE)
│   ├── powerlaw_fits.py          # Ajustes de distribuição
│   ├── orientation_clustering.py # Clustering de orientações
│   ├── bias_corrections.py       # Correções de vieses
│   ├── bias_visualizations.py    # Visualizações de correções
│   ├── intensity_spacing.py      # Cálculo de intensidades
│   ├── dfn_generator.py          # Gerador de DFN 2D/3D
│   ├── visualizations.py         # Gráficos e plots
│   ├── results_exporter.py       # Exportação de resultados
│
├── data/                         # Dados de exemplo para validação da dissertação
|   ├── README.md                 # Descrição dos dados usados para validação e dados ficticios
│   ├── itapoama_fraturas.xlsx    # Tabela 7 validação da dissertação MUCHANGA (2025)
│   ├── itapoama_segmentos.xlsx   # Tabela 8 validação dissertação MUCHANGA (2025)
|   ├── scanline_dataset.xlsx     # Dados ficticios gerados para teste (gerado com IA)
│
├── docs/                         # Documentação
│   ├── methodology.md            # Matodologia usada desdo o processamento até a geração de DFNs
│
└── tests/                        # Testes unitários
    ├── test_powerlaw.py

```

---

## 📊 Caso de Estudo: Itapoama (PE)

A metodologia foi validada com dados do afloramento de traquito da Praia de Itapoama, Pernambuco, Brasil.

### Dados de Entrada
- **34 fraturas principais** (Tabela 7 - Muchanga, 2025)
- **53 segmentos lineares** (Tabela 8 - Muchanga, 2025)
- **Área**: 8,34 m²
- **Litologia**: Traquito (rocha vulcânica)

### Resultados
| Parâmetro | Valor |
|-----------|-------|
| Expoente α | 1.80 |
| l_min | 0.107 m |
| l_max | 18.105 m |
| D-L Validation | 100% válidas |
| P₂₁ bruto | 3.65 m/m² |
| P₂₁ corrigido (Marrett) | 4.78 m/m² (+31%) |
| P₃₂ estimado | 5.73 - 7.49 m²/m³ |
| Classificação (Nelson) | Alta intensidade |

### Famílias Identificadas
| Família | Orientação Média | Desvio Padrão | Proporção |
|---------|------------------|---------------|-----------|
| NE-SW | 66° | ±18° | 67.9% |
| NW-SE | 161° | ±24° | 32.1% |

---

## 🔬 Referências Científicas

### Fundamentais
- **Nelson, R.A. (2001)**. Geologic Analysis of Naturally Fractured Reservoirs. Gulf Professional Publishing.
- **Bonnet, E. et al. (2001)**. Scaling of fracture systems in geological media. Reviews of Geophysics, 39(3), 347-383.
- **Dershowitz, W.S. & Einstein, H.H. (1988)**. Characterizing rock joint geometry with joint system models. Rock Mechanics and Rock Engineering, 21(1), 21-51.

### Metodológicas
- **Marrett, R. (1996)**. Aggregate properties of fracture populations. Journal of Structural Geology, 18(2-3), 169-178.
- **Schultz, R.A. et al. (2008)**. Dependence of displacement–length scaling relations for fractures and deformation bands on the volumetric changes across them. Journal of Structural Geology, 30(11), 1405-1411.
- **Clauset, A. et al. (2009)**. Power-law distributions in empirical data. SIAM Review, 51(4), 661-703.
- **Terzaghi, R.D. (1965)**. Sources of error in joint surveys. Geotechnique, 15(3), 287-304.

### Dados
- **Muchanga, A. (2025)**. Sistema FRAMFRAT para processamento de imagens de fraturas. Tese de Doutorado, UFPE.

---

## 👨‍💻 Autor

**Wagner José Devete**
- 📧 Email: wagner.devete@ufpe.br / wagnerdevete@gmail.com
- 🎓 Mestrando em Engenharia Civil - UFPE
- 🔬 Área: Reservatórios Naturalmente Fraturados

### Orientadores
- **Prof. Dr. Igor Gomes** (Orientador)
- **Prof. Dr. Tiago Miranda** (Coorientador)

---

## 📄 Citação

Se utilizar este software em sua pesquisa, por favor cite:

```bibtex
@mastersthesis{devete2026dfn,
  author  = {Devete, Wagner José},
  title   = {Modelagem 3D de Redes de Fraturas Discretas a partir de Dados de Scanline: Aplicação em Simulação de Reservatórios Fraturados no Setor Petrolífero},
  school  = {Universidade Federal de Pernambuco},
  year    = {2026},
  address = {Recife, Brasil},
  type    = {Dissertação de Mestrado}
}
```

---

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor, leia [CONTRIBUTING.md](CONTRIBUTING.md) para detalhes sobre o processo de submissão de pull requests.

### Como Contribuir
1. Fork o projeto
2. Crie sua branch (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

---

## 🙏 Agradecimentos

- A DEUS por tudo que fez, faz e fará por mim...
- UFPE/PPGEC pelo suporte acadêmico
- Prof. Igor Gomes (UFPE) & Prof. Tiago Miranda (UFPE) pela orientação conjunta.
- Prof. Armando Muchanga pelos dados do FRAMFRAT
- Comunidade open-source pelas bibliotecas utilizadas
- A CAPES pelo apoio concedido.

---

<p align="center">
  <b>Desenvolvido com ❤️ na UFPE</b><br>
  <i>Recife, Pernambuco - Brasil</i>
</p>
