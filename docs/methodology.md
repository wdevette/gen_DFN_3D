# 📐 Metodologia DFN-Pro

## Visão Geral do Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Dados Brutos   │ -> │  Pré-Processo   │ -> │ Análise Power-  │
│  (FRAMFRAT/     │    │  & Conversão    │    │    Law          │
│   Scanline)     │    │  para Metros    │    │  (OLS + MLE)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    DFN 3D       │ <- │   Correções     │ <- │  Clustering     │
│   Geração       │    │   de Vieses     │    │  de Famílias    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 1. Entrada de Dados

### 1.1 Formato FRAMFRAT

O sistema FRAMFRAT (Muchanga, 2025) processa imagens de afloramento e exporta:

| Coluna | Descrição | Unidade |
|--------|-----------|---------|
| ID_Fratura | Identificador único | - |
| Comprimento (mm) | Comprimento da fratura | mm |
| Abertura Média (mm) | Abertura média | mm |
| Orientação (graus) | Azimute da fratura | ° |

### 1.2 Formato Scanline

Dados de scanline 1D seguem o formato:

| Coluna | Descrição | Unidade |
|--------|-----------|---------|
| Posição (m) | Posição ao longo da linha | m |
| Comprimento (m) | Trace length | m |
| Orientação (°) | Azimute | ° |

### 1.3 Conversão Automática de Unidades

O sistema detecta automaticamente a unidade pelos valores:
- Comprimentos > 100 → assume mm → converte para m
- Comprimentos 10-100 → assume mm → converte para m
- Comprimentos < 10 → assume já em metros

---

## 2. Análise de Lei de Potência

### 2.1 Distribuição Power-Law

A distribuição cumulativa complementar (CCDF) de comprimentos:

$$P(L \geq l) = \left(\frac{l}{l_{min}}\right)^{-\alpha+1}$$

Onde:
- $\alpha$ = expoente da power-law
- $l_{min}$ = comprimento mínimo do ajuste

### 2.2 Métodos de Ajuste

#### OLS (Mínimos Quadrados Ordinários)
```
log(N) = log(C) - α × log(L)
```
- Rápido e intuitivo
- Fornece R² para avaliação

#### MLE (Máxima Verossimilhança)
$$\hat{\alpha} = 1 + n \left[ \sum_{i=1}^{n} \ln\frac{x_i}{x_{min}} \right]^{-1}$$

- Estatisticamente mais robusto
- Clauset et al. (2009)

### 2.3 Seleção de Método

| Condição | Método Recomendado |
|----------|-------------------|
| Δα < 0.2 | OLS |
| Apenas um válido | O válido |
| Ambos válidos, Δα ≥ 0.2 | Média Ponderada |

---

## 3. Validação D-L Scaling

### 3.1 Relação Displacement-Length

Baseado em Schultz et al. (2008):

$$D_{max} = a \cdot L^n$$

Para juntas (modo I): $n = 0.5$

### 3.2 Coeficientes por Litologia

| Litologia | Coeficiente a | Referência |
|-----------|---------------|------------|
| Rochas Vulcânicas | 0.078 | Ethiopian dikes |
| Traquito | 0.078 | Similar a basalto |
| Basalto | 0.078 | Ethiopian dikes |
| Arenito | 0.022 | - |
| Calcário | 0.025 | - |
| Granito | 0.035 | - |
| Folhelho | 0.012 | - |

### 3.3 Critério de Validação

Uma fratura é válida se:

$$\frac{1}{tolerance} \leq \frac{b_{obs}}{b_{exp}} \leq tolerance$$

Onde:
- $b_{obs}$ = abertura observada
- $b_{exp} = a \cdot \sqrt{L}$ = abertura esperada
- $tolerance$ = 5 (padrão, moderado)

---

## 4. Correções de Vieses

### 4.1 Terzaghi (1965) - Orientação

Para scanlines, fraturas paralelas à linha são sub-representadas:

$$w_i = \frac{1}{\cos(\theta_i)}$$

Onde $\theta$ = ângulo entre fratura e scanline.

**Blind zone**: Fraturas com θ < 5° são removidas.

### 4.2 Marrett (1996) - Truncamento

Corrige a subestimação por fraturas abaixo de $l_{min}$:

$$F = \frac{l_{max}^{1-\alpha} - l_{min}^{1-\alpha}}{(1-\alpha) \cdot l_{max}^{1-\alpha}}$$

| α | Fator Típico |
|---|-------------|
| 1.5 | 1.15× |
| 1.8 | 1.31× |
| 2.0 | 1.50× |
| 2.5 | 2.50× |

---

## 5. Clustering de Orientações

### 5.1 Distribuição de Fisher

Concentração de vetores direcionais:

$$f(\theta) = \frac{\kappa}{4\pi \sinh(\kappa)} e^{\kappa \cos(\theta)}$$

Onde $\kappa$ = parâmetro de concentração (Fisher, 1953).

### 5.2 K-Means em Vetores Normais

1. Converter azimutes para vetores normais 3D
2. Aplicar K-means clustering
3. Ajustar parâmetros Fisher por cluster
4. Extrair estatísticas (média, desvio, kappa)

---

## 6. Cálculo de Intensidades

### 6.1 Índices 1D (Scanline)

| Índice | Fórmula | Descrição |
|--------|---------|-----------|
| P10 | N / L_scan | Fraturas por metro linear |
| P11 | Σl / L_scan | Comprimento por metro linear |

### 6.2 Índices 2D (FRAMFRAT)

| Índice | Fórmula | Descrição |
|--------|---------|-----------|
| P20 | N / A | Fraturas por m² |
| P21 | ΣL / A | Comprimento por m² |
| P22 | ΣAf / A | Área de fraturas por m² |

### 6.3 Conversão 2D → 3D

| Conversão | Fator | Referência |
|-----------|-------|------------|
| P10 → P30 | 2.5× (geométrico) | Dershowitz (1988) |
| P21 → P32 | π/2 (C23 isotrópico) | Wang (2005) |

---

## 7. Geração de DFN

### 7.1 Processo Estocástico

1. **Número de fraturas**: Poisson(λ = P30 × V)
2. **Tamanhos**: Power-law(α, l_min, l_max)
3. **Posições**: Uniforme no volume
4. **Orientações**: Fisher(μ, κ) por família

### 7.2 Classificação (Nelson, 2001)

| P32 (m²/m³) | Classificação |
|-------------|---------------|
| < 1.0 | Baixa intensidade |
| 1.0 - 3.0 | Moderada |
| 3.0 - 6.0 | Alta |
| > 6.0 | Muito alta |

---

## Referências

1. Bonnet, E. et al. (2001). Scaling of fracture systems. Reviews of Geophysics.
2. Clauset, A. et al. (2009). Power-law distributions. SIAM Review.
3. Dershowitz, W.S. & Einstein, H.H. (1988). Rock joint geometry. RMRE.
4. Fisher, R.A. (1953). Dispersion on a sphere. Proc. R. Soc. London.
5. Marrett, R. (1996). Aggregate properties of fracture populations. JSG.
6. Muchanga, A. (2025). Sistema FRAMFRAT. Tese UFPE.
7. Nelson, R.A. (2001). Geologic Analysis of NFRs. Gulf Publishing.
8. Schultz, R.A. et al. (2008). Displacement-length scaling. JSG.
9. Terzaghi, R.D. (1965). Sources of error in joint surveys. Geotechnique.
