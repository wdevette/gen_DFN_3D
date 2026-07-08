# 📁 Dados de Exemplo

Esta pasta contém dados de exemplo para testar o DFN-Pro.

## Arquivos

### `itapoama_fraturas.xlsx` (Tabela 7)
- **Descrição**: 34 fraturas principais do afloramento de Itapoama
- **Colunas**: ID_Fratura, Comprimento (mm), Abertura Média (mm)
- **Fonte**: Muchanga (2025)

### `itapoama_segmentos.xlsx` (Tabela 8)
- **Descrição**: 53 segmentos lineares com orientações
- **Colunas**: ID_Segmento, ID_Fratura, Comprimento (mm), Orientação (graus)
- **Fonte**: Muchanga (2025)

### `exemplo_scanline.csv`
- **Descrição**: Exemplo sintético de dados de scanline
- **Colunas**: Posição (m), Comprimento (m), Abertura (mm), Orientação (°)

## Formato Esperado

### FRAMFRAT (Excel)
```
| ID_Fratura | Comprimento (mm) | Abertura Média (mm) | Orientação (graus) |
|------------|------------------|---------------------|-------------------|
| F_1        | 450.5            | 2.3                 | 145.7             |
| F_2        | 320.1            | 1.8                 | 67.2              |
```

### Scanline (CSV)
```
Posição (m),Comprimento (m),Abertura (mm),Orientação (°)
0.5,0.45,2.3,145.7
1.2,0.32,1.8,67.2
```

## ⚠️ Notas

- Os dados de Itapoama são propriedade de Muchanga (2025) e UFPE
- Use apenas para fins acadêmicos e de teste
- Para uso comercial, consulte os autores
