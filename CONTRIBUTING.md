# Guia de Contribuição

Obrigado pelo interesse em contribuir com o DFN-Pro! 🎉

## 📋 Código de Conduta

Este projeto adota um código de conduta que esperamos que todos os participantes sigam. Por favor, seja respeitoso e construtivo em todas as interações.

## 🚀 Como Contribuir

### Reportando Bugs

1. Verifique se o bug já não foi reportado nas [Issues](https://github.com/wagnerdevete/dfn-pro/issues)
2. Se não encontrar, abra uma nova issue com:
   - Título descritivo
   - Passos para reproduzir
   - Comportamento esperado vs. observado
   - Screenshots (se aplicável)
   - Versão do Python e dependências

### Sugerindo Melhorias

1. Abra uma issue com a tag `enhancement`
2. Descreva claramente a funcionalidade proposta
3. Explique o caso de uso e benefícios

### Pull Requests

1. Fork o repositório
2. Clone seu fork: `git clone https://github.com/SEU_USER/dfn-pro.git`
3. Crie uma branch: `git checkout -b feature/minha-feature`
4. Faça suas alterações
5. Teste localmente: `streamlit run app.py`
6. Commit: `git commit -m "Adiciona minha feature"`
7. Push: `git push origin feature/minha-feature`
8. Abra um Pull Request

## 🧪 Testes

Antes de submeter um PR, certifique-se de que:

```bash
# Instalar dependências de teste
pip install pytest pytest-cov

# Executar testes
pytest tests/ -v --cov=modules
```

## 📝 Padrões de Código

- **Python**: Siga PEP 8
- **Docstrings**: Use formato Google
- **Commits**: Mensagens claras e descritivas
- **Nomes**: Variáveis e funções em inglês

### Exemplo de Docstring

```python
def calculate_powerlaw(lengths: np.ndarray, method: str = 'OLS') -> dict:
    """
    Calcula parâmetros de distribuição power-law.
    
    Args:
        lengths: Array de comprimentos de fraturas em metros
        method: Método de ajuste ('OLS' ou 'MLE')
    
    Returns:
        Dicionário com alpha, l_min, l_max, r_squared
    
    Raises:
        ValueError: Se lengths estiver vazio ou com valores negativos
    
    Example:
        >>> result = calculate_powerlaw(np.array([0.1, 0.5, 1.0, 2.0]))
        >>> print(result['alpha'])
        1.85
    """
```

## 📚 Documentação

Ao adicionar novas funcionalidades:
1. Atualize docstrings
2. Adicione exemplos em `docs/tutorials/`
3. Atualize o README se necessário

## ❓ Dúvidas

Abra uma issue com a tag `question` ou entre em contato: wagner.devete@ufpe.br

---

Obrigado por contribuir! 🙏
