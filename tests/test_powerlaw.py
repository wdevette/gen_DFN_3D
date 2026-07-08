"""
Testes unitários para módulo powerlaw_analysis
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from modules.powerlaw_analysis import PowerLawAnalyzer


class TestPowerLawAnalyzer:
    """Testes para PowerLawAnalyzer"""
    
    @pytest.fixture
    def sample_lengths(self):
        """Gera dados de exemplo com distribuição power-law"""
        np.random.seed(42)
        # Gerar dados power-law sintéticos (alpha ~ 1.8)
        u = np.random.uniform(0, 1, 100)
        alpha = 1.8
        l_min = 0.1
        lengths = l_min * (1 - u) ** (-1 / (alpha - 1))
        return lengths
    
    @pytest.fixture
    def analyzer(self):
        """Instância do analisador"""
        return PowerLawAnalyzer()
    
    def test_analyze_both_methods(self, analyzer, sample_lengths):
        """Testa análise com OLS e MLE"""
        results = analyzer.analyze_both_methods(
            lengths=sample_lengths,
            percentile_min=5.0,
            percentile_max=95.0,
            validate=True
        )
        
        # Verificar estrutura do resultado
        assert 'ols' in results
        assert 'mle' in results
        assert 'weighted' in results
        assert 'recommendation' in results
        assert 'l_min' in results
        assert 'l_max' in results
        
        # Verificar valores razoáveis
        assert 1.0 < results['ols']['alpha'] < 3.0
        assert 1.0 < results['mle']['alpha'] < 3.0
        assert results['l_min'] > 0
        assert results['l_max'] > results['l_min']
    
    def test_alpha_validation(self, analyzer, sample_lengths):
        """Testa validação do expoente alpha"""
        results = analyzer.analyze_both_methods(
            lengths=sample_lengths,
            validate=True
        )
        
        # Alpha deve estar no range aceitável
        assert results['ols']['validation']['is_valid'] or len(results['ols']['validation']['warnings']) > 0
    
    def test_get_params(self, analyzer, sample_lengths):
        """Testa obtenção de parâmetros pelo método selecionado"""
        analyzer.analyze_both_methods(lengths=sample_lengths)
        
        # Testar diferentes métodos
        for method in ['auto', 'OLS', 'MLE', 'weighted']:
            params = analyzer.get_params(method=method)
            assert params is not None
            assert hasattr(params, 'alpha')
            assert hasattr(params, 'l_min')
            assert hasattr(params, 'l_max')
    
    def test_empty_input(self, analyzer):
        """Testa comportamento com entrada vazia"""
        with pytest.raises(ValueError):
            analyzer.analyze_both_methods(lengths=np.array([]))
    
    def test_negative_values(self, analyzer):
        """Testa comportamento com valores negativos"""
        lengths = np.array([-1, 0.5, 1.0, 2.0])
        with pytest.raises(ValueError):
            analyzer.analyze_both_methods(lengths=lengths)


class TestPowerLawValidation:
    """Testes para validação de power-law"""
    
    def test_alpha_range(self):
        """Testa range aceitável de alpha"""
        # Alpha típico para fraturas: 1.2 - 2.8
        valid_alphas = [1.2, 1.5, 1.8, 2.0, 2.5, 2.8]
        invalid_alphas = [0.5, 1.0, 3.5, 4.0]
        
        for alpha in valid_alphas:
            assert 1.2 <= alpha <= 2.8, f"Alpha {alpha} deveria ser válido"
        
        for alpha in invalid_alphas:
            assert not (1.2 <= alpha <= 2.8), f"Alpha {alpha} deveria ser inválido"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
