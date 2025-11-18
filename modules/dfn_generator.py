import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class Fracture2D:
    """Representa uma fratura 2D - dimensões em mm"""
    x1: float
    y1: float
    x2: float
    y2: float
    length: float  # mm
    aperture: float  # mm
    orientation: float  # graus
    family: int = 0  # ID da família/set
    
    def to_dict(self):
        return {
            'x1': self.x1, 'y1': self.y1,
            'x2': self.x2, 'y2': self.y2,
            'length': self.length,
            'aperture': self.aperture,
            'orientation': self.orientation,
            'family': self.family
        }

@dataclass
class Fracture3D:
    """Representa uma fratura 3D como um disco - dimensões em mm"""
    center: np.ndarray  # [x, y, z] em mm
    normal: np.ndarray  # vetor normal
    radius: float  # mm
    aperture: float  # mm
    dip: float  # graus
    dip_direction: float  # graus
    family: int = 0  # ID da família/set
    
    def to_dict(self):
        return {
            'center': self.center.tolist(),
            'normal': self.normal.tolist(),
            'radius': float(self.radius),
            'aperture': float(self.aperture),
            'dip': float(self.dip),
            'dip_direction': float(self.dip_direction),
            'family': int(self.family)
        }

@dataclass
class FractureFamily:
    """Representa uma família/set de fraturas com orientação preferencial"""
    orientation_mean: float  # graus (para 2D) ou dip para 3D
    orientation_std: float   # desvio padrão
    dip_dir_mean: float = 0.0  # apenas para 3D
    dip_dir_std: float = 20.0  # apenas para 3D
    weight: float = 1.0  # peso relativo da família

class DFNGenerator:
    """Gerador de Discrete Fracture Networks com suporte a famílias"""
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            np.random.seed(seed)
    
    def generate_2d_dfn(self, params: Dict, domain_size: Tuple[float, float],
                       n_fractures: Optional[int] = None,
                       families: Optional[List[FractureFamily]] = None) -> List[Fracture2D]:
        """
        Gera DFN 2D baseado em parâmetros estatísticos
        ALTERAÇÃO: Dimensões em mm, suporte a famílias
        
        Args:
            params: Parâmetros da distribuição (expoente, coeficiente, etc)
            domain_size: (largura, altura) do domínio em mm
            n_fractures: Número de fraturas (None = calcular por P21)
            families: Lista de famílias de fraturas (default: 2 famílias)
        
        Returns:
            Lista de fraturas 2D
        """
        width, height = domain_size
        area = width * height
        
        # Configurar famílias padrão se não fornecidas
        if families is None:
            # Default: 2 famílias com orientações ortogonais
            families = [
                FractureFamily(
                    orientation_mean=params.get('orientation_mean', 0.0),
                    orientation_std=params.get('orientation_std', 10.0),
                    weight=0.6
                ),
                FractureFamily(
                    orientation_mean=params.get('orientation_mean', 0.0) + 90.0,
                    orientation_std=params.get('orientation_std', 10.0),
                    weight=0.4
                )
            ]
        
        # Determinar número de fraturas
        if n_fractures is None:
            if 'p21' in params:
                mean_length = params.get('mean_length', 1.0)
                n_fractures = int(params['p21'] * area / mean_length)
            else:
                n_fractures = 100
        
        # Distribuir fraturas entre famílias
        total_weight = sum(f.weight for f in families)
        fractures_per_family = [
            int(n_fractures * (f.weight / total_weight)) 
            for f in families
        ]
        
        fractures = []
        
        for family_id, (family, n_fam) in enumerate(zip(families, fractures_per_family)):
            for _ in range(n_fam):
                # Gerar comprimento da distribuição power-law (em mm)
                length = self._sample_power_law(
                    params['exponent'],
                    params['x_min'],
                    params.get('x_max', width/2)
                )
                
                # Gerar abertura da relação b = g * l^m (em mm)
                if 'g' in params and 'm' in params:
                    aperture = params['g'] * length ** params['m']
                else:
                    aperture = length * 0.001  # Default: 0.1% do comprimento
                
                # Gerar orientação da família
                orientation = np.random.normal(
                    family.orientation_mean,
                    family.orientation_std
                )
                
                # Converter para radianos
                angle_rad = np.radians(orientation)
                
                # Gerar centro aleatório (em mm)
                cx = np.random.uniform(0, width)
                cy = np.random.uniform(0, height)
                
                # Calcular extremidades
                dx = length * np.cos(angle_rad) / 2
                dy = length * np.sin(angle_rad) / 2
                
                fracture = Fracture2D(
                    x1=cx - dx,
                    y1=cy - dy,
                    x2=cx + dx,
                    y2=cy + dy,
                    length=length,
                    aperture=aperture,
                    orientation=orientation,
                    family=family_id
                )
                
                fractures.append(fracture)
        
        return fractures
    
    def generate_3d_dfn(self, params: Dict, domain_size: Tuple[float, float, float],
                       n_fractures: Optional[int] = None,
                       families: Optional[List[FractureFamily]] = None) -> List[Fracture3D]:
        """
        Gera DFN 3D com fraturas como discos
        ALTERAÇÃO: Dimensões em mm, suporte a famílias
        
        Args:
            params: Parâmetros da distribuição
            domain_size: (largura, altura, profundidade) do domínio em mm
            n_fractures: Número de fraturas
            families: Lista de famílias de fraturas (default: 2 famílias)
        
        Returns:
            Lista de fraturas 3D
        """
        width, height, depth = domain_size
        volume = width * height * depth
        
        # Configurar famílias padrão se não fornecidas
        if families is None:
            # Default: 2 famílias com orientações diferentes
            families = [
                FractureFamily(
                    orientation_mean=params.get('dip_mean', 45.0),
                    orientation_std=params.get('dip_std', 10.0),
                    dip_dir_mean=params.get('dip_dir_mean', 90.0),
                    dip_dir_std=params.get('dip_dir_std', 20.0),
                    weight=0.6
                ),
                FractureFamily(
                    orientation_mean=params.get('dip_mean', 45.0),
                    orientation_std=params.get('dip_std', 10.0),
                    dip_dir_mean=params.get('dip_dir_mean', 90.0) + 90.0,
                    dip_dir_std=params.get('dip_dir_std', 20.0),
                    weight=0.4
                )
            ]
        
        # Determinar número de fraturas
        if n_fractures is None:
            if 'p32' in params:
                mean_area = params.get('mean_area', 1.0)
                n_fractures = int(params['p32'] * volume / mean_area)
            else:
                n_fractures = 100
        
        # Distribuir fraturas entre famílias
        total_weight = sum(f.weight for f in families)
        fractures_per_family = [
            int(n_fractures * (f.weight / total_weight)) 
            for f in families
        ]
        
        fractures = []
        
        for family_id, (family, n_fam) in enumerate(zip(families, fractures_per_family)):
            for _ in range(n_fam):
                # Gerar raio da distribuição (em mm)
                radius = self._sample_power_law(
                    params['exponent'],
                    params['x_min'],
                    params.get('x_max', min(domain_size)/4)
                ) / 2
                
                # Gerar abertura (em mm)
                if 'g' in params and 'm' in params:
                    aperture = params['g'] * (2*radius) ** params['m']
                else:
                    aperture = radius * 0.002
                
                # Gerar orientação da família
                dip = np.random.normal(family.orientation_mean, family.orientation_std)
                dip = np.clip(dip, 0, 90)
                
                dip_direction = np.random.normal(family.dip_dir_mean, family.dip_dir_std)
                
                # Calcular vetor normal
                dip_rad = np.radians(dip)
                dd_rad = np.radians(dip_direction)
                
                normal = np.array([
                    np.sin(dip_rad) * np.sin(dd_rad),
                    np.sin(dip_rad) * np.cos(dd_rad),
                    np.cos(dip_rad)
                ])
                
                # Gerar centro (em mm)
                center = np.array([
                    np.random.uniform(0, width),
                    np.random.uniform(0, height),
                    np.random.uniform(0, depth)
                ])
                
                fracture = Fracture3D(
                    center=center,
                    normal=normal,
                    radius=radius,
                    aperture=aperture,
                    dip=dip,
                    dip_direction=dip_direction,
                    family=family_id
                )
                
                fractures.append(fracture)
        
        return fractures
    
    def _sample_power_law(self, exponent: float, x_min: float, 
                         x_max: float) -> float:
        """
        Amostra de uma distribuição power-law truncada
        
        Args:
            exponent: Expoente da lei de potência
            x_min: Valor mínimo
            x_max: Valor máximo
        
        Returns:
            Valor amostrado
        """
        if exponent == 1:
            return x_min * np.exp(np.random.uniform(0, np.log(x_max/x_min)))
        else:
            u = np.random.uniform(0, 1)
            if exponent > 1:
                a = 1 - exponent
                term1 = x_min**a
                term2 = x_max**a
                x = (term1 + u * (term2 - term1))**(1/a)
            else:
                while True:
                    x = np.random.uniform(x_min, x_max)
                    y = np.random.uniform(0, x_min**(-exponent))
                    if y <= x**(-exponent):
                        break
            return x
    
    def calculate_connectivity(self, fractures: List, threshold: float = 0.1) -> Dict:
        """
        Calcula métricas de conectividade da rede
        
        Args:
            fractures: Lista de fraturas
            threshold: Distância máxima para considerar conexão (em mm)
        
        Returns:
            Métricas de conectividade
        """
        n = len(fractures)
        if n < 2:
            return {'connectivity': 0, 'clusters': 1}
        
        # Matriz de adjacência
        connections = np.zeros((n, n))
        
        if isinstance(fractures[0], Fracture2D):
            for i in range(n):
                for j in range(i+1, n):
                    if self._check_intersection_2d(fractures[i], fractures[j], threshold):
                        connections[i, j] = 1
                        connections[j, i] = 1
        else:
            for i in range(n):
                for j in range(i+1, n):
                    if self._check_intersection_3d(fractures[i], fractures[j], threshold):
                        connections[i, j] = 1
                        connections[j, i] = 1
        
        # Calcular métricas
        n_connections = np.sum(connections) / 2
        connectivity = n_connections / (n * (n-1) / 2) if n > 1 else 0
        
        # Contar clusters
        visited = np.zeros(n, dtype=bool)
        n_clusters = 0
        
        for i in range(n):
            if not visited[i]:
                n_clusters += 1
                self._dfs(i, connections, visited)
        
        return {
            'connectivity': connectivity,
            'n_connections': int(n_connections),
            'n_clusters': n_clusters,
            'percolation': n_clusters == 1
        }
    
    def _check_intersection_2d(self, f1: Fracture2D, f2: Fracture2D, 
                              threshold: float) -> bool:
        """Verifica interseção entre duas fraturas 2D"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        A = np.array([f1.x1, f1.y1])
        B = np.array([f1.x2, f1.y2])
        C = np.array([f2.x1, f2.y1])
        D = np.array([f2.x2, f2.y2])
        
        return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
    def _check_intersection_3d(self, f1: Fracture3D, f2: Fracture3D,
                              threshold: float) -> bool:
        """Verifica interseção entre duas fraturas 3D (discos)"""
        dist = np.linalg.norm(f1.center - f2.center)
        
        if dist > f1.radius + f2.radius + threshold:
            return False
        
        cos_angle = np.abs(np.dot(f1.normal, f2.normal))
        
        if cos_angle > 0.99:
            return False
        
        return True
    
    def _dfs(self, node: int, connections: np.ndarray, visited: np.ndarray):
        """Busca em profundidade para encontrar componentes conectados"""
        visited[node] = True
        for neighbor in range(len(connections)):
            if connections[node, neighbor] and not visited[neighbor]:
                self._dfs(neighbor, connections, visited)









# import numpy as np
# import pandas as pd
# from typing import Dict, List, Tuple, Optional
# from dataclasses import dataclass
# import json

# @dataclass
# class Fracture2D:
#     """Representa uma fratura 2D"""
#     x1: float
#     y1: float
#     x2: float
#     y2: float
#     length: float
#     aperture: float
#     orientation: float
    
#     def to_dict(self):
#         return {
#             'x1': self.x1, 'y1': self.y1,
#             'x2': self.x2, 'y2': self.y2,
#             'length': self.length,
#             'aperture': self.aperture,
#             'orientation': self.orientation
#         }

# @dataclass
# class Fracture3D:
#     """Representa uma fratura 3D como um disco"""
#     center: np.ndarray  # [x, y, z]
#     normal: np.ndarray  # vetor normal
#     radius: float
#     aperture: float
#     dip: float
#     dip_direction: float
    
#     def to_dict(self):
#         return {
#             'center': self.center.tolist(),
#             'normal': self.normal.tolist(),
#             'radius': float(self.radius),
#             'aperture': float(self.aperture),
#             'dip': float(self.dip),
#             'dip_direction': float(self.dip_direction)
#         }

# class DFNGenerator:
#     """Gerador de Discrete Fracture Networks"""
    
#     def __init__(self, seed: Optional[int] = None):
#         if seed:
#             np.random.seed(seed)
    
#     def generate_2d_dfn(self, params: Dict, domain_size: Tuple[float, float],
#                        n_fractures: Optional[int] = None) -> List[Fracture2D]:
#         """
#         Gera DFN 2D baseado em parâmetros estatísticos
        
#         Args:
#             params: Parâmetros da distribuição (expoente, coeficiente, etc)
#             domain_size: (largura, altura) do domínio
#             n_fractures: Número de fraturas (None = calcular por P21)
        
#         Returns:
#             Lista de fraturas 2D
#         """
#         width, height = domain_size
#         area = width * height
        
#         # Determinar número de fraturas
#         if n_fractures is None:
#             if 'p21' in params:
#                 # Estimar baseado em P21
#                 mean_length = params.get('mean_length', 1.0)
#                 n_fractures = int(params['p21'] * area / mean_length)
#             else:
#                 n_fractures = 100  # Default
        
#         fractures = []
        
#         for _ in range(n_fractures):
#             # Gerar comprimento da distribuição power-law
#             length = self._sample_power_law(
#                 params['exponent'],
#                 params['x_min'],
#                 params.get('x_max', width/2)
#             )
            
#             # Gerar abertura da relação b = g * l^m
#             if 'g' in params and 'm' in params:
#                 aperture = params['g'] * length ** params['m']
#             else:
#                 aperture = length * 0.001  # Default: 1mm por metro
            
#             # Gerar orientação
#             if 'orientation_mean' in params:
#                 orientation = np.random.normal(
#                     params['orientation_mean'],
#                     params.get('orientation_std', 10)
#                 )
#             else:
#                 orientation = np.random.uniform(0, 180)
            
#             # Converter para radianos
#             angle_rad = np.radians(orientation)
            
#             # Gerar centro aleatório
#             cx = np.random.uniform(0, width)
#             cy = np.random.uniform(0, height)
            
#             # Calcular extremidades
#             dx = length * np.cos(angle_rad) / 2
#             dy = length * np.sin(angle_rad) / 2
            
#             fracture = Fracture2D(
#                 x1=cx - dx,
#                 y1=cy - dy,
#                 x2=cx + dx,
#                 y2=cy + dy,
#                 length=length,
#                 aperture=aperture,
#                 orientation=orientation
#             )
            
#             fractures.append(fracture)
        
#         return fractures
    
#     def generate_3d_dfn(self, params: Dict, domain_size: Tuple[float, float, float],
#                        n_fractures: Optional[int] = None) -> List[Fracture3D]:
#         """
#         Gera DFN 3D com fraturas como discos
        
#         Args:
#             params: Parâmetros da distribuição
#             domain_size: (largura, altura, profundidade) do domínio
#             n_fractures: Número de fraturas
        
#         Returns:
#             Lista de fraturas 3D
#         """
#         width, height, depth = domain_size
#         volume = width * height * depth
        
#         # Determinar número de fraturas
#         if n_fractures is None:
#             if 'p32' in params:
#                 # Estimar baseado em P32
#                 mean_area = params.get('mean_area', 1.0)
#                 n_fractures = int(params['p32'] * volume / mean_area)
#             else:
#                 n_fractures = 100
        
#         fractures = []
        
#         for _ in range(n_fractures):
#             # Gerar raio da distribuição
#             radius = self._sample_power_law(
#                 params['exponent'],
#                 params['x_min'],
#                 params.get('x_max', min(domain_size)/4)
#             ) / 2
            
#             # Gerar abertura
#             if 'g' in params and 'm' in params:
#                 aperture = params['g'] * (2*radius) ** params['m']
#             else:
#                 aperture = radius * 0.002
            
#             # Gerar orientação (dip e dip direction)
#             if 'dip_mean' in params:
#                 dip = np.random.normal(
#                     params['dip_mean'],
#                     params.get('dip_std', 10)
#                 )
#                 dip = np.clip(dip, 0, 90)
#             else:
#                 dip = np.random.uniform(0, 90)
            
#             if 'dip_dir_mean' in params:
#                 dip_direction = np.random.normal(
#                     params['dip_dir_mean'],
#                     params.get('dip_dir_std', 20)
#                 )
#             else:
#                 dip_direction = np.random.uniform(0, 360)
            
#             # Calcular vetor normal
#             dip_rad = np.radians(dip)
#             dd_rad = np.radians(dip_direction)
            
#             normal = np.array([
#                 np.sin(dip_rad) * np.sin(dd_rad),
#                 np.sin(dip_rad) * np.cos(dd_rad),
#                 np.cos(dip_rad)
#             ])
            
#             # Gerar centro
#             center = np.array([
#                 np.random.uniform(0, width),
#                 np.random.uniform(0, height),
#                 np.random.uniform(0, depth)
#             ])
            
#             fracture = Fracture3D(
#                 center=center,
#                 normal=normal,
#                 radius=radius,
#                 aperture=aperture,
#                 dip=dip,
#                 dip_direction=dip_direction
#             )
            
#             fractures.append(fracture)
        
#         return fractures
    
#     def _sample_power_law(self, exponent: float, x_min: float, 
#                          x_max: float) -> float:
#         """
#         Amostra de uma distribuição power-law truncada
        
#         Args:
#             exponent: Expoente da lei de potência
#             x_min: Valor mínimo
#             x_max: Valor máximo
        
#         Returns:
#             Valor amostrado
#         """
#         if exponent == 1:
#             # Caso especial: distribuição logarítmica
#             return x_min * np.exp(np.random.uniform(0, np.log(x_max/x_min)))
#         else:
#             # Método de transformação inversa
#             u = np.random.uniform(0, 1)
#             if exponent > 1:
#                 a = 1 - exponent
#                 term1 = x_min**a
#                 term2 = x_max**a
#                 x = (term1 + u * (term2 - term1))**(1/a)
#             else:
#                 # Para expoentes < 1, usar rejeição
#                 while True:
#                     x = np.random.uniform(x_min, x_max)
#                     y = np.random.uniform(0, x_min**(-exponent))
#                     if y <= x**(-exponent):
#                         break
#             return x
    
#     def calculate_connectivity(self, fractures: List, threshold: float = 0.1) -> Dict:
#         """
#         Calcula métricas de conectividade da rede
        
#         Args:
#             fractures: Lista de fraturas
#             threshold: Distância máxima para considerar conexão
        
#         Returns:
#             Métricas de conectividade
#         """
#         n = len(fractures)
#         if n < 2:
#             return {'connectivity': 0, 'clusters': 1}
        
#         # Matriz de adjacência
#         connections = np.zeros((n, n))
        
#         if isinstance(fractures[0], Fracture2D):
#             # Conectividade 2D
#             for i in range(n):
#                 for j in range(i+1, n):
#                     if self._check_intersection_2d(fractures[i], fractures[j], threshold):
#                         connections[i, j] = 1
#                         connections[j, i] = 1
#         else:
#             # Conectividade 3D
#             for i in range(n):
#                 for j in range(i+1, n):
#                     if self._check_intersection_3d(fractures[i], fractures[j], threshold):
#                         connections[i, j] = 1
#                         connections[j, i] = 1
        
#         # Calcular métricas
#         n_connections = np.sum(connections) / 2
#         connectivity = n_connections / (n * (n-1) / 2) if n > 1 else 0
        
#         # Contar clusters (componentes conectados)
#         visited = np.zeros(n, dtype=bool)
#         n_clusters = 0
        
#         for i in range(n):
#             if not visited[i]:
#                 n_clusters += 1
#                 self._dfs(i, connections, visited)
        
#         return {
#             'connectivity': connectivity,
#             'n_connections': int(n_connections),
#             'n_clusters': n_clusters,
#             'percolation': n_clusters == 1
#         }
    
#     def _check_intersection_2d(self, f1: Fracture2D, f2: Fracture2D, 
#                               threshold: float) -> bool:
#         """Verifica interseção entre duas fraturas 2D"""
#         # Algoritmo de interseção de segmentos de linha
#         def ccw(A, B, C):
#             return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
#         A = np.array([f1.x1, f1.y1])
#         B = np.array([f1.x2, f1.y2])
#         C = np.array([f2.x1, f2.y1])
#         D = np.array([f2.x2, f2.y2])
        
#         return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    
#     def _check_intersection_3d(self, f1: Fracture3D, f2: Fracture3D,
#                               threshold: float) -> bool:
#         """Verifica interseção entre duas fraturas 3D (discos)"""
#         # Distância entre centros
#         dist = np.linalg.norm(f1.center - f2.center)
        
#         # Verificar se estão próximos o suficiente
#         if dist > f1.radius + f2.radius + threshold:
#             return False
        
#         # Verificar ângulo entre planos
#         cos_angle = np.abs(np.dot(f1.normal, f2.normal))
        
#         # Se planos são quase paralelos, não há interseção linear
#         if cos_angle > 0.99:
#             return False
        
#         return True
    
#     def _dfs(self, node: int, connections: np.ndarray, visited: np.ndarray):
#         """Busca em profundidade para encontrar componentes conectados"""
#         visited[node] = True
#         for neighbor in range(len(connections)):
#             if connections[node, neighbor] and not visited[neighbor]:
#                 self._dfs(neighbor, connections, visited)