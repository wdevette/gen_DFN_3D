import pandas as pd
import numpy as np
import json
from io import BytesIO
from typing import Dict, Any, List, Optional
import xlsxwriter
from datetime import datetime

class ResultsExporter:
    """Exportador de resultados para diferentes formatos"""
    
    def export_to_csv(self, data: pd.DataFrame) -> str:
        """Exporta DataFrame para CSV"""
        return data.to_csv(index=False)
    
    def export_parameters(self, params: Dict) -> str:
        """Exporta parâmetros para JSON"""
        # Converter arrays numpy para listas
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        clean_params = convert_numpy(params)
        return json.dumps(clean_params, indent=2)
    
    def export_dfn_2d_geojson(self, fractures: List) -> str:
        """Exporta DFN 2D para GeoJSON"""
        features = []
        
        for i, frac in enumerate(fractures):
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [frac.x1, frac.y1],
                        [frac.x2, frac.y2]
                    ]
                },
                "properties": {
                    "id": i + 1,
                    "length": float(frac.length) if not np.isnan(frac.length) else 0,
                    "aperture": float(frac.aperture) if not np.isnan(frac.aperture) else 0,
                    "orientation": float(frac.orientation) if not np.isnan(frac.orientation) else 0
                }
            }
            features.append(feature)
        
        geojson = {
            "type": "FeatureCollection",
            "features": features,
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                }
            }
        }
        
        return json.dumps(geojson, indent=2)
    
    def export_dfn_3d_vtk(self, fractures: List) -> str:
        """Exporta DFN 3D para formato VTK"""
        vtk_content = []
        vtk_content.append("# vtk DataFile Version 3.0")
        vtk_content.append("DFN 3D Fracture Network")
        vtk_content.append("ASCII")
        vtk_content.append("DATASET POLYDATA")
        
        # Coletar todos os pontos
        all_points = []
        all_polygons = []
        point_index = 0
        
        for frac in fractures:
            # Gerar pontos do disco
            n_points = 20
            theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            
            # Calcular pontos no plano do disco
            if abs(frac.normal[2]) < 0.99:
                v1 = np.cross(frac.normal, [0, 0, 1])
            else:
                v1 = np.cross(frac.normal, [1, 0, 0])
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.cross(frac.normal, v1)
            
            # Adicionar centro
            all_points.append(frac.center)
            center_idx = point_index
            point_index += 1
            
            # Adicionar pontos do perímetro
            polygon_indices = [center_idx]
            for t in theta:
                point = frac.center + frac.radius * (np.cos(t) * v1 + np.sin(t) * v2)
                all_points.append(point)
                polygon_indices.append(point_index)
                point_index += 1
            
            all_polygons.append(polygon_indices)
        
        # Escrever pontos
        vtk_content.append(f"POINTS {len(all_points)} float")
        for point in all_points:
            vtk_content.append(f"{point[0]} {point[1]} {point[2]}")
        
        # Escrever polígonos
        n_cells = len(all_polygons)
        n_ints = sum(len(p) + 1 for p in all_polygons)
        vtk_content.append(f"POLYGONS {n_cells} {n_ints}")
        
        for polygon in all_polygons:
            line = f"{len(polygon)}"
            for idx in polygon:
                line += f" {idx}"
            vtk_content.append(line)
        
        return "\n".join(vtk_content)
    
    def generate_full_report(self, framfrat_data: Optional[pd.DataFrame],
                           scanline_data: Optional[pd.DataFrame],
                           analysis_results: Dict,
                           metadata: Dict) -> bytes:
        """
        Gera relatório Excel completo
        
        Args:
            framfrat_data: Dados FRAMFRAT
            scanline_data: Dados scanline
            analysis_results: Resultados das análises
            metadata: Metadados (área, comprimento, etc)
        
        Returns:
            Bytes do arquivo Excel
        """
        output = BytesIO()
        
        # CORREÇÃO: Adicionar opção para lidar com NaN/Inf
        with xlsxwriter.Workbook(output, {'in_memory': True, 'nan_inf_to_errors': True}) as workbook:
            # Formatos
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#D7E4BD',
                'border': 1
            })
            
            title_format = workbook.add_format({
                'bold': True,
                'font_size': 14,
                'bg_color': '#366092',
                'font_color': 'white'
            })
            
            number_format = workbook.add_format({'num_format': '0.000'})
            
            # 1. Aba de resumo
            summary_sheet = workbook.add_worksheet('Resumo')
            summary_sheet.merge_range('A1:E1', 'Análise de Fraturas - Relatório', title_format)
            
            row = 2
            summary_sheet.write(row, 0, 'Data:', header_format)
            summary_sheet.write(row, 1, datetime.now().strftime('%Y-%m-%d %H:%M'))
            
            row += 2
            summary_sheet.write(row, 0, 'Metadados', header_format)
            row += 1
            for key, value in metadata.items():
                summary_sheet.write(row, 0, key)
                # Tratar valores antes de escrever
                if isinstance(value, (int, float)):
                    if np.isnan(value) or np.isinf(value):
                        summary_sheet.write(row, 1, 'N/A')
                    else:
                        summary_sheet.write(row, 1, value, number_format)
                else:
                    summary_sheet.write(row, 1, str(value))
                row += 1
            
            # 2. Aba de dados FRAMFRAT
            if framfrat_data is not None:
                data_sheet = workbook.add_worksheet('Dados FRAMFRAT')
                
                # Limpar dados NaN/Inf
                framfrat_clean = framfrat_data.copy()
                framfrat_clean = framfrat_clean.replace([np.inf, -np.inf], np.nan)
                framfrat_clean = framfrat_clean.fillna(0)  # ou outro valor padrão
                
                # Cabeçalhos
                for col, header in enumerate(framfrat_clean.columns):
                    data_sheet.write(0, col, header, header_format)
                
                # Dados
                for idx, row_data in framfrat_clean.iterrows():
                    for col, value in enumerate(row_data):
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            try:
                                if np.isnan(value) or np.isinf(value):
                                    data_sheet.write(idx + 1, col, 0)
                                else:
                                    data_sheet.write_number(idx + 1, col, float(value), number_format)
                            except:
                                data_sheet.write(idx + 1, col, str(value))
                        else:
                            data_sheet.write(idx + 1, col, str(value))
            
            # 3. Aba de dados Scanline
            if scanline_data is not None:
                scan_sheet = workbook.add_worksheet('Dados Scanline')
                
                # Limpar dados
                scanline_clean = scanline_data.copy()
                scanline_clean = scanline_clean.replace([np.inf, -np.inf], np.nan)
                scanline_clean = scanline_clean.fillna(0)
                
                for col, header in enumerate(scanline_clean.columns):
                    scan_sheet.write(0, col, header, header_format)
                
                for idx, row_data in scanline_clean.iterrows():
                    for col, value in enumerate(row_data):
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            try:
                                if np.isnan(value) or np.isinf(value):
                                    scan_sheet.write(idx + 1, col, 0)
                                else:
                                    scan_sheet.write_number(idx + 1, col, float(value), number_format)
                            except:
                                scan_sheet.write(idx + 1, col, str(value))
                        else:
                            scan_sheet.write(idx + 1, col, str(value))
            
            # 4. Aba de resultados
            if analysis_results:
                results_sheet = workbook.add_worksheet('Resultados')
                
                row = 0
                for analysis_name, results in analysis_results.items():
                    results_sheet.merge_range(row, 0, row, 3, analysis_name, title_format)
                    row += 1
                    
                    if isinstance(results, dict):
                        for key, value in results.items():
                            results_sheet.write(row, 0, key, header_format)
                            
                            # Tratar diferentes tipos de valores
                            if isinstance(value, (list, np.ndarray)):
                                # Converter para string representação
                                if isinstance(value, np.ndarray):
                                    value_clean = np.where(np.isfinite(value), value, 0)
                                    results_sheet.write(row, 1, str(value_clean.tolist()[:10]) + '...' if len(value_clean) > 10 else str(value_clean.tolist()))
                                else:
                                    results_sheet.write(row, 1, str(value[:10]) + '...' if len(value) > 10 else str(value))
                            elif isinstance(value, (int, float, np.integer, np.floating)):
                                if np.isnan(value) or np.isinf(value):
                                    results_sheet.write(row, 1, 'N/A')
                                else:
                                    results_sheet.write(row, 1, float(value), number_format)
                            else:
                                results_sheet.write(row, 1, str(value))
                            row += 1
                    row += 1
            
            # 5. Adicionar estatísticas resumidas
            if framfrat_data is not None and 'length' in framfrat_data.columns:
                stats_sheet = workbook.add_worksheet('Estatísticas')
                
                # Calcular estatísticas básicas
                stats = {
                    'Comprimento - Média (m)': framfrat_data['length'].mean(),
                    'Comprimento - Mediana (m)': framfrat_data['length'].median(),
                    'Comprimento - Desvio Padrão (m)': framfrat_data['length'].std(),
                    'Comprimento - Mínimo (m)': framfrat_data['length'].min(),
                    'Comprimento - Máximo (m)': framfrat_data['length'].max(),
                }
                
                if 'aperture' in framfrat_data.columns:
                    stats.update({
                        'Abertura - Média (mm)': framfrat_data['aperture'].mean() * 1000,
                        'Abertura - Mediana (mm)': framfrat_data['aperture'].median() * 1000,
                        'Abertura - Desvio Padrão (mm)': framfrat_data['aperture'].std() * 1000,
                        'Abertura - Mínimo (mm)': framfrat_data['aperture'].min() * 1000,
                        'Abertura - Máximo (mm)': framfrat_data['aperture'].max() * 1000,
                    })
                
                row = 0
                stats_sheet.merge_range(row, 0, row, 1, 'Estatísticas Descritivas', title_format)
                row += 2
                
                for stat_name, stat_value in stats.items():
                    stats_sheet.write(row, 0, stat_name, header_format)
                    if isinstance(stat_value, (int, float)) and np.isfinite(stat_value):
                        stats_sheet.write(row, 1, stat_value, number_format)
                    else:
                        stats_sheet.write(row, 1, 'N/A')
                    row += 1
        
        output.seek(0)
        return output.read()
    
    def save_session(self, session_state: Any) -> str:
        """Salva estado da sessão"""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'data_loaded': session_state.data_loaded,
            'analysis_results': {}
        }
        
        # Salvar resultados de análise
        if hasattr(session_state, 'analysis_results'):
            session_data['analysis_results'] = self._serialize_results(
                session_state.analysis_results
            )
        
        # Salvar configurações
        config_keys = ['image_area', 'scanline_length', 'l_min', 'b_min']
        for key in config_keys:
            if hasattr(session_state, key):
                value = getattr(session_state, key)
                # Tratar NaN/Inf
                if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
                    session_data[key] = None
                else:
                    session_data[key] = value
        
        return json.dumps(session_data, indent=2, default=str)
    
    def load_session(self, file, session_state: Any):
        """Carrega estado da sessão"""
        content = file.read()
        session_data = json.loads(content)
        
        # Restaurar dados
        for key, value in session_data.items():
            if key != 'timestamp':
                setattr(session_state, key, value)
    
    def _serialize_results(self, results: Any) -> Any:
        """Serializa resultados para JSON"""
        if isinstance(results, np.ndarray):
            # Limpar NaN/Inf antes de serializar
            clean = np.where(np.isfinite(results), results, 0)
            return clean.tolist()
        elif isinstance(results, pd.DataFrame):
            # Limpar DataFrame
            clean_df = results.replace([np.inf, -np.inf], np.nan).fillna(0)
            return clean_df.to_dict('records')
        elif isinstance(results, dict):
            return {k: self._serialize_results(v) for k, v in results.items()}
        elif isinstance(results, (list, tuple)):
            return [self._serialize_results(item) for item in results]
        elif isinstance(results, (np.integer, np.floating)):
            if np.isnan(results) or np.isinf(results):
                return None
            return float(results)
        elif isinstance(results, (int, float)):
            if np.isnan(results) or np.isinf(results):
                return None
            return results
        else:
            return results




# import pandas as pd
# import numpy as np
# import json
# from io import BytesIO
# from typing import Dict, Any, List, Optional
# import xlsxwriter
# from datetime import datetime

# class ResultsExporter:
#     """Exportador de resultados para diferentes formatos"""
    
#     def export_to_csv(self, data: pd.DataFrame) -> str:
#         """Exporta DataFrame para CSV"""
#         return data.to_csv(index=False)
    
#     def export_parameters(self, params: Dict) -> str:
#         """Exporta parâmetros para JSON"""
#         # Converter arrays numpy para listas
#         def convert_numpy(obj):
#             if isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             elif isinstance(obj, np.integer):
#                 return int(obj)
#             elif isinstance(obj, np.floating):
#                 return float(obj)
#             elif isinstance(obj, dict):
#                 return {k: convert_numpy(v) for k, v in obj.items()}
#             elif isinstance(obj, list):
#                 return [convert_numpy(item) for item in obj]
#             return obj
        
#         clean_params = convert_numpy(params)
#         return json.dumps(clean_params, indent=2)
    
#     def export_dfn_2d_geojson(self, fractures: List) -> str:
#         """Exporta DFN 2D para GeoJSON"""
#         features = []
        
#         for i, frac in enumerate(fractures):
#             feature = {
#                 "type": "Feature",
#                 "geometry": {
#                     "type": "LineString",
#                     "coordinates": [
#                         [frac.x1, frac.y1],
#                         [frac.x2, frac.y2]
#                     ]
#                 },
#                 "properties": {
#                     "id": i + 1,
#                     "length": frac.length,
#                     "aperture": frac.aperture,
#                     "orientation": frac.orientation
#                 }
#             }
#             features.append(feature)
        
#         geojson = {
#             "type": "FeatureCollection",
#             "features": features,
#             "crs": {
#                 "type": "name",
#                 "properties": {
#                     "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
#                 }
#             }
#         }
        
#         return json.dumps(geojson, indent=2)
    
#     def export_dfn_3d_vtk(self, fractures: List) -> str:
#         """Exporta DFN 3D para formato VTK"""
#         vtk_content = []
#         vtk_content.append("# vtk DataFile Version 3.0")
#         vtk_content.append("DFN 3D Fracture Network")
#         vtk_content.append("ASCII")
#         vtk_content.append("DATASET POLYDATA")
        
#         # Coletar todos os pontos
#         all_points = []
#         all_polygons = []
#         point_index = 0
        
#         for frac in fractures:
#             # Gerar pontos do disco
#             n_points = 20
#             theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
            
#             # Calcular pontos no plano do disco
#             if abs(frac.normal[2]) < 0.99:
#                 v1 = np.cross(frac.normal, [0, 0, 1])
#             else:
#                 v1 = np.cross(frac.normal, [1, 0, 0])
#             v1 = v1 / np.linalg.norm(v1)
#             v2 = np.cross(frac.normal, v1)
            
#             # Adicionar centro
#             all_points.append(frac.center)
#             center_idx = point_index
#             point_index += 1
            
#             # Adicionar pontos do perímetro
#             polygon_indices = [center_idx]
#             for t in theta:
#                 point = frac.center + frac.radius * (np.cos(t) * v1 + np.sin(t) * v2)
#                 all_points.append(point)
#                 polygon_indices.append(point_index)
#                 point_index += 1
            
#             all_polygons.append(polygon_indices)
        
#         # Escrever pontos
#         vtk_content.append(f"POINTS {len(all_points)} float")
#         for point in all_points:
#             vtk_content.append(f"{point[0]} {point[1]} {point[2]}")
        
#         # Escrever polígonos
#         n_cells = len(all_polygons)
#         n_ints = sum(len(p) + 1 for p in all_polygons)
#         vtk_content.append(f"POLYGONS {n_cells} {n_ints}")
        
#         for polygon in all_polygons:
#             line = f"{len(polygon)}"
#             for idx in polygon:
#                 line += f" {idx}"
#             vtk_content.append(line)
        
#         return "\n".join(vtk_content)
    
#     def generate_full_report(self, framfrat_data: Optional[pd.DataFrame],
#                            scanline_data: Optional[pd.DataFrame],
#                            analysis_results: Dict,
#                            metadata: Dict) -> bytes:
#         """
#         Gera relatório Excel completo
        
#         Args:
#             framfrat_data: Dados FRAMFRAT
#             scanline_data: Dados scanline
#             analysis_results: Resultados das análises
#             metadata: Metadados (área, comprimento, etc)
        
#         Returns:
#             Bytes do arquivo Excel
#         """
#         output = BytesIO()
        
#         with xlsxwriter.Workbook(output, {'in_memory': True}) as workbook:
#             # Formatos
#             header_format = workbook.add_format({
#                 'bold': True,
#                 'bg_color': '#D7E4BD',
#                 'border': 1
#             })
            
#             title_format = workbook.add_format({
#                 'bold': True,
#                 'font_size': 14,
#                 'bg_color': '#366092',
#                 'font_color': 'white'
#             })
            
#             # 1. Aba de resumo
#             summary_sheet = workbook.add_worksheet('Resumo')
#             summary_sheet.merge_range('A1:E1', 'Análise de Fraturas - Relatório', title_format)
            
#             row = 2
#             summary_sheet.write(row, 0, 'Data:', header_format)
#             summary_sheet.write(row, 1, datetime.now().strftime('%Y-%m-%d %H:%M'))
            
#             row += 2
#             summary_sheet.write(row, 0, 'Metadados', header_format)
#             row += 1
#             for key, value in metadata.items():
#                 summary_sheet.write(row, 0, key)
#                 summary_sheet.write(row, 1, value)
#                 row += 1
            
#             # 2. Aba de dados FRAMFRAT
#             if framfrat_data is not None:
#                 data_sheet = workbook.add_worksheet('Dados FRAMFRAT')
                
#                 # Cabeçalhos
#                 for col, header in enumerate(framfrat_data.columns):
#                     data_sheet.write(0, col, header, header_format)
                
#                 # Dados
#                 for idx, row_data in framfrat_data.iterrows():
#                     for col, value in enumerate(row_data):
#                         data_sheet.write(idx + 1, col, float(value) if isinstance(value, (int, float)) else value)
            
#             # 3. Aba de dados Scanline
#             if scanline_data is not None:
#                 scan_sheet = workbook.add_worksheet('Dados Scanline')
                
#                 for col, header in enumerate(scanline_data.columns):
#                     scan_sheet.write(0, col, header, header_format)
                
#                 for idx, row_data in scanline_data.iterrows():
#                     for col, value in enumerate(row_data):
#                         scan_sheet.write(idx + 1, col, float(value) if isinstance(value, (int, float)) else value)
            
#             # 4. Aba de resultados
#             if analysis_results:
#                 results_sheet = workbook.add_worksheet('Resultados')
                
#                 row = 0
#                 for analysis_name, results in analysis_results.items():
#                     results_sheet.merge_range(row, 0, row, 3, analysis_name, title_format)
#                     row += 1
                    
#                     if isinstance(results, dict):
#                         for key, value in results.items():
#                             results_sheet.write(row, 0, key, header_format)
#                             if isinstance(value, (list, np.ndarray)):
#                                 results_sheet.write(row, 1, str(value))
#                             else:
#                                 results_sheet.write(row, 1, value)
#                             row += 1
#                     row += 1
            
#             # 5. Adicionar gráfico se possível
#             if framfrat_data is not None and 'length' in framfrat_data.columns:
#                 chart_sheet = workbook.add_worksheet('Gráficos')
                
#                 # Criar gráfico de distribuição
#                 chart = workbook.add_chart({'type': 'scatter', 'subtype': 'straight'})
                
#                 # Configurar gráfico (exemplo simplificado)
#                 chart.set_title({'name': 'Distribuição de Comprimentos'})
#                 chart.set_x_axis({'name': 'Comprimento (m)', 'log_base': 10})
#                 chart.set_y_axis({'name': 'Frequência Cumulativa', 'log_base': 10})
                
#                 chart_sheet.insert_chart('B2', chart, {'width': 600, 'height': 400})
        
#         output.seek(0)
#         return output.read()
    
#     def save_session(self, session_state: Any) -> str:
#         """Salva estado da sessão"""
#         session_data = {
#             'timestamp': datetime.now().isoformat(),
#             'data_loaded': session_state.data_loaded,
#             'analysis_results': {}
#         }
        
#         # Salvar resultados de análise
#         if hasattr(session_state, 'analysis_results'):
#             session_data['analysis_results'] = self._serialize_results(
#                 session_state.analysis_results
#             )
        
#         # Salvar configurações
#         config_keys = ['image_area', 'scanline_length', 'l_min', 'b_min']
#         for key in config_keys:
#             if hasattr(session_state, key):
#                 session_data[key] = getattr(session_state, key)
        
#         return json.dumps(session_data, indent=2)
    
#     def load_session(self, file, session_state: Any):
#         """Carrega estado da sessão"""
#         content = file.read()
#         session_data = json.loads(content)
        
#         # Restaurar dados
#         for key, value in session_data.items():
#             if key != 'timestamp':
#                 setattr(session_state, key, value)
    
#     def _serialize_results(self, results: Any) -> Any:
#         """Serializa resultados para JSON"""
#         if isinstance(results, np.ndarray):
#             return results.tolist()
#         elif isinstance(results, pd.DataFrame):
#             return results.to_dict('records')
#         elif isinstance(results, dict):
#             return {k: self._serialize_results(v) for k, v in results.items()}
#         elif isinstance(results, (list, tuple)):
#             return [self._serialize_results(item) for item in results]
#         elif isinstance(results, (np.integer, np.floating)):
#             return float(results)
#         else:
#             return results