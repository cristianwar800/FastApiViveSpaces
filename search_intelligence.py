# VIVESPACES-AI/search_intelligence.py - VERSIÓN CORREGIDA SIN COLUMNA STATUS

import pandas as pd
import numpy as np
import mysql.connector
import re
import json
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

class SearchPatternAnalyzer:
    def __init__(self):
        # Soporta Railway MYSQL_URL o variables individuales para desarrollo local
        if os.getenv('MYSQL_URL'):
            # Railway proporciona MYSQL_URL completa
            import urllib.parse as urlparse
            mysql_url = os.getenv('MYSQL_URL')

            # Parsear la URL de MySQL
            url = urlparse.urlparse(mysql_url)

            self.db_config = {
                'host': url.hostname,
                'user': url.username,
                'password': url.password,
                'database': url.path[1:] if url.path else 'vivespaces',  # Remover el '/' inicial
                'port': url.port or 3306
            }
        else:
            # Configuración con variables individuales (desarrollo local)
            self.db_config = {
                'host': os.getenv('DB_HOST', 'localhost'),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', ''),
                'database': os.getenv('DB_DATABASE', 'vivespaces'),
                'port': int(os.getenv('DB_PORT', 3306))
            }
        
        # TRES MODELOS ML - SISTEMA HÍBRIDO
        self.vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        
        # 1. Naive Bayes - Clasificación rápida de intención
        self.naive_bayes_model = MultinomialNB(alpha=1.0)
        
        # 2. KNN - Recomendaciones basadas en similitud
        self.knn_model = KNeighborsClassifier(n_neighbors=5, weights='distance')
        
        # 3. MLP - Predicciones complejas
        self.mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        
        self.label_encoder = LabelEncoder()
        
        # Estados de entrenamiento
        self.models_trained = False
        self.training_data = []
        
        # Diccionarios de clasificación
        self.property_keywords = {
            'casa': ['casa', 'casas', 'residencia', 'hogar', 'vivienda', 'chalet'],
            'departamento': ['departamento', 'depto', 'apartamento', 'piso', 'flat'],
            'terreno': ['terreno', 'lote', 'predio', 'solar', 'parcela'],
            'local': ['local', 'comercial', 'negocio', 'tienda', 'plaza'],
            'oficina': ['oficina', 'corporativo', 'workspace', 'coworking']
        }
        
        self.action_keywords = {
            'compra': ['compra', 'comprar', 'adquirir', 'venta', 'vender'],
            'renta': ['renta', 'rentar', 'alquiler', 'alquilar', 'arrendar'],
            'inversion': ['inversion', 'invertir', 'negocio', 'ganancia', 'roi']
        }
        
        self.feature_keywords = {
            'piscina': ['piscina', 'alberca', 'pool'],
            'jardin': ['jardin', 'verde', 'patio', 'garden', 'areas verdes'],
            'garage': ['garage', 'cochera', 'estacionamiento', 'parking'],
            'seguridad': ['seguridad', 'vigilancia', 'caseta', 'privada', 'fraccionamiento'],
            'amueblado': ['amueblado', 'muebles', 'equipado', 'furnished']
        }
        
        self.location_keywords = {
            'guadalajara': ['guadalajara', 'gdl', 'centro', 'downtown'],
            'zapopan': ['zapopan', 'andares', 'puerta de hierro'],
            'tlaquepaque': ['tlaquepaque', 'san pedro tlaquepaque'],
            'tonala': ['tonala', 'san pedro'],
            'tlajomulco': ['tlajomulco', 'haciendas', 'valle real']
        }
        
        # Sinónimos para búsqueda flexible
        self.synonyms = {
            'casa': ['casa', 'vivienda', 'hogar', 'residencia'],
            'departamento': ['departamento', 'depto', 'apartamento', 'piso'],
            'renta': ['renta', 'alquiler', 'arrendamiento'],
            'venta': ['venta', 'compra', 'adquisición'],
            'amplio': ['amplio', 'grande', 'espacioso'],
            'pequeño': ['pequeño', 'chico', 'compacto']
        }
        
        # Ciudades cercanas para búsqueda expandida
        self.nearby_cities = {
            'zapopan': ['guadalajara', 'tlaquepaque', 'tonala'],
            'guadalajara': ['zapopan', 'tlaquepaque', 'tonala'],
            'tlaquepaque': ['guadalajara', 'zapopan', 'tonala'],
            'tonala': ['guadalajara', 'zapopan', 'tlaquepaque'],
            'tlajomulco': ['guadalajara', 'zapopan']
        }
        
        # Inicializar con datos de ejemplo
        self.initialize_with_sample_data()
        
    def initialize_with_sample_data(self):
        """Inicializar modelos con datos de ejemplo"""
        sample_queries = [
            'casa venta zapopan',
            'departamento renta centro',
            'casa 3 recamaras guadalajara',
            'oficina renta andares',
            'terreno venta tlaquepaque',
            'departamento amueblado providencia',
            'casa jardin piscina',
            'local comercial plaza del sol',
            'casa inversion zapopan',
            'departamento 2 recamaras renta',
            'casa residencial puerta hierro',
            'terreno comercial guadalajara',
            'oficina corporativa andares',
            'casa venta tlajomulco',
            'departamento lujo zapopan',
            'casa garage jardin',
            'local comercial centro',
            'departamento estudiante barato',
            'casa 4 recamaras venta',
            'terreno habitacional tonala'
        ]
        
        sample_labels = [
            'compra_casa', 'renta_departamento', 'compra_casa',
            'renta_oficina', 'compra_terreno', 'renta_departamento',
            'compra_casa', 'renta_local', 'inversion_casa',
            'renta_departamento', 'compra_casa', 'compra_terreno',
            'renta_oficina', 'compra_casa', 'compra_departamento',
            'compra_casa', 'renta_local', 'renta_departamento',
            'compra_casa', 'compra_terreno'
        ]
        
        self.train_models(sample_queries, sample_labels)
    
    def train_models(self, queries, labels):
        """Entrenar los 3 modelos con datos"""
        try:
            X = self.vectorizer.fit_transform(queries)
            y = self.label_encoder.fit_transform(labels)
            
            self.naive_bayes_model.fit(X, y)
            print("✅ Naive Bayes entrenado")
            
            self.knn_model.fit(X, y)
            print("✅ KNN entrenado")
            
            self.mlp_model.fit(X, y)
            print("✅ MLP entrenado")
            
            self.models_trained = True
            self.training_data = list(zip(queries, labels))
            
            return True
        except Exception as e:
            print(f"❌ Error entrenando modelos: {e}")
            return False
    
    def classify_with_naive_bayes(self, query):
        """Clasificación RÁPIDA con Naive Bayes"""
        if not self.models_trained:
            return self.extract_search_intent(query)
        
        try:
            X = self.vectorizer.transform([query])
            
            prediction = self.naive_bayes_model.predict(X)[0]
            probabilities = self.naive_bayes_model.predict_proba(X)[0]
            
            category = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            all_probs = {
                self.label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            
            return {
                'algorithm': 'Naive Bayes',
                'category': category,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'speed': 'ultra_fast',
                'use_case': 'clasificacion_instantanea'
            }
        except Exception as e:
            print(f"Error en Naive Bayes: {e}")
            return self.extract_search_intent(query)
    
    def find_similar_users(self, user_query, n_similar=5):
        """Encontrar búsquedas similares con KNN"""
        if not self.models_trained or not self.training_data:
            return []
        
        try:
            X_user = self.vectorizer.transform([user_query])
            
            distances, indices = self.knn_model.kneighbors(X_user, n_neighbors=min(n_similar, len(self.training_data)))
            
            similar_searches = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx < len(self.training_data):
                    query, label = self.training_data[idx]
                    similar_searches.append({
                        'query': query,
                        'category': label,
                        'similarity': 1 - (dist / (distances[0].max() + 0.0001)),
                        'distance': float(dist)
                    })
            
            return {
                'algorithm': 'KNN',
                'user_query': user_query,
                'similar_searches': similar_searches,
                'use_case': 'usuarios_similares_buscaron'
            }
        except Exception as e:
            print(f"Error en KNN: {e}")
            return []
    
    def predict_with_mlp(self, query):
        """Predicción compleja con MLP"""
        if not self.models_trained:
            return self.extract_search_intent(query)
        
        try:
            X = self.vectorizer.transform([query])
            
            prediction = self.mlp_model.predict(X)[0]
            probabilities = self.mlp_model.predict_proba(X)[0]
            
            category = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            all_probs = {
                self.label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            }
            
            return {
                'algorithm': 'MLP (Neural Network)',
                'category': category,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'speed': 'medium',
                'use_case': 'predicciones_complejas'
            }
        except Exception as e:
            print(f"Error en MLP: {e}")
            return self.extract_search_intent(query)
    
    def ensemble_prediction(self, query):
        """Combinar los 3 algoritmos"""
        if not self.models_trained:
            return self.extract_search_intent(query)
        
        try:
            X = self.vectorizer.transform([query])
            
            nb_probs = self.naive_bayes_model.predict_proba(X)[0]
            knn_probs = self.knn_model.predict_proba(X)[0]
            mlp_probs = self.mlp_model.predict_proba(X)[0]
            
            ensemble_probs = (nb_probs * 0.25) + (knn_probs * 0.25) + (mlp_probs * 0.5)
            
            prediction = np.argmax(ensemble_probs)
            category = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(ensemble_probs[prediction])
            
            nb_pred = self.label_encoder.inverse_transform([np.argmax(nb_probs)])[0]
            knn_pred = self.label_encoder.inverse_transform([np.argmax(knn_probs)])[0]
            mlp_pred = self.label_encoder.inverse_transform([np.argmax(mlp_probs)])[0]
            
            return {
                'algorithm': 'Ensemble (NB + KNN + MLP)',
                'category': category,
                'confidence': confidence,
                'individual_predictions': {
                    'naive_bayes': nb_pred,
                    'knn': knn_pred,
                    'mlp': mlp_pred
                },
                'agreement': len(set([nb_pred, knn_pred, mlp_pred])) == 1,
                'use_case': 'maxima_precision'
            }
        except Exception as e:
            print(f"Error en ensemble: {e}")
            return self.extract_search_intent(query)
    
    # 🔥 RECOMENDACIÓN DE PROPIEDADES
    def get_property_recommendations(self, query, filters={}, limit=5):
        """
        Obtener propiedades recomendadas desde MySQL
        Sistema de 3 niveles: Exacto → Cercano → Genérico
        """
        try:
            search_info = self.extract_search_intent(query)
            
            # Nivel 1: Búsqueda EXACTA
            properties = self._search_exact(query, filters, limit)
            
            if len(properties) >= 3:
                return {
                    'level': 1,
                    'strategy': 'exact_match',
                    'properties': properties,
                    'total': len(properties),
                    'message': f'Encontramos {len(properties)} propiedades que coinciden exactamente'
                }
            
            # Nivel 2: Búsqueda CERCANA
            print(f"🔍 Nivel 1 insuficiente ({len(properties)}), expandiendo búsqueda...")
            properties = self._search_nearby(query, filters, search_info, limit)
            
            if len(properties) >= 3:
                return {
                    'level': 2,
                    'strategy': 'nearby_expanded',
                    'properties': properties,
                    'total': len(properties),
                    'message': f'Encontramos {len(properties)} propiedades similares cerca de tu búsqueda'
                }
            
            # Nivel 3: Búsqueda GENÉRICA
            print(f"🔍 Nivel 2 insuficiente ({len(properties)}), mostrando propiedades populares...")
            properties = self._search_generic(filters, limit)
            
            return {
                'level': 3,
                'strategy': 'generic_popular',
                'properties': properties,
                'total': len(properties),
                'message': f'Aquí hay {len(properties)} propiedades que podrían interesarte'
            }
            
        except Exception as e:
            print(f"❌ Error en get_property_recommendations: {e}")
            return {
                'level': 0,
                'strategy': 'error',
                'properties': [],
                'total': 0,
                'message': 'Error al buscar propiedades',
                'error': str(e)
            }
    
    def _search_exact(self, query, filters, limit):
        """Nivel 1: Búsqueda EXACTA por keywords - SIN COLUMNA STATUS"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor(dictionary=True)
        
        keywords = self._extract_keywords(query)
        
        if not keywords:
            cursor.close()
            conn.close()
            return []
        
        where_conditions = []
        params = []
        
        # Buscar en título, descripción y ciudad
        for keyword in keywords:
            where_conditions.append("(title LIKE %s OR description LIKE %s OR city LIKE %s)")
            params.extend([f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'])
        
        # Aplicar filtros adicionales
        if filters.get('property_type') and filters['property_type'] != 'all':
            where_conditions.append("type = %s")
            params.append(filters['property_type'])
        
        if filters.get('min_price'):
            where_conditions.append("price >= %s")
            params.append(filters['min_price'])
        
        if filters.get('max_price'):
            where_conditions.append("price <= %s")
            params.append(filters['max_price'])
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # 🔥 QUERY SIN COLUMNA STATUS
        query_sql = f"""
            SELECT 
                id, title, description, type, price, city, 
                bedrooms, bathrooms, area, image, 
                latitude, longitude,
                created_at
            FROM properties 
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
        """
        
        params.append(limit)
        
        try:
            cursor.execute(query_sql, params)
            properties = cursor.fetchall()
            return properties
        except Exception as e:
            print(f"❌ Error en _search_exact: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def _search_nearby(self, query, filters, search_info, limit):
        """Nivel 2: Búsqueda CERCANA - SIN COLUMNA STATUS"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor(dictionary=True)
        
        expanded_keywords = self._expand_keywords(query)
        
        location = search_info.get('locations', [])
        nearby_locations = []
        
        for loc in location:
            if loc.lower() in self.nearby_cities:
                nearby_locations.extend(self.nearby_cities[loc.lower()])
        
        where_conditions = []
        params = []
        
        # Buscar con keywords expandidas
        if expanded_keywords:
            keyword_conditions = []
            for keyword in expanded_keywords[:5]:
                keyword_conditions.append("(title LIKE %s OR description LIKE %s OR city LIKE %s)")
                params.extend([f'%{keyword}%', f'%{keyword}%', f'%{keyword}%'])
            
            if keyword_conditions:
                where_conditions.append("(" + " OR ".join(keyword_conditions) + ")")
        
        # Buscar en ubicaciones cercanas
        if nearby_locations:
            location_conditions = " OR ".join(["city LIKE %s" for _ in nearby_locations])
            where_conditions.append(f"({location_conditions})")
            params.extend([f'%{loc}%' for loc in nearby_locations])
        
        # Tipo de propiedad más flexible
        property_type = search_info.get('property_type')
        if property_type and property_type != 'unknown':
            related_types = ['casa', 'departamento'] if property_type in ['casa', 'departamento'] else [property_type]
            type_conditions = " OR ".join(["type = %s" for _ in related_types])
            where_conditions.append(f"({type_conditions})")
            params.extend(related_types)
        
        # Rango de precio expandido (±30%)
        if filters.get('min_price'):
            expanded_min = float(filters['min_price']) * 0.7
            where_conditions.append("price >= %s")
            params.append(expanded_min)
        
        if filters.get('max_price'):
            expanded_max = float(filters['max_price']) * 1.3
            where_conditions.append("price <= %s")
            params.append(expanded_max)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # 🔥 QUERY SIN COLUMNA STATUS
        query_sql = f"""
            SELECT 
                id, title, description, type, price, city, 
                bedrooms, bathrooms, area, image, 
                latitude, longitude,
                created_at
            FROM properties 
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
        """
        
        params.append(limit * 2)
        
        try:
            cursor.execute(query_sql, params)
            properties = cursor.fetchall()
            return properties[:limit]
        except Exception as e:
            print(f"❌ Error en _search_nearby: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def _search_generic(self, filters, limit):
        """Nivel 3: Búsqueda GENÉRICA - SIN COLUMNA STATUS"""
        conn = self.get_db_connection()
        if not conn:
            return []
        
        cursor = conn.cursor(dictionary=True)
        
        # 🔥 QUERY SIN COLUMNA STATUS - Mostrar últimas propiedades
        query_sql = """
            SELECT 
                id, title, description, type, price, city, 
                bedrooms, bathrooms, area, image, 
                latitude, longitude,
                created_at
            FROM properties 
            ORDER BY created_at DESC
            LIMIT %s
        """
        
        try:
            cursor.execute(query_sql, (limit,))
            properties = cursor.fetchall()
            return properties
        except Exception as e:
            print(f"❌ Error en _search_generic: {e}")
            return []
        finally:
            cursor.close()
            conn.close()
    
    def _extract_keywords(self, query):
        """Extraer keywords relevantes"""
        stop_words = ['el', 'la', 'los', 'las', 'un', 'una', 'de', 'en', 'con', 'para', 'por']
        
        words = re.findall(r'\w+', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _expand_keywords(self, query):
        """Expandir keywords con sinónimos"""
        keywords = self._extract_keywords(query)
        expanded = set(keywords)
        
        for keyword in keywords:
            if keyword in self.synonyms:
                expanded.update(self.synonyms[keyword])
        
        return list(expanded)
    
    def get_db_connection(self):
        """Conexión a base de datos"""
        try:
            return mysql.connector.connect(**self.db_config)
        except mysql.connector.Error as err:
            print(f"❌ Error conectando a BD: {err}")
            return None
    
    def create_search_events_table(self):
        """Crear tabla para eventos de búsqueda"""
        conn = self.get_db_connection()
        if not conn:
            return False
            
        cursor = conn.cursor()
        
        create_table_query = """
        CREATE TABLE IF NOT EXISTS search_events (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            user_id INT NULL,
            session_id VARCHAR(100),
            search_query TEXT,
            search_type VARCHAR(50) DEFAULT 'general',
            filters JSON,
            results_count INT DEFAULT 0,
            click_position INT NULL,
            time_spent INT DEFAULT 0,
            device VARCHAR(20) DEFAULT 'unknown',
            page_url VARCHAR(255),
            referrer VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            
            INDEX idx_user_id (user_id),
            INDEX idx_session_id (session_id),
            INDEX idx_created_at (created_at),
            INDEX idx_search_type (search_type),
            INDEX idx_search_query (search_query(100))
        )
        """
        
        try:
            cursor.execute(create_table_query)
            conn.commit()
            print("✅ Tabla search_events creada/verificada")
            return True
        except mysql.connector.Error as err:
            print(f"❌ Error creando tabla: {err}")
            return False
        finally:
            cursor.close()
            conn.close()
    
    def extract_search_intent(self, query):
        """Método original de extracción de intención (fallback)"""
        if not query or len(query.strip()) < 2:
            return {'intent': 'unknown', 'confidence': 0.0}
        
        query_lower = query.lower().strip()
        
        # Detectar tipo de propiedad
        property_type = 'unknown'
        property_score = 0
        for prop_type, keywords in self.property_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > property_score:
                property_score = matches
                property_type = prop_type
        
        # Detectar acción
        action = 'unknown'
        action_score = 0
        for act_type, keywords in self.action_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in query_lower)
            if matches > action_score:
                action_score = matches
                action = act_type
        
        # Detectar características
        features = []
        for feature, keywords in self.feature_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                features.append(feature)
        
        # Detectar ubicaciones
        locations = []
        for location, keywords in self.location_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                locations.append(location)
        
        # Calcular confianza
        total_matches = property_score + action_score + len(features) + len(locations)
        confidence = min(total_matches / 5.0, 1.0)
        
        return {
            'intent': f"{action}_{property_type}" if action != 'unknown' and property_type != 'unknown' else 'browse',
            'property_type': property_type,
            'action': action,
            'features': features,
            'locations': locations,
            'confidence': confidence,
            'algorithm': 'rule_based'
        }
    
    def save_search_event(self, search_data):
        """Guardar evento de búsqueda en BD"""
        conn = self.get_db_connection()
        if not conn:
            return False
        
        cursor = conn.cursor()
        
        try:
            insert_query = """
            INSERT INTO search_events 
            (user_id, session_id, search_query, search_type, filters, results_count, 
             time_spent, device, page_url, referrer)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            values = (
                search_data.get('user_id'),
                search_data.get('session_id'),
                search_data.get('search_query'),
                search_data.get('search_type', 'general'),
                json.dumps(search_data.get('filters', {})),
                search_data.get('results_count', 0),
                search_data.get('time_spent', 0),
                search_data.get('device', 'unknown'),
                search_data.get('page_url'),
                search_data.get('referrer')
            )
            
            cursor.execute(insert_query, values)
            conn.commit()
            
            return True
            
        except Exception as e:
            print(f"❌ Error guardando evento: {e}")
            return False
        
        finally:
            cursor.close()
            conn.close()

# FUNCIONES PÚBLICAS PARA USAR EN FASTAPI

def classify_quick(query):
    """Clasificación rápida con Naive Bayes"""
    analyzer = SearchPatternAnalyzer()
    return analyzer.classify_with_naive_bayes(query)

def find_similar(query, n=5):
    """Encontrar búsquedas similares con KNN"""
    analyzer = SearchPatternAnalyzer()
    return analyzer.find_similar_users(query, n)

def predict_complex(query):
    """Predicción compleja con MLP"""
    analyzer = SearchPatternAnalyzer()
    return analyzer.predict_with_mlp(query)

def predict_ensemble(query):
    """Predicción combinando los 3 algoritmos"""
    analyzer = SearchPatternAnalyzer()
    return analyzer.ensemble_prediction(query)

def save_search_event(search_data):
    """Guardar evento de búsqueda"""
    analyzer = SearchPatternAnalyzer()
    return analyzer.save_search_event(search_data)

def initialize_search_table():
    """Crear tabla de búsquedas"""
    analyzer = SearchPatternAnalyzer()
    return analyzer.create_search_events_table()

def recommend_properties(query, filters={}, limit=5):
    """Obtener propiedades recomendadas (3 niveles)"""
    analyzer = SearchPatternAnalyzer()
    return analyzer.get_property_recommendations(query, filters, limit)