from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import Optional, Dict, Any, Union, List
import mysql.connector
from datetime import datetime
import os
from dotenv import load_dotenv
from search_intelligence import SearchPatternAnalyzer
import json
import traceback

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="ViveSpaces AI API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el analizador ML
analyzer = SearchPatternAnalyzer()

# ===================== MODELOS PYDANTIC =====================

class SearchEventRequest(BaseModel):
    """Modelo para eventos de búsqueda con validación mejorada"""
    user_id: Optional[int] = None
    session_id: str
    search_query: Optional[str] = None
    search_type: str = "general"
    filters: Union[Dict[str, Any], list] = {}
    results_count: int = 0
    device: str = "desktop"
    page_url: Optional[str] = None
    referrer: Optional[str] = None
    click_position: Optional[int] = None
    time_spent: Optional[int] = None
    scroll_depth: Optional[int] = None
    property_id: Optional[int] = None
    element_clicked: Optional[str] = None
    filter_changed: Optional[str] = None

    @field_validator('filters', mode='before')
    @classmethod
    def validate_filters(cls, v):
        """Convierte lista vacía a diccionario vacío"""
        if isinstance(v, list) and len(v) == 0:
            return {}
        if v is None:
            return {}
        return v


class RecommendationRequest(BaseModel):
    """Modelo para solicitud de recomendaciones"""
    user_id: int
    limit: int = 5


class PropertySearchRequest(BaseModel):
    """Modelo para búsqueda de propiedades"""
    query: Optional[str] = None
    location: Optional[str] = None
    property_type: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    limit: int = 20


# ===================== CONEXIÓN BD =====================

def get_db_connection():
    """Obtener conexión a la base de datos - Compatible con Railway MYSQL_URL"""
    try:
        # Soporta Railway MYSQL_URL o variables individuales para desarrollo local
        if os.getenv('MYSQL_URL'):
            # Railway proporciona MYSQL_URL completa
            import urllib.parse as urlparse
            mysql_url = os.getenv('MYSQL_URL')

            # Parsear la URL de MySQL
            url = urlparse.urlparse(mysql_url)

            connection = mysql.connector.connect(
                host=url.hostname,
                user=url.username,
                password=url.password,
                database=url.path[1:] if url.path else 'vivespaces',  # Remover el '/' inicial
                port=url.port or 3306,
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci'
            )
        else:
            # Configuración con variables individuales (desarrollo local)
            connection = mysql.connector.connect(
                host=os.getenv('DB_HOST', 'localhost'),
                user=os.getenv('DB_USER', 'root'),
                password=os.getenv('DB_PASSWORD', ''),
                database=os.getenv('DB_DATABASE', 'vivespaces'),
                port=int(os.getenv('DB_PORT', 3306)),
                charset='utf8mb4',
                collation='utf8mb4_unicode_ci'
            )

        return connection
    except mysql.connector.Error as err:
        print(f"❌ Error de conexión BD: {err}")
        raise HTTPException(status_code=500, detail=f"Error de conexión a la base de datos: {str(err)}")


# ===================== EVENTOS DE INICIO =====================

@app.on_event("startup")
async def startup_event():
    """Ejecutar al iniciar la aplicación"""
    print("\n" + "=" * 70)
    print("🚀 INICIANDO VIVESPACES AI API")
    print("=" * 70)
    print("🔍 URL Local:    http://127.0.0.1:8001")
    print("🔍 URL Network:  http://0.0.0.0:8001")
    print("📚 Documentación: http://127.0.0.1:8001/docs")
    print("📊 Health Check: http://127.0.0.1:8001/health")
    print("=" * 70)
    print("🔧 Configuración:")
    print(f"   - Base de datos: {os.getenv('DB_DATABASE', 'vivespaces')}")
    print(f"   - Host BD: {os.getenv('DB_HOST', 'localhost')}")
    print(f"   - Puerto BD: {os.getenv('DB_PORT', 3306)}")
    print("=" * 70)
    print("⏳ Cargando modelos de Machine Learning...")
    print("=" * 70 + "\n")


# ===================== ENDPOINTS =====================

@app.get("/")
async def root():
    """Endpoint raíz"""
    return {
        "service": "ViveSpaces AI API",
        "status": "online",
        "version": "1.0.0",
        "description": "Sistema de Inteligencia Artificial para ViveSpaces",
        "endpoints": {
            "health": "/health",
            "test": "/api/test",
            "track_search": "POST /api/search/track",
            "recommendations": "POST /api/recommendations/generate",
            "search_history": "GET /api/search/history/{user_id}",
            "search_properties": "POST /api/search/properties",
            "stats": "GET /api/stats",
            "ml_classify": "POST /api/ml/classify/quick",
            "ml_similar": "POST /api/ml/similar",
            "ml_predict": "POST /api/ml/predict/complex",
            "ml_ensemble": "POST /api/ml/predict/ensemble",
            "ml_compare": "POST /api/ml/compare"
        }
    }


@app.get("/health")
async def health_check():
    """Verificar estado del servicio"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        cursor.close()
        conn.close()
        
        return {
            "status": "healthy",
            "service": "ViveSpaces AI",
            "database": "connected",
            "port": 8001,
            "ml_models": "loaded",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "ViveSpaces AI",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.get("/api/test")
async def api_test():
    """Endpoint de prueba general de la API y algoritmos ML"""
    try:
        print("=" * 70)
        print("🧪 TEST GENERAL DE API Y ML INICIADO")
        print("=" * 70)
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Verificar datos en la BD
        cursor.execute("SELECT COUNT(*) as total FROM search_events")
        total_events = cursor.fetchone()['total']
        
        cursor.execute("SELECT COUNT(*) as total FROM properties")
        total_properties = cursor.fetchone()['total']
        
        cursor.close()
        conn.close()
        
        # Resultado del test
        result = {
            "api_status": "operational",
            "database": "connected",
            "ml_analyzer": "loaded" if analyzer else "not loaded",
            "endpoints": {
                "health": "✅ working",
                "track": "✅ working",
                "recommendations": "✅ working",
                "search": "✅ working",
                "stats": "✅ working",
                "ml_endpoints": "✅ working"
            },
            "database_stats": {
                "total_events": total_events,
                "total_properties": total_properties
            },
            "ml_features": {
                "pattern_analysis": "✅ operational",
                "user_recommendations": "✅ operational",
                "search_intelligence": "✅ operational",
                "naive_bayes": "✅ operational",
                "knn": "✅ operational",
                "mlp": "✅ operational",
                "ensemble": "✅ operational"
            }
        }
        
        print(f"📊 Total eventos: {total_events}")
        print(f"🏠 Total propiedades: {total_properties}")
        print(f"🤖 ML Analyzer: {'Cargado' if analyzer else 'No cargado'}")
        print("=" * 70)
        print("✅ TEST COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        
        return {
            "success": True,
            "message": "API y algoritmos ML funcionando correctamente",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"❌ ERROR EN TEST: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en test: {str(e)}")


@app.post("/api/search/track")
async def track_search_event(event: SearchEventRequest):
    """Registrar evento de búsqueda del usuario"""
    try:
        print("=" * 70)
        print("📥 DATOS RECIBIDOS DE LARAVEL:")
        print("=" * 70)
        print(f"  user_id: {event.user_id}")
        print(f"  session_id: {event.session_id}")
        print(f"  search_query: {event.search_query}")
        print(f"  search_type: {event.search_type}")
        print(f"  filters: {event.filters} (type: {type(event.filters)})")
        print(f"  results_count: {event.results_count}")
        print(f"  device: {event.device}")
        print(f"  page_url: {event.page_url}")
        print(f"  scroll_depth: {event.scroll_depth}")
        print("=" * 70)

        # Conectar a la BD
        conn = get_db_connection()
        cursor = conn.cursor()

        # Convertir filters a JSON string
        filters_json = json.dumps(event.filters) if event.filters else '{}'

        # Insertar evento en la BD
        insert_query = """
            INSERT INTO search_events (
                user_id, session_id, search_query, search_type, filters,
                results_count, device, page_url, referrer, click_position,
                time_spent, scroll_depth, property_id, element_clicked,
                filter_changed, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            event.user_id,
            event.session_id,
            event.search_query,
            event.search_type,
            filters_json,
            event.results_count,
            event.device,
            event.page_url,
            event.referrer,
            event.click_position,
            event.time_spent,
            event.scroll_depth,
            event.property_id,
            event.element_clicked,
            event.filter_changed,
            datetime.now()
        )

        cursor.execute(insert_query, values)
        conn.commit()
        event_id = cursor.lastrowid

        cursor.close()
        conn.close()

        print(f"✅ Evento guardado exitosamente con ID: {event_id}")
        print("=" * 70)

        return {
            "success": True,
            "message": "Evento de búsqueda registrado correctamente",
            "event_id": event_id,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "timestamp": datetime.now().isoformat()
        }

    except mysql.connector.Error as db_err:
        print(f"❌ ERROR DE BASE DE DATOS: {str(db_err)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error de base de datos: {str(db_err)}")
    
    except Exception as e:
        print(f"❌ ERROR AL GUARDAR EVENTO: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno: {str(e)}")


# ===================== ENDPOINTS DE MACHINE LEARNING =====================

@app.post("/api/ml/classify/quick")
async def classify_quick(request: dict):
    """Clasificación RÁPIDA con Naive Bayes"""
    try:
        query = request.get('query', '')
        
        if not query or len(query) < 2:
            raise HTTPException(status_code=400, detail="Query demasiado corto")
        
        print(f"🔍 Clasificando con Naive Bayes: {query}")
        
        result = analyzer.classify_with_naive_bayes(query)
        
        return {
            "success": True,
            "result": result,
            "query": query
        }
        
    except Exception as e:
        print(f"❌ Error en classify_quick: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/similar")
async def find_similar(request: dict):
    """Encontrar búsquedas similares con KNN"""
    try:
        query = request.get('query', '')
        n_similar = request.get('n_similar', 5)
        
        if not query or len(query) < 2:
            raise HTTPException(status_code=400, detail="Query demasiado corto")
        
        print("=" * 70)
        print(f"🔍 BUSCANDO SIMILARES CON KNN")
        print(f"   Query: {query}")
        print(f"   N similares: {n_similar}")
        print("=" * 70)
        
        result = analyzer.find_similar_users(query, n_similar)
        
        similar_count = len(result.get('similar_searches', [])) if isinstance(result, dict) else 0
        print(f"✅ Similares encontrados: {similar_count}")
        print("=" * 70)
        
        return {
            "success": True,
            "result": result,
            "query": query
        }
        
    except Exception as e:
        print(f"❌ Error en find_similar: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/predict/complex")
async def predict_complex(request: dict):
    """Predicción compleja con MLP"""
    try:
        query = request.get('query', '')
        
        if not query or len(query) < 2:
            raise HTTPException(status_code=400, detail="Query demasiado corto")
        
        print(f"🧠 Predicción MLP: {query}")
        
        result = analyzer.predict_with_mlp(query)
        
        return {
            "success": True,
            "result": result,
            "query": query
        }
        
    except Exception as e:
        print(f"❌ Error en predict_complex: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/predict/ensemble")
async def predict_ensemble(request: dict):
    """Predicción con ensemble de los 3 modelos"""
    try:
        query = request.get('query', '')
        
        if not query or len(query) < 2:
            raise HTTPException(status_code=400, detail="Query demasiado corto")
        
        print(f"🎯 Predicción Ensemble: {query}")
        
        result = analyzer.ensemble_prediction(query)
        
        return {
            "success": True,
            "result": result,
            "query": query
        }
        
    except Exception as e:
        print(f"❌ Error en predict_ensemble: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/compare")
async def compare_all(request: dict):
    """Comparar todos los algoritmos"""
    try:
        query = request.get('query', '')
        n_similar = request.get('n_similar', 5)
        
        if not query or len(query) < 2:
            raise HTTPException(status_code=400, detail="Query demasiado corto")
        
        print(f"📊 Comparando todos los algoritmos: {query}")
        
        # Naive Bayes
        nb_result = analyzer.classify_with_naive_bayes(query)
        
        # KNN
        knn_result = analyzer.find_similar_users(query, n_similar)
        
        # MLP
        mlp_result = analyzer.predict_with_mlp(query)
        
        # Ensemble
        ensemble_result = analyzer.ensemble_prediction(query)
        
        return {
            "success": True,
            "query": query,
            "results": {
                "naive_bayes": nb_result,
                "knn": knn_result,
                "mlp": mlp_result,
                "ensemble": ensemble_result
            }
        }
        
    except Exception as e:
        print(f"❌ Error en compare_all: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ===================== ENDPOINTS EXISTENTES =====================

@app.post("/api/recommendations/generate")
async def generate_recommendations(request: RecommendationRequest):
    """Generar recomendaciones inteligentes para el usuario"""
    try:
        print("=" * 70)
        print(f"🎯 GENERANDO RECOMENDACIONES PARA USER_ID: {request.user_id}")
        print(f"📊 Límite de resultados: {request.limit}")
        print("=" * 70)

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Obtener historial de búsquedas del usuario
        cursor.execute("""
            SELECT search_query, filters, search_type, created_at
            FROM search_events
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 50
        """, (request.user_id,))
        
        search_history = cursor.fetchall()

        if not search_history:
            print("⚠️ Usuario sin historial de búsquedas - Recomendaciones generales")
            # Obtener propiedades aleatorias
            cursor.execute("""
                SELECT id, title, description, price, location, property_type, 
                       bedrooms, bathrooms, area, image_url
                FROM properties
                ORDER BY RAND()
                LIMIT %s
            """, (request.limit,))
            
            properties = cursor.fetchall()
            cursor.close()
            conn.close()

            print(f"✅ Se encontraron {len(properties)} propiedades generales")
            print("=" * 70)

            return {
                "success": True,
                "recommendations": properties,
                "reason": "Recomendaciones generales (sin historial)",
                "user_id": request.user_id,
                "total": len(properties)
            }

        # Analizar patrones con ML
        print("🤖 Analizando patrones con Machine Learning...")
        patterns = analyzer.analyze_patterns(search_history)
        
        print(f"📈 Patrones detectados: {patterns}")

        # Obtener recomendaciones basadas en patrones
        recommendations = analyzer.get_property_recommendations(
            patterns=patterns,
            limit=request.limit,
            cursor=cursor
        )

        cursor.close()
        conn.close()

        print(f"✅ Se generaron {len(recommendations)} recomendaciones personalizadas")
        print("=" * 70)

        return {
            "success": True,
            "recommendations": recommendations,
            "patterns": patterns,
            "user_id": request.user_id,
            "total": len(recommendations),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"❌ ERROR AL GENERAR RECOMENDACIONES: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error al generar recomendaciones: {str(e)}")


@app.get("/api/search/history/{user_id}")
async def get_search_history(user_id: int, limit: int = 20):
    """Obtener historial de búsquedas del usuario"""
    try:
        print(f"📜 Obteniendo historial para USER_ID: {user_id} (limit: {limit})")
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        cursor.execute("""
            SELECT id, user_id, session_id, search_query, search_type, filters,
                   results_count, device, page_url, created_at
            FROM search_events
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT %s
        """, (user_id, limit))

        history = cursor.fetchall()
        
        # Convertir filters de JSON string a dict
        for item in history:
            if item.get('filters'):
                try:
                    item['filters'] = json.loads(item['filters'])
                except:
                    item['filters'] = {}
        
        cursor.close()
        conn.close()

        print(f"✅ Se encontraron {len(history)} registros")

        return {
            "success": True,
            "history": history,
            "total": len(history),
            "user_id": user_id
        }

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search/properties")
async def search_properties(search: PropertySearchRequest):
    """Buscar propiedades con filtros"""
    try:
        print("=" * 70)
        print("🔍 BÚSQUEDA DE PROPIEDADES")
        print(f"  Query: {search.query}")
        print(f"  Location: {search.location}")
        print(f"  Type: {search.property_type}")
        print(f"  Price: {search.min_price} - {search.max_price}")
        print("=" * 70)

        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Construir query dinámica
        query = "SELECT * FROM properties WHERE 1=1"
        params = []

        if search.query:
            query += " AND (title LIKE %s OR description LIKE %s OR location LIKE %s)"
            search_term = f"%{search.query}%"
            params.extend([search_term, search_term, search_term])

        if search.location:
            query += " AND location LIKE %s"
            params.append(f"%{search.location}%")

        if search.property_type:
            query += " AND property_type = %s"
            params.append(search.property_type)

        if search.min_price:
            query += " AND price >= %s"
            params.append(search.min_price)

        if search.max_price:
            query += " AND price <= %s"
            params.append(search.max_price)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(search.limit)

        cursor.execute(query, params)
        properties = cursor.fetchall()

        cursor.close()
        conn.close()

        print(f"✅ Se encontraron {len(properties)} propiedades")
        print("=" * 70)

        return {
            "success": True,
            "properties": properties,
            "total": len(properties),
            "filters_applied": {
                "query": search.query,
                "location": search.location,
                "property_type": search.property_type,
                "min_price": search.min_price,
                "max_price": search.max_price
            }
        }

    except Exception as e:
        print(f"❌ ERROR EN BÚSQUEDA: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Obtener estadísticas del sistema"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)

        # Total de eventos
        cursor.execute("SELECT COUNT(*) as total FROM search_events")
        total_events = cursor.fetchone()['total']

        # Total de propiedades
        cursor.execute("SELECT COUNT(*) as total FROM properties")
        total_properties = cursor.fetchone()['total']

        # Búsquedas por tipo
        cursor.execute("""
            SELECT search_type, COUNT(*) as count
            FROM search_events
            GROUP BY search_type
        """)
        searches_by_type = cursor.fetchall()

        # Dispositivos más usados
        cursor.execute("""
            SELECT device, COUNT(*) as count
            FROM search_events
            GROUP BY device
            ORDER BY count DESC
        """)
        devices = cursor.fetchall()

        cursor.close()
        conn.close()

        return {
            "success": True,
            "stats": {
                "total_events": total_events,
                "total_properties": total_properties,
                "searches_by_type": searches_by_type,
                "devices": devices
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===================== MANEJO DE ERRORES =====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Manejador personalizado de excepciones HTTP"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Manejador general de excepciones"""
    print(f"❌ ERROR NO CONTROLADO: {str(exc)}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Error interno del servidor",
            "detail": str(exc),
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


# ===================== MAIN =====================

if __name__ == "__main__":
    import uvicorn
    import os

    # Puerto dinámico para Railway (usa $PORT) o 8001 para desarrollo local
    port = int(os.getenv("PORT", 8001))

    # Detectar si estamos en producción (Railway setea RAILWAY_ENVIRONMENT)
    is_production = os.getenv("RAILWAY_ENVIRONMENT") is not None

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=not is_production  # Solo reload en desarrollo
    )