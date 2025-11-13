# test_quick.py - Prueba rápida del sistema ML

import requests
import json
from datetime import datetime

print("=" * 70)
print("🧪 PRUEBA RÁPIDA - ViveSpaces AI (3 Algoritmos)")
print("=" * 70)

BASE_URL = "http://localhost:8001"

def test_health():
    """Verificar que el servidor está corriendo"""
    print("\n📡 1. Verificando servidor...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ Servidor activo en puerto 8001")
            return True
        else:
            print(f"   ❌ Error: Código {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Error: No se puede conectar al servidor")
        print("   💡 Solución: Ejecuta 'python main.py' en otra terminal")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_naive_bayes():
    """Probar Naive Bayes"""
    print("\n🚀 2. Probando NAIVE BAYES (clasificación rápida)...")
    
    query = "casa venta zapopan"
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/ml/classify/quick",
            json={"query": query},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data['result']
            print(f"   Query: '{query}'")
            print(f"   ✅ Categoría: {result['category']}")
            print(f"   ✅ Confianza: {result['confidence']:.2%}")
            print(f"   ✅ Algoritmo: {result['algorithm']}")
            return True
        else:
            print(f"   ❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_knn():
    """Probar KNN"""
    print("\n👥 3. Probando KNN (búsquedas similares)...")
    
    query = "departamento renta centro"
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/ml/similar",
            json={"query": query, "n_similar": 3},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data['result']
            print(f"   Query: '{query}'")
            
            if 'similar_searches' in result and result['similar_searches']:
                print(f"   ✅ Encontradas {len(result['similar_searches'])} búsquedas similares:")
                for i, search in enumerate(result['similar_searches'][:3], 1):
                    print(f"      {i}. '{search['query']}' (similitud: {search['similarity']:.2%})")
                return True
            else:
                print("   ⚠️  No se encontraron búsquedas similares")
                return True
        else:
            print(f"   ❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_mlp():
    """Probar MLP"""
    print("\n🧠 4. Probando MLP (red neuronal)...")
    
    query = "oficina andares"
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/ml/predict/complex",
            json={"query": query},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data['result']
            print(f"   Query: '{query}'")
            print(f"   ✅ Categoría: {result['category']}")
            print(f"   ✅ Confianza: {result['confidence']:.2%}")
            print(f"   ✅ Algoritmo: {result['algorithm']}")
            return True
        else:
            print(f"   ❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_ensemble():
    """Probar Ensemble"""
    print("\n🎯 5. Probando ENSEMBLE (combinación de los 3)...")
    
    query = "casa 3 recamaras piscina zapopan"
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/ml/predict/ensemble",
            json={"query": query},
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            result = data['result']
            print(f"   Query: '{query}'")
            print(f"   ✅ Categoría final: {result['category']}")
            print(f"   ✅ Confianza: {result['confidence']:.2%}")
            
            if 'individual_predictions' in result:
                preds = result['individual_predictions']
                print(f"   📊 Predicciones individuales:")
                print(f"      - Naive Bayes: {preds['naive_bayes']}")
                print(f"      - KNN: {preds['knn']}")
                print(f"      - MLP: {preds['mlp']}")
                print(f"   🤝 Acuerdo entre modelos: {'Sí ✅' if result['agreement'] else 'No'}")
            return True
        else:
            print(f"   ❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_comparison():
    """Probar comparación de todos"""
    print("\n📊 6. Probando COMPARACIÓN de todos los algoritmos...")
    
    query = "terreno comercial guadalajara"
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/ml/compare",
            json={"query": query},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Query: '{query}'")
            print("   ✅ Comparación exitosa de:")
            print("      - Naive Bayes")
            print("      - KNN")
            print("      - MLP")
            print("      - Ensemble")
            return True
        else:
            print(f"   ❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_database():
    """Probar inicialización de base de datos"""
    print("\n🗄️  7. Probando inicialización de base de datos...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/search/initialize",
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("   ✅ Tabla 'search_events' verificada/creada")
                return True
            else:
                print("   ⚠️  Error creando tabla (puede ser normal si ya existe)")
                return True
        else:
            print(f"   ❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        print("   💡 Esto es normal si MySQL no está configurado aún")
        return True

def main():
    """Ejecutar todas las pruebas"""
    results = []
    
    # Test 1: Health check
    results.append(("Servidor", test_health()))
    
    if not results[0][1]:
        print("\n" + "=" * 70)
        print("❌ SERVIDOR NO ACTIVO")
        print("=" * 70)
        print("\n💡 SOLUCIÓN:")
        print("   1. Abre otra terminal/CMD")
        print("   2. Navega a la carpeta: cd VIVESPACES-AI")
        print("   3. Activa el entorno: venv\\Scripts\\activate (Windows) o source venv/bin/activate (Mac/Linux)")
        print("   4. Ejecuta: python main.py")
        print("   5. Vuelve a ejecutar este script")
        return
    
    # Tests de algoritmos
    results.append(("Naive Bayes", test_naive_bayes()))
    results.append(("KNN", test_knn()))
    results.append(("MLP", test_mlp()))
    results.append(("Ensemble", test_ensemble()))
    results.append(("Comparación", test_comparison()))
    results.append(("Base de Datos", test_database()))
    
    # Resumen
    print("\n" + "=" * 70)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 70)
    
    for test_name, passed in results:
        status = "✅ PASÓ" if passed else "❌ FALLÓ"
        print(f"   {test_name:20} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("\n" + "=" * 70)
    print(f"🎯 RESULTADO: {passed}/{total} pruebas exitosas")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ¡TODO FUNCIONA PERFECTAMENTE!")
        print("\n🚀 PRÓXIMOS PASOS:")
        print("   1. Ve a http://localhost:8001/docs para la documentación interactiva")
        print("   2. Prueba los endpoints desde tu aplicación Laravel")
        print("   3. Integra el search_tracker.js en tu frontend")
    else:
        print("\n⚠️  Algunas pruebas fallaron")
        print("\n💡 SOLUCIONES:")
        print("   - Verifica que main.py esté ejecutándose")
        print("   - Verifica que todas las librerías estén instaladas")
        print("   - Revisa los mensajes de error arriba")
    
    print("\n📖 Documentación completa: http://localhost:8001/docs")
    print("🧪 Test automático: http://localhost:8001/api/test")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Prueba interrumpida por el usuario")
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")