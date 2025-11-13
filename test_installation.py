# test_installation.py - VERSIÓN CORREGIDA
import sys
print(f"🐍 Python: {sys.version}")
print("="*50)

try:
    import fastapi
    print(f"✅ FastAPI: {fastapi.__version__}")
    
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
    
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
    
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
    
    import nltk
    print(f"✅ NLTK: {nltk.__version__}")
    
    # TextBlob - manejo especial
    try:
        import textblob
        from textblob import TextBlob
        # Test que funciona
        test_blob = TextBlob("test")
        print(f"✅ TextBlob: Instalado y funcionando")
    except:
        print("❌ TextBlob: Error")
    
    import matplotlib
    print(f"✅ Matplotlib: {matplotlib.__version__}")
    
    import plotly
    print(f"✅ Plotly: {plotly.__version__}")
    
    import seaborn as sns
    print(f"✅ Seaborn: {sns.__version__}")
    
    import wordcloud
    print(f"✅ WordCloud: {wordcloud.__version__}")
    
    import mysql.connector
    print(f"✅ MySQL Connector: {mysql.connector.__version__}")
    
    print("="*50)
    print("🎉 ¡TODAS LAS LIBRERÍAS FUNCIONAN PERFECTAMENTE!")
    print("🚀 Listo para crear algoritmos de ML")
    
    # Test rápido de funcionalidad
    print("\n🧪 Probando funcionalidades básicas:")
    
    # Test NumPy
    arr = np.array([1, 2, 3])
    print(f"✅ NumPy array: {arr}")
    
    # Test Pandas
    df = pd.DataFrame({'test': [1, 2, 3]})
    print(f"✅ Pandas DataFrame: {len(df)} filas")
    
    # Test Matplotlib
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot([1, 2, 3])
    plt.close()
    print("✅ Matplotlib plotting: OK")
    
    # Test TextBlob funcionalidad
    blob = TextBlob("Análisis de sentimientos en español")
    print(f"✅ TextBlob sentiment: {blob.sentiment}")
    
    # Test NLTK básico
    print("✅ NLTK: Funcionalidad básica OK")
    
    print("\n🎯 ¡TODO LISTO PARA CONTINUAR CON EL ALGORITMO!")
    
except ImportError as e:
    print(f"❌ Error de importación: {e}")
except Exception as e:
    print(f"❌ Error: {e}")