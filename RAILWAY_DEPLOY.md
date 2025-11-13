# 🚀 Guía de Deploy en Railway - ViveSpaces AI

Esta guía te ayudará a deployar la API de ViveSpaces AI en Railway.

## ✅ Pre-requisitos

- Cuenta en [Railway.app](https://railway.app)
- Repositorio de GitHub con este código
- Base de datos MySQL (puede ser en Railway o externa)

## 📋 Archivos de Configuración

Ya están creados y listos:

- ✅ `Procfile` - Comando para iniciar la app
- ✅ `runtime.txt` - Versión de Python (3.12)
- ✅ `.gitignore` - Archivos a ignorar
- ✅ `requirements.txt` - Dependencias Python

## 🔧 Configuración Actualizada

### Cambios Realizados:

1. **Puerto Dinámico** - `main.py` ahora usa `$PORT` de Railway
2. **MYSQL_URL Support** - Soporte para URL completa de Railway
3. **Detección de Producción** - Auto-desactiva reload en Railway

## 📦 Pasos para Deploy

### Opción A: Deploy desde GitHub (RECOMENDADO)

#### 1. Subir código a GitHub

```bash
cd d:\ViveSpaces-AI

# Inicializar git (si no está inicializado)
git init

# Agregar archivos
git add .

# Commit
git commit -m "Preparado para deploy en Railway"

# Conectar con tu repositorio
git remote add origin https://github.com/TU_USUARIO/vivespaces-ai.git

# Push
git branch -M main
git push -u origin main
```

#### 2. Crear Proyecto en Railway

1. Ve a [Railway.app](https://railway.app)
2. Click en **"New Project"**
3. Selecciona **"Deploy from GitHub repo"**
4. Autoriza Railway en GitHub
5. Selecciona tu repositorio `vivespaces-ai`
6. Railway detectará automáticamente que es Python

#### 3. Agregar MySQL a Railway (RECOMENDADO)

1. En tu proyecto de Railway, click en **"+ New"**
2. Selecciona **"Database"** → **"Add MySQL"**
3. Railway creará una base de datos MySQL automáticamente
4. Railway generará la variable `MYSQL_URL` automáticamente

#### 4. Configurar Variables de Entorno

Ve a tu servicio → **Variables** y verifica que exista:

```env
MYSQL_URL=mysql://root:password@host:port/railway
```

**Nota:** Si Railway creó la base de datos, esta variable ya está configurada automáticamente.

#### Opción B: Si tu MySQL está fuera de Railway

Si tu MySQL está en otro servidor (externo):

```env
DB_HOST=tu-host-mysql.com
DB_USER=root
DB_PASSWORD=tu_password
DB_DATABASE=vivespaces
DB_PORT=3306
```

### Opción B: Deploy con Railway CLI

#### 1. Instalar Railway CLI

```bash
npm install -g @railway/cli
```

#### 2. Login y Deploy

```bash
# Login en Railway
railway login

# Inicializar proyecto
railway init

# Deploy
railway up
```

## 🔗 Conectar con Laravel

### 1. Obtener URL de la API

Después del deploy, Railway te dará una URL:
```
https://vivespaces-ai-production.up.railway.app
```

### 2. Configurar Laravel

En tu archivo `.env` de Laravel:

```env
AI_API_URL=https://vivespaces-ai-production.up.railway.app
```

### 3. Actualizar Código Laravel

```php
// Antes:
$response = Http::post('http://localhost:8001/api/search/track', $data);

// Después:
$response = Http::post(env('AI_API_URL') . '/api/search/track', $data);
```

## 🗄️ Configurar Base de Datos

### Opción 1: MySQL en Railway (FÁCIL)

Railway maneja todo automáticamente:
- Crea la base de datos
- Genera `MYSQL_URL`
- La API la detecta automáticamente

**Solo necesitas:**
1. Importar tu estructura de tablas (`search_events`, `properties`)
2. Importar datos (si los tienes)

### Opción 2: MySQL Externa

Si usas PlanetScale, AWS RDS, u otro:

1. Obtén las credenciales de conexión
2. Agrégalas como variables de entorno en Railway
3. La API las usará automáticamente

## 📊 Verificar el Deploy

### 1. Health Check

```bash
curl https://tu-url.railway.app/health
```

Respuesta esperada:
```json
{
  "status": "healthy",
  "service": "ViveSpaces AI",
  "database": "connected",
  "ml_models": "loaded"
}
```

### 2. Test de la API

```bash
curl https://tu-url.railway.app/api/test
```

### 3. Ver Logs

En Railway Dashboard → Tu servicio → **Deployments** → **View Logs**

## ⚙️ Variables de Entorno

### Variables que Railway Genera Automáticamente:

- `PORT` - Puerto dinámico
- `RAILWAY_ENVIRONMENT` - Detecta producción
- `MYSQL_URL` - URL completa de MySQL (si usas MySQL de Railway)

### Variables que TÚ debes configurar (si MySQL es externo):

```env
DB_HOST=tu-host.com
DB_USER=root
DB_PASSWORD=tu_password
DB_DATABASE=vivespaces
DB_PORT=3306
```

## 🚨 Troubleshooting

### Error: "Application failed to respond"

- Verifica que el `Procfile` existe
- Revisa los logs en Railway
- Verifica que `PORT` se esté usando correctamente

### Error: "Database connection failed"

- Verifica que `MYSQL_URL` o las variables de DB están configuradas
- Verifica que la base de datos está accesible
- Revisa que las tablas existen

### Build muy lento

- Es normal. Las dependencias científicas (numpy, pandas, scikit-learn) son pesadas
- Primer build: 5-10 minutos
- Builds subsecuentes: más rápidos (Railway cachea)

### Error: "Out of memory"

- Railway Free Plan: 512MB RAM
- Tus dependencias ML pueden necesitar más
- Solución: Upgrade a Hobby Plan ($5/mes, 8GB RAM)

## 📝 Checklist de Deploy

- [ ] Código subido a GitHub
- [ ] Proyecto creado en Railway
- [ ] MySQL configurado (Railway o externo)
- [ ] Variables de entorno configuradas
- [ ] Deploy exitoso
- [ ] Health check responde OK
- [ ] Tablas de BD creadas/importadas
- [ ] Laravel apuntando a nueva URL
- [ ] Prueba de tracking funcionando

## 🎯 Próximos Pasos

1. **Configurar dominio personalizado** (opcional)
   - Railway permite dominios custom
   - Ejemplo: `api.vivespaces.com`

2. **Configurar CORS específico** (recomendado para producción)
   ```python
   # En main.py, cambiar:
   allow_origins=["https://tu-dominio-laravel.com"]
   ```

3. **Monitoreo**
   - Railway tiene métricas built-in
   - Considera agregar Sentry para error tracking

4. **Backups de BD**
   - Railway hace backups automáticos
   - Considera backups adicionales para producción

## 🆘 Soporte

- Railway Docs: https://docs.railway.app
- Railway Discord: https://discord.gg/railway
- ViveSpaces Issues: (tu repo de GitHub)

---

**¡Listo para producción! 🎉**
