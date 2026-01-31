# TestPilot AI Engine 🚗

Motor de inteligencia artificial para generación de tests de autoescuela basado en RAG (Retrieval-Augmented Generation).

## Índice

- [Requisitos Previos](#requisitos-previos)
- [Instalación](#instalación)
- [Configuración](#configuración)
- [Uso desde Terminal](#uso-desde-terminal)
- [Documentación Interactiva](#documentación-interactiva)
- [Endpoints de la API](#endpoints-de-la-api)
- [Ejemplos de Uso](#ejemplos-de-uso)

---

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado:

- **Python 3.10+**
- **Ollama** con un modelo LLM descargado (por defecto: `gemma2:9b`)

### Instalar Ollama

```bash
# Windows (PowerShell como administrador)
winget install Ollama.Ollama

# Una vez instalado, descarga el modelo
ollama pull gemma2:9b
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd test-pilot-ai
```

### 2. Crear entorno virtual

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Windows (CMD)
venv\Scripts\activate.bat

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

> **Nota para actualizaciones:** Si ya tenías una versión anterior instalada, ejecuta:
> ```bash
> pip install -r requirements.txt --upgrade
> ```

### 4. Configurar variables de entorno

```bash
# Copiar archivo de ejemplo
cp .env.example .env

# Editar según necesidades (opcional)
```

---

## Configuración

El archivo `.env` permite personalizar el comportamiento del sistema:

| Variable | Descripción | Valor por defecto |
|----------|-------------|-------------------|
| `LLM_MODEL` | Modelo de Ollama a utilizar | `gemma2:9b` |
| `LLM_TEMPERATURE` | Creatividad del modelo (0.0-1.0) | `0.7` |
| `CHUNK_SIZE` | Tamaño de chunks al procesar PDFs | `1000` |
| `CHUNK_OVERLAP` | Solapamiento entre chunks | `200` |
| `DEFAULT_SEARCH_K` | Chunks a recuperar por búsqueda | `4` |
| `EMBEDDINGS_MODEL` | Modelo de embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| `CORS_ORIGINS` | Orígenes permitidos para CORS | `http://localhost:3000` |
| `CHROMA_PATH` | Ruta de la base de datos vectorial | `vector_db` |

---

## Uso desde Terminal

### Iniciar el servidor

```bash
# Desarrollo (con recarga automática)
uvicorn app.main:app --reload

# Producción
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

El servidor estará disponible en: `http://localhost:8000`

### Verificar estado del servidor

```bash
# Comprobar que el servidor está funcionando
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health-check
```

### Subir un manual PDF

```bash
curl -X POST "http://localhost:8000/admin/ai/upload-manual" \
  -F "file=@/ruta/al/manual.pdf" \
  -F "topic=señales_trafico"
```

### Buscar en la base de datos

```bash
curl "http://localhost:8000/admin/ai/test-search?query=velocidad%20maxima&topic=señales_trafico"
```

### Generar una pregunta

```bash
curl "http://localhost:8000/admin/ai/generate-question?topic=señales_trafico"
```

### Generar un test completo

```bash
# Test de 10 preguntas (por defecto)
curl "http://localhost:8000/admin/ai/generate-full-test?topic=señales_trafico"

# Test personalizado (20 preguntas)
curl "http://localhost:8000/admin/ai/generate-full-test?topic=señales_trafico&num_questions=20"
```

### Resetear la base de datos

```bash
curl -X DELETE "http://localhost:8000/admin/ai/reset-db"
```

---

## Documentación Interactiva

FastAPI genera documentación automática e interactiva. Una vez el servidor esté corriendo, accede a:

### Swagger UI
📍 **http://localhost:8000/docs**

Interfaz interactiva que permite:
- Ver todos los endpoints disponibles
- Probar cada endpoint directamente desde el navegador
- Ver los esquemas de request/response
- Descargar la especificación OpenAPI

### ReDoc
📍 **http://localhost:8000/redoc**

Documentación alternativa con un diseño más limpio, ideal para consulta rápida.

### OpenAPI JSON
📍 **http://localhost:8000/openapi.json**

Especificación OpenAPI 3.0 en formato JSON, útil para:
- Generar clientes automáticamente
- Importar en Postman/Insomnia
- Integración con otras herramientas

---

## Endpoints de la API

### `GET /`
Verifica que el servidor está funcionando.

**Respuesta:**
```json
{
  "status": "online",
  "message": "TestPilot AI Engine funcionando"
}
```

---

### `GET /health-check`
Comprueba el estado del modelo LLM.

**Respuesta:**
```json
{
  "model": "gemma2",
  "ollama_status": "ready"
}
```

---

### `POST /admin/ai/upload-manual`
Sube y procesa un archivo PDF para alimentar la base de conocimientos.

**Parámetros (form-data):**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `file` | File | Archivo PDF a procesar |
| `topic` | String | Tema/categoría del contenido |

**Respuesta exitosa:**
```json
{
  "message": "Archivo 'manual_dgt.pdf' procesado con éxito",
  "topic": "señales_trafico",
  "chunks_created": 45
}
```

---

### `GET /admin/ai/test-search`
Busca contenido relevante en la base de datos vectorial.

**Parámetros (query):**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `query` | String | Texto de búsqueda |
| `topic` | String | Tema donde buscar |

**Respuesta:**
```json
{
  "query": "límite de velocidad",
  "topic": "señales_trafico",
  "results_found": 4,
  "data": [
    {
      "content": "El límite de velocidad en vías urbanas...",
      "metadata": { "topic": "señales_trafico", "source": "manual.pdf" }
    }
  ]
}
```

---

### `GET /admin/ai/generate-question`
Genera una pregunta de test basada en el contenido almacenado.

**Parámetros (query):**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `topic` | String | Tema del que generar la pregunta |

**Respuesta:**
```json
{
  "pregunta": "¿Cuál es la velocidad máxima en vías urbanas?",
  "opciones": ["30 km/h", "50 km/h", "80 km/h"],
  "respuesta_correcta": "50 km/h",
  "explicacion": "Según el Reglamento General de Circulación..."
}
```

---

### `GET /admin/ai/generate-full-test`
Genera un test completo con múltiples preguntas.

**Parámetros (query):**
| Campo | Tipo | Descripción |
|-------|------|-------------|
| `topic` | String | Tema del test |
| `num_questions` | Integer | Número de preguntas (default: 10) |

**Respuesta:**
```json
{
  "topic": "señales_trafico",
  "total_questions": 10,
  "test": [
    {
      "pregunta": "...",
      "opciones": ["...", "...", "..."],
      "respuesta_correcta": "...",
      "explicacion": "..."
    }
  ]
}
```

---

### `DELETE /admin/ai/reset-db`
Elimina todos los documentos de la base de datos vectorial.

**Respuesta:**
```json
{
  "message": "Base de datos vectorial reseteada con éxito"
}
```

---

## Ejemplos de Uso

### Flujo típico de trabajo

```bash
# 1. Iniciar Ollama (en otra terminal)
ollama serve

# 2. Iniciar el servidor
uvicorn app.main:app --reload

# 3. Subir un manual PDF
curl -X POST "http://localhost:8000/admin/ai/upload-manual" \
  -F "file=@manual_dgt.pdf" \
  -F "topic=normas_generales"

# 4. Generar un test de 15 preguntas
curl "http://localhost:8000/admin/ai/generate-full-test?topic=normas_generales&num_questions=15"
```

### Usando Python (requests)

```python
import requests

BASE_URL = "http://localhost:8000"

# Subir PDF
with open("manual.pdf", "rb") as f:
    response = requests.post(
        f"{BASE_URL}/admin/ai/upload-manual",
        files={"file": f},
        data={"topic": "señales"}
    )
    print(response.json())

# Generar test
response = requests.get(
    f"{BASE_URL}/admin/ai/generate-full-test",
    params={"topic": "señales", "num_questions": 5}
)
test = response.json()
print(f"Test generado con {test['total_questions']} preguntas")
```

### Usando JavaScript (fetch)

```javascript
// Generar una pregunta
const response = await fetch(
  'http://localhost:8000/admin/ai/generate-question?topic=señales_trafico'
);
const question = await response.json();
console.log(question);
```

---

## Arquitectura

```
test-pilot-ai/
├── app/
│   ├── main.py           # Punto de entrada FastAPI
│   ├── config.py         # Configuración centralizada
│   ├── routes/
│   │   └── admin_ai.py   # Endpoints de la API
│   └── services/
│       ├── llm_service.py   # Lógica de generación con LLM
│       └── rag_service.py   # Lógica de RAG y vectores
├── vector_db/            # Base de datos ChromaDB
├── .env                  # Variables de entorno
├── .env.example          # Plantilla de configuración
└── requirements.txt      # Dependencias Python
```

---

## Tecnologías

- **FastAPI** - Framework web moderno y rápido
- **LangChain** - Orquestación de LLMs y RAG
- **Ollama** - Ejecución local de modelos LLM
- **ChromaDB** - Base de datos vectorial (via `langchain-chroma`)
- **HuggingFace** - Modelos de embeddings

---

## Licencia

MIT License