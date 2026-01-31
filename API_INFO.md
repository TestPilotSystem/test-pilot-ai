# API TestPilot AI Engine - Chatbot Endpoint

Documentación del endpoint `/chat/` para integración con la app core (puerto 3000).

## Endpoint

```
POST http://localhost:8000/chat/
Content-Type: application/json
```

## Request Schema

| Campo | Tipo | Requerido | Default | Descripción |
|-------|------|-----------|---------|-------------|
| `question` | `string` | ✅ Sí | - | Pregunta del usuario |
| `topic` | `string` | No | `null` | Tema específico para filtrar búsqueda. Si no se envía, busca en todo el material |
| `tone` | `string` | No | `"formal"` | Tono de respuesta: `formal`, `informal`, `conciso`, `detallado` |
| `user_name` | `string` | No | `null` | Nombre del usuario para personalizar respuesta |
| `history` | `array` | No | `null` | Historial de conversación (últimos 6 mensajes usados) |

### Formato de `history`

```typescript
interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}
```

## Ejemplos de Request

### Básico (sin personalización)
```json
{
  "question": "¿Qué es un remolque?"
}
```

### Con personalización completa
```json
{
  "question": "¿Cuál es la velocidad máxima en autopista?",
  "topic": "velocidad",
  "tone": "informal",
  "user_name": "Carlos",
  "history": [
    {"role": "user", "content": "Hola, tengo dudas sobre límites de velocidad"},
    {"role": "assistant", "content": "¡Hola! Claro, pregúntame lo que necesites sobre velocidad."}
  ]
}
```

## Response Schema

```json
{
  "question": "¿Qué es un remolque?",
  "response": "Un remolque es un vehículo no autopropulsado destinado a ser arrastrado por otro vehículo..."
}
```

## Códigos de Estado

| Código | Descripción |
|--------|-------------|
| `200` | Respuesta exitosa |
| `404` | No hay material disponible para el tema/búsqueda |
| `500` | Error interno del servidor |

## Integración con Next.js (App Core)

### Función de utilidad

```typescript
// lib/api.ts
const AI_API_URL = process.env.NEXT_PUBLIC_AI_API_URL || "http://localhost:8000";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

interface ChatRequest {
  question: string;
  topic?: string;
  tone?: "formal" | "informal" | "conciso" | "detallado";
  user_name?: string;
  history?: ChatMessage[];
}

interface ChatResponse {
  question: string;
  response: string;
}

export async function sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
  const res = await fetch(`${AI_API_URL}/chat/`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!res.ok) {
    throw new Error(`Error ${res.status}: ${res.statusText}`);
  }

  return res.json();
}
```

### Uso en componente React

```tsx
// components/ChatBot.tsx
import { useState } from "react";
import { sendChatMessage } from "@/lib/api";

export function ChatBot() {
  const [messages, setMessages] = useState<{role: string; content: string}[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMessage = { role: "user" as const, content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await sendChatMessage({
        question: input,
        tone: "formal",
        history: messages.slice(-6) // Últimos 6 mensajes
      });

      setMessages(prev => [...prev, { role: "assistant", content: response.response }]);
    } catch (error) {
      console.error("Error en chat:", error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {messages.map((msg, i) => (
        <div key={i} className={msg.role === "user" ? "user-msg" : "bot-msg"}>
          {msg.content}
        </div>
      ))}
      <input value={input} onChange={e => setInput(e.target.value)} />
      <button onClick={handleSend} disabled={loading}>
        {loading ? "Pensando..." : "Enviar"}
      </button>
    </div>
  );
}
```

## Variables de Entorno (App Core)

```env
# .env.local
NEXT_PUBLIC_AI_API_URL=http://localhost:8000
```

## Notas Importantes

1. **CORS**: La API permite requests desde `http://localhost:3000` por defecto
2. **Historial**: Solo se procesan los últimos 6 mensajes para optimizar tokens
3. **Topic opcional**: Si no se envía topic, busca en todo el material disponible
4. **Tonos**: El tono afecta cómo responde el tutor (más/menos formal, más/menos detallado)
