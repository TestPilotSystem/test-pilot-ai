import hashlib
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from cachetools import TTLCache

from app.config import settings

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class QuestionSchema(BaseModel):
    pregunta: str = Field(description="El enunciado de la pregunta de test")
    opciones: List[str] = Field(description="Lista de 3 opciones de respuesta")
    respuesta_correcta: str = Field(description="La opción que es correcta (debe coincidir exactamente)")
    explicacion: str = Field(description="Breve explicación de por qué esa es la correcta")


class TestSchema(BaseModel):
    preguntas: List[QuestionSchema] = Field(description="Lista de preguntas de test generadas")


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm():
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.llm_model,
        format="json",
        temperature=settings.llm_temperature
    )


def get_chat_llm():
    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=settings.llm_temperature
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(model=settings.llm_model, temperature=settings.llm_temperature)


llm = get_llm()

# ---------------------------------------------------------------------------
# Response cache: 200 entries, 1-hour TTL
# ---------------------------------------------------------------------------

_question_cache: TTLCache = TTLCache(maxsize=200, ttl=3600)


def _cache_key(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def _format_context(chunks) -> str:
    """Wrap each retrieved chunk in a labelled delimiter to reduce hallucinations."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[FRAGMENTO {i}]\n{chunk.page_content}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Few-shot example injected into question prompts
# ---------------------------------------------------------------------------

_FEW_SHOT = """EJEMPLO DE PREGUNTA BIEN FORMADA:
Pregunta: ¿Cuál es la velocidad máxima en autopista para un turismo?
Opciones: ["100 km/h", "120 km/h", "110 km/h"]
Respuesta correcta: 120 km/h
Explicación: El Reglamento General de Circulación fija 120 km/h como límite genérico en autopistas y autovías para turismos.
"""

# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------

def generate_test_from_chunks(chunks, topic_name: str):
    # Check cache before calling the LLM
    key = _cache_key("single", topic_name, "".join(c.page_content[:60] for c in chunks[:3]))
    if key in _question_cache:
        return _question_cache[key]

    context = _format_context(chunks)
    parser = JsonOutputParser(pydantic_object=QuestionSchema)

    prompt = ChatPromptTemplate.from_template(
        """Eres un profesor de autoescuela experto en el carnet tipo B.

{few_shot}

CONTEXTO (Manual DGT - {topic}):
{context}

INSTRUCCIONES:
1. Identifica el concepto más importante del contexto.
2. Crea 1 pregunta basada EXCLUSIVAMENTE en ese concepto.
3. Genera 2 distractores plausibles pero incorrectos y 1 respuesta correcta.
4. Escribe una explicación breve citando el manual.
5. Si la respuesta no aparece en los fragmentos anteriores, no la inventes.

{format_instructions}"""
    )

    chain = prompt | llm | parser
    result = chain.invoke({
        "topic": topic_name,
        "context": context,
        "few_shot": _FEW_SHOT,
        "format_instructions": parser.get_format_instructions()
    })

    _question_cache[key] = result
    return result


def generate_bulk_questions(chunks, topic_name: str, count: int):
    context = _format_context(chunks)
    parser = JsonOutputParser(pydantic_object=TestSchema)

    prompt = ChatPromptTemplate.from_template(
        """Eres un profesor de autoescuela experto en el carnet tipo B.

PROCESO INTERNO (no incluir en respuesta):
- Paso 1: Identifica {count} conceptos distintos en el contexto (velocidad, señales, prioridad, etc.).
- Paso 2: Para cada concepto, diseña una pregunta con distractores plausibles pero incorrectos.
- Paso 3: Verifica que ningún concepto se repita entre preguntas.

CONTEXTO:
{context}

INSTRUCCIONES DE SALIDA:
1. Genera exactamente {count} preguntas únicas, cada una sobre un concepto diferente.
2. Los distractores deben ser plausibles para quien no domina el tema.
3. Lenguaje claro, profesional, en español.
4. Si la información no está en los fragmentos, no la inventes.

{format_instructions}"""
    )

    chain = prompt | llm | parser
    return chain.invoke({
        "topic": topic_name,
        "context": context,
        "count": count,
        "format_instructions": parser.get_format_instructions()
    })


# ---------------------------------------------------------------------------
# Chat tutor
# ---------------------------------------------------------------------------

TONE_INSTRUCTIONS = {
    "formal": "Sé formal, profesional y cortés.",
    "informal": "Sé cercano y amigable, tutea al alumno.",
    "conciso": "Responde de forma muy breve y directa, solo lo esencial.",
    "detallado": "Proporciona explicaciones completas con ejemplos cuando sea posible."
}


def chat_with_tutor(question: str, chunks, tone: str = "formal",
                    user_name: str = None, history: list = None) -> str:
    context = _format_context(chunks)
    chat_llm = get_chat_llm()

    greeting = f"Dirígete al alumno como {user_name}. " if user_name else ""
    tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["formal"])

    history_text = ""
    if history:
        for msg in history[-10:]:  # keep last 10 turns (up from 6)
            role = "Alumno" if msg.get("role") == "user" else "Tutor"
            history_text += f"{role}: {msg.get('content')}\n"

    prompt = ChatPromptTemplate.from_template(
        """Eres un tutor virtual de autoescuela en España. Tu rol es ayudar a estudiantes
a preparar el examen teórico del carnet de conducir tipo B.

ESTILO: {tone_instruction}
{greeting}

REGLAS:
1. Responde SOLO preguntas relacionadas con el examen teórico de conducir.
2. Basa tu respuesta EXCLUSIVAMENTE en los fragmentos de contexto proporcionados.
3. Si la información NO aparece en los fragmentos, responde:
   "No tengo información suficiente sobre eso en el manual. Consulta la sección correspondiente."
4. Si la pregunta NO está relacionada con conducir/examen teórico, responde:
   "Lo siento, solo puedo ayudarte con dudas del examen teórico de conducir."
5. Considera el historial de conversación para dar continuidad.

FRAGMENTOS DEL MANUAL DGT:
{context}

HISTORIAL DE CONVERSACIÓN:
{history_text}

PREGUNTA ACTUAL DEL ALUMNO:
{question}

RESPUESTA:"""
    )

    chain = prompt | chat_llm
    response = chain.invoke({
        "context": context,
        "question": question,
        "tone_instruction": tone_instruction,
        "greeting": greeting,
        "history_text": history_text if history_text else "(Sin historial previo)"
    })
    return response.content
