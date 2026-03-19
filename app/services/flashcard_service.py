import json
from langchain_core.prompts import ChatPromptTemplate
from app.config import settings


def get_flashcard_llm():
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


def generate_flashcards(topic: str) -> dict:
    llm = get_flashcard_llm()

    prompt = ChatPromptTemplate.from_template(
        """Eres un profesor experto de autoescuela en España especializado en el carnet tipo B.

TEMA: {topic}

Genera EXACTAMENTE 20 flashcards sobre el tema indicado.
Cada flashcard tiene: pregunta (directa, sin opciones), respuesta (clara y concisa), explicacion (breve).

Responde SOLO con un JSON válido con esta estructura exacta:
{{
  "flashcards": [
    {{
      "pregunta": "...",
      "respuesta": "...",
      "explicacion": "..."
    }}
  ]
}}

No incluyas nada más, solo el JSON."""
    )

    chain = prompt | llm

    response = chain.invoke({"topic": topic})

    import json
    raw = response.content if hasattr(response, "content") else str(response)

    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No se encontró JSON válido en la respuesta del LLM")

    parsed = json.loads(raw[start:end])

    if "flashcards" in parsed:
        return parsed

    if isinstance(parsed, list):
        return {"flashcards": parsed}

    for key in parsed:
        if isinstance(parsed[key], list) and len(parsed[key]) > 0:
            return {"flashcards": parsed[key]}

    raise ValueError("No se pudieron extraer flashcards de la respuesta")

