from typing import List

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.config import settings


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class FlashcardSchema(BaseModel):
    pregunta: str = Field(description="Pregunta directa, sin opciones de respuesta")
    respuesta: str = Field(description="Respuesta clara y concisa")
    explicacion: str = Field(description="Explicación breve del concepto")


class FlashcardBatchSchema(BaseModel):
    flashcards: List[FlashcardSchema] = Field(description="Lista de exactamente 20 flashcards")


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_flashcards(topic: str) -> dict:
    llm = get_flashcard_llm()
    parser = JsonOutputParser(pydantic_object=FlashcardBatchSchema)

    prompt = ChatPromptTemplate.from_template(
        """Eres un profesor experto de autoescuela en España especializado en el carnet tipo B.

TEMA: {topic}

Genera EXACTAMENTE 20 flashcards sobre el tema indicado.
Cubre conceptos variados: no repitas ideas entre flashcards.
Cada flashcard tiene pregunta directa (sin opciones), respuesta concisa y explicación breve.

{format_instructions}"""
    )

    chain = prompt | llm | parser
    response = chain.invoke({
        "topic": topic,
        "format_instructions": parser.get_format_instructions()
    })

    # Normalize in case the parser returns a list instead of a wrapped dict
    if isinstance(response, dict) and "flashcards" in response:
        return response
    if isinstance(response, list):
        return {"flashcards": response}
    return {"flashcards": []}
