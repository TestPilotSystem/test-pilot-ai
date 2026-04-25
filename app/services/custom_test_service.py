from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.config import settings


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class CustomQuestionSchema(BaseModel):
    pregunta: str = Field(description="El enunciado de la pregunta de test")
    opciones: List[str] = Field(description="Lista de 3 opciones de respuesta")
    respuesta_correcta: str = Field(description="La opción que es correcta (debe coincidir exactamente)")
    explicacion: str = Field(description="Breve explicación de por qué esa es la correcta")
    tema: str = Field(description="El tema asociado a la pregunta")
    dificultad: str = Field(description="Nivel de dificultad estimado: bajo, medio, alto")


class CustomTestSchema(BaseModel):
    preguntas: List[CustomQuestionSchema] = Field(description="Lista de 10 preguntas generadas")


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_independent_llm():
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

def generate_custom_test(student_profile: dict) -> dict:
    llm = get_independent_llm()
    parser = JsonOutputParser(pydantic_object=CustomTestSchema)

    prompt = ChatPromptTemplate.from_template(
        """Eres un experto examinador de autoescuela en España.
Genera un test personalizado de EXACTAMENTE 10 preguntas para el carnet B.

PROCESO INTERNO (no incluir en respuesta):
- Paso 1: Analiza el perfil del alumno e identifica sus áreas débiles.
- Paso 2: Selecciona los temas que más necesita reforzar.
- Paso 3: Para cada pregunta, diseña un distractor que confunda a quien no domina ese tema.
- Paso 4: Asigna dificultad alta a los temas con más fallos y baja a los ya dominados.

PERFIL DEL ALUMNO:
{profile}

INSTRUCCIONES DE SALIDA:
1. Genera exactamente 10 preguntas únicas y personalizadas al perfil.
2. Prioriza los temas con más fallos registrados en el perfil.
3. Si falta información del perfil, asume nivel medio y distribuye temas equilibradamente.
4. Opciones de respuesta claras y concisas.
5. Incluye los campos 'tema' y 'dificultad' para cada pregunta.

{format_instructions}"""
    )

    chain = prompt | llm | parser
    return chain.invoke({
        "profile": str(student_profile),
        "format_instructions": parser.get_format_instructions()
    })
