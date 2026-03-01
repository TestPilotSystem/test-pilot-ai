from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.config import settings

class CustomQuestionSchema(BaseModel):
    pregunta: str = Field(description="El enunciado de la pregunta de test")
    opciones: List[str] = Field(description="Lista de 3 opciones de respuesta")
    respuesta_correcta: str = Field(description="La opción que es correcta (debe coincidir exactamente)")
    explicacion: str = Field(description="Breve explicación de por qué esa es la correcta")
    tema: str = Field(description="El tema asociado a la pregunta")
    dificultad: str = Field(description="Nivel de dificultad estimado: bajo, medio, alto")

class CustomTestSchema(BaseModel):
    preguntas: List[CustomQuestionSchema] = Field(description="Lista de 10 preguntas generadas")

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

def generate_custom_test(student_profile: dict) -> dict:
    llm = get_independent_llm()
    parser = JsonOutputParser(pydantic_object=CustomTestSchema)
    
    prompt = ChatPromptTemplate.from_template(
        """Eres un experto examinador de autoescuela en España.
        Genera un test personalizado de EXACTAMENTE 10 preguntas para el carnet B.
        
        PERFIL DEL ALUMNO:
        {profile}
        
        INSTRUCCIONES:
        1. Genera 10 preguntas únicas.
        2. Céntrate en los temas donde el alumno tiene más fallos si se indican.
        3. Si falta información, asume un perfil medio y genera un test equilibrado.
        4. Opciones claras y concisas.
        5. Formato estrictamente JSON.
        6. Incluye el campo 'tema' y 'dificultad' para cada pregunta.
        
        {format_instructions}"""
    )
    
    profile_str = str(student_profile)
    
    chain = prompt | llm | parser
    
    response = chain.invoke({
        "profile": profile_str,
        "format_instructions": parser.get_format_instructions()
    })
    
    return response
