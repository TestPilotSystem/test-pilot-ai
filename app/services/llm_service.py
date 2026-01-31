from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

from app.config import settings


class QuestionSchema(BaseModel):
    pregunta: str = Field(description="El enunciado de la pregunta de test")
    opciones: List[str] = Field(description="Lista de 3 opciones de respuesta")
    respuesta_correcta: str = Field(description="La opción que es correcta (debe coincidir exactamente)")
    explicacion: str = Field(description="Breve explicación de por qué esa es la correcta")


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


llm = get_llm()

def generate_test_from_chunks(chunks, topic_name: str):
    context = "\n\n".join([c.page_content for c in chunks])
    
    parser = JsonOutputParser(pydantic_object=QuestionSchema)

    prompt = ChatPromptTemplate.from_template(
        """Eres un profesor de autoescuela experto. Tu tarea es generar una pregunta de test profesional para el carnet tipo B.
        
        CONTEXTO (Manual de la DGT - {topic}):
        {context}
        
        INSTRUCCIONES:
        1. Crea 1 pregunta basada EXCLUSIVAMENTE en el contexto anterior.
        2. Proporciona 3 opciones de respuesta claras.
        3. Indica la respuesta correcta.
        4. Escribe una breve explicación basada en el manual.
        
        {format_instructions}"""
    )

    chain = prompt | llm | parser

    response = chain.invoke({
        "topic": topic_name,
        "context": context,
        "format_instructions": parser.get_format_instructions()
    })
    
    return response

class TestSchema(BaseModel):
    preguntas: List[QuestionSchema] = Field(description="Lista de preguntas de test generadas")

def generate_bulk_questions(chunks, topic_name: str, count: int):
    context = "\n\n".join([c.page_content for c in chunks])
    parser = JsonOutputParser(pydantic_object=TestSchema)

    prompt = ChatPromptTemplate.from_template(
        """Eres un profesor de autoescuela. Genera {count} preguntas de test distintas.
        
        CONTEXTO:
        {context}
        
        INSTRUCCIONES:
        1. No repitas conceptos. Si una pregunta es sobre velocidad, la siguiente debe ser sobre otro detalle del texto.
        2. Usa un lenguaje claro y profesional.
        3. Formato estrictamente JSON.
        
        {format_instructions}"""
    )

    chain = prompt | llm | parser
    return chain.invoke({
        "topic": topic_name, 
        "context": context, 
        "count": count,
        "format_instructions": parser.get_format_instructions()
    })