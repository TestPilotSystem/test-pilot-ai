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


def get_chat_llm():
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
        temperature=settings.llm_temperature
    )


TONE_INSTRUCTIONS = {
    "formal": "Sé formal, profesional y cortés.",
    "informal": "Sé cercano y amigable, tutea al alumno.",
    "conciso": "Responde de forma muy breve y directa, solo lo esencial.",
    "detallado": "Proporciona explicaciones completas con ejemplos cuando sea posible."
}


def chat_with_tutor(question: str, chunks, tone: str = "formal", 
                    user_name: str = None, history: list = None) -> str:
    context = "\n\n".join([c.page_content for c in chunks])
    chat_llm = get_chat_llm()
    
    greeting = f"Dirígete al alumno como {user_name}. " if user_name else ""
    tone_instruction = TONE_INSTRUCTIONS.get(tone, TONE_INSTRUCTIONS["formal"])
    
    history_text = ""
    if history:
        for msg in history[-6:]:
            role = "Alumno" if msg.get("role") == "user" else "Tutor"
            history_text += f"{role}: {msg.get('content')}\n"
    
    prompt = ChatPromptTemplate.from_template(
        """Eres un tutor virtual de autoescuela en España. Tu rol es ayudar a estudiantes 
a preparar el examen teórico del carnet de conducir tipo B.

ESTILO: {tone_instruction}
{greeting}

REGLAS:
1. Responde SOLO preguntas relacionadas con el examen teórico de conducir.
2. Basa tu respuesta EXCLUSIVAMENTE en el contexto proporcionado.
3. Si la pregunta NO está relacionada con conducir/examen teórico, responde:
   "Lo siento, solo puedo ayudarte con dudas del examen teórico de conducir."
4. Considera el historial de conversación para dar continuidad.

CONTEXTO (Manual DGT):
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