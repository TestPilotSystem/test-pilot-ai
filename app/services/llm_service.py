from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define question schema for structured output
class QuestionSchema(BaseModel):
    pregunta: str = Field(description="El enunciado de la pregunta de test")
    opciones: List[str] = Field(description="Lista de 3 opciones de respuesta")
    respuesta_correcta: str = Field(description="La opción que es correcta (debe coincidir exactamente)")
    explicacion: str = Field(description="Breve explicación de por qué esa es la correcta")

# Initialize the LLM model
llm = ChatOllama(model="gemma2:2b", format="json", temperature=0.7)

def generate_test_from_chunks(chunks, topic_name: str):
    # Join all chunk contents to form the context
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

    # Execute the chain with the provided context and topic
    response = chain.invoke({
        "topic": topic_name,
        "context": context,
        "format_instructions": parser.get_format_instructions()
    })
    
    return response