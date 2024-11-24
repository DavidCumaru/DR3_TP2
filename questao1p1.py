from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, set_seed

app = FastAPI()

text_generator = pipeline("text-generation", model="gpt2")
set_seed(42)

class InputText(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate_text(input_text: InputText):
    generated = text_generator(input_text.prompt, max_length=50, num_return_sequences=1)
    response = {"input": input_text.prompt, "generated_text": generated[0]["generated_text"]}
    return response

"""
Limitações do modelo GPT-2 na geração de texto: Coerência: Perda de consistência em textos longos e dificuldade em manter ideias conectadas. Incoerências: Geração de conteúdo sem sentido, falsas afirmações e respostas não relacionadas ao contexto. Desempenho: Latência em hardware não otimizado e necessidade de técnicas avançadas para inferência eficiente. Conteúdo Apropriado: Reflexo de vieses dos dados de treinamento e falta de controle sobre o conteúdo gerado.
"""