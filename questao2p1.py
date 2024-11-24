from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")

class InputText(BaseModel):
    text: str

@app.post("/translate/")
async def translate_text(input_text: InputText):
    translated = translator(input_text.text, max_length=500)
    response = {"input_text": input_text.text, "translated_text": translated[0]["translation_text"]}
    return response

"""
Limitações do modelo de tradução Helsinki-NLP/opus-mt-en-fr: Precisão da Tradução: Erros em contexto, ambiguidade lexical e vocabulários técnicos. Desempenho e Escalabilidade: Latência elevada em grandes volumes e gargalos em sistemas com recursos limitados. Custo e Infraestrutura: Alto custo para rodar em GPUs e dificuldades de escalabilidade para alta demanda. Linguagem Coloquial: Problemas em traduzir gírias, expressões idiomáticas e linguagem culturalmente carregada.
"""