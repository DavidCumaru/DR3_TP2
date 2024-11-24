from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = FastAPI()

class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translated_text: str

llm = ChatOpenAI(
    model="gpt-3.5-turbo", 
    openai_api_key=minha_key
)

prompt = PromptTemplate(
    input_variables=["text"],
    template="Traduza o seguinte texto para o francês: {text}"
)

translation_chain = LLMChain(llm=llm, prompt=prompt)

@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest):
    translated_text = translation_chain.run(text=request.text)
    return TranslationResponse(translated_text=translated_text.strip())

"""
A latência de resposta da OpenAI pode ser alta devido à rede e carga no servidor, sendo possível reduzir essa latência com cache ou chamadas assíncronas. A OpenAI impõe limites de uso, como número de tokens e requisições, e para contornar isso, é importante monitorar e controlar o fluxo de chamadas. O uso contínuo da API pode gerar altos custos, especialmente em grandes volumes de dados, por isso, é recomendável usá-la de forma seletiva ou considerar modelos locais. Além disso, a qualidade das traduções pode variar, e em casos críticos, é fundamental validar ou realizar o pós-processamento das traduções.
"""