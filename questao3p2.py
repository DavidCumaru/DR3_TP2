from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chains import SimpleChain
from langchain.prompts import PromptTemplate
from langchain.llms.huggingface_hub import HuggingFaceHub

app = FastAPI()

model_name = "Helsinki-NLP/opus-mt-en-de"
llm = HuggingFaceHub(repo_id=model_name, model_kwargs={"max_length": 500})

template = PromptTemplate(input_variables=["text"], template="Translate this to German: {text}")
translation_chain = SimpleChain(llm=llm, prompt=template)

class InputText(BaseModel):
    text: str

@app.post("/translate/")
async def translate_text(input_text: InputText):
    translated_text = translation_chain.run({"text": input_text.text})
    response = {"input_text": input_text.text, "translated_text": translated_text}
    return response

"""
O uso de LangChain com o HuggingFace pode apresentar desafios, como a lentidão de modelos grandes como o Helsinki-NLP em comparação com APIs como a OpenAI, que pode ser mitigada otimizando o fluxo de trabalho ou escolhendo modelos mais rápidos. Além disso, modelos de tradução consomem muitos recursos computacionais, exigindo ambientes de execução eficientes ou servidores especializados. O ajuste fino também pode ser limitado, já que o modelo é pré-treinado, e em alguns casos, o fine-tuning é necessário para melhorar a precisão. Por fim, integrar o HuggingFace via LangChain pode ser mais complexo do que usar a API diretamente, mas oferece vantagens ao integrar com outros componentes em sistemas maiores.
"""