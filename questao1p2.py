from fastapi import FastAPI
from pydantic import BaseModel
from langchain.llms import FakeListLLM

app = FastAPI()

class UserInput(BaseModel):
    question: str

responses = [
    "Olá! Como posso ajudar?",
    "Estou aqui para responder suas dúvidas!",
    "Essa é uma pergunta interessante!"
]

fake_llm = FakeListLLM(responses=responses)

@app.post("/chat")
def chat(input: UserInput):
    response = fake_llm(input.question)
    return {"response": response}
