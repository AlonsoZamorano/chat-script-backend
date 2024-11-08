from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import requests
from pydantic import BaseModel

class Query(BaseModel):
    texto: str

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a ciertos dominios si es necesario
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

d = 768
index = faiss.IndexFlatL2(d)
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

all_fragments = []
# Cargamos los guiones
for file in os.listdir("cleaned_scripts"):
    with open(f"./cleaned_scripts/{file}", 'r', encoding='utf-8') as f:
        texto = f.read()

    # Dividimos el texto en fragmentos de 500 palabras
    fragmentos = [texto[i:i+500] for i in range(0, len(texto), 500)]
    all_fragments.extend(fragmentos)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/query")
async def query(query: Query):
    print("Query recibida")
    texto = query.texto
    print(texto)
    embedding = model.encode("search_document: " + texto)
    D, I = index.search(np.array([embedding], dtype='float32'), 5)

    retrieved_fragmets = [all_fragments[i] for i in I[0]]

    prompt = "Eres un asistente experto en peliculas. Por favor, responde a la siguiente pregunta: " + texto + ". Además te entregamos el siguiente contexto" + " ".join(retrieved_fragmets)
     
    # Pedimos a la API que nos genere una respuesta con el contexto generado
    # tormenta.ing.puc.cl/api/generate POST
    print(prompt)
    
    response = requests.post("http://tormenta.ing.puc.cl/api/generate", data={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": "false",
        })
    
    print(response)
    if response.status_code != 200:
        # print the error
        print(response.json())
        raise HTTPException(status_code=500, detail="Error al generar respuesta")
    
    return response.json()