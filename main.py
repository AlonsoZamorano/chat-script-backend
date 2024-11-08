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

fragmentos_texto = np.load("fragmentos_texto.npy", allow_pickle=True)

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

    retrieved_fragments = []
    # Aplanar I para que sea una lista unidimensional de índices
    for indice in I[0]:
        retrieved_fragments.append(fragmentos_texto[indice])

    prompt = "Eres un asistente experto en peliculas. Por favor, responde a la siguiente pregunta: " + texto + ". Además te entregamos el siguiente contexto" + "\n".join(retrieved_fragments)
     
    # Pedimos a la API que nos genere una respuesta con el contexto generado
    # tormenta.ing.puc.cl/api/generate POST
    print(prompt)
    
    response = requests.post("http://tormenta.ing.puc.cl/api/generate", json={
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