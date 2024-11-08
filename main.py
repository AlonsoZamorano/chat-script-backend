from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import faiss
import numpy as np
import requests
from pydantic import BaseModel
import json

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

fragmentos_texto = np.load("fragmentos_texto.npy", allow_pickle=True)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/query")
async def query(query: Query):
    print("Query recibida")
    texto = query.texto
    responseEmbedding = requests.post("http://tormenta.ing.puc.cl/api/embed", json={
        "model": "nomic-embed-text",
        "input": texto
    })
    if responseEmbedding.status_code != 200:
        # print the error
        print("ERROR")
        raise HTTPException(status_code=500, detail="Error al generar embedding")
    print("EMBEDDING")
    embedding = responseEmbedding.json()["embeddings"]
    D, I = index.search(np.array(embedding, dtype='float32'), 5)

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
        })
    
    if response.status_code != 200:
        # print the error
        print(response.json())
        raise HTTPException(status_code=500, detail="Error al generar respuesta")
    lines = response.text.splitlines()

    # Reconstruir el contenido concatenando los campos "response"
    full_response = ""
    for line in lines:
        data = json.loads(line)
        full_response += data.get("response", "")

    return full_response