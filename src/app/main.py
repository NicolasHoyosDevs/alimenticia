from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .Routes import router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Orígenes permitidos
    allow_credentials=True,
    # Permite todos los métodos (GET, POST, PUT, DELETE, etc.)
    allow_methods=["*"],
    allow_headers=["*"],  # Permite todos los headers
)

app.include_router(router)