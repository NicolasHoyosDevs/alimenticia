from fastapi import FastAPI
from .Routes.routes import router

app = FastAPI()
app.include_router(router)

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # O ["*"] para permitir todos (solo para desarrollo)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)