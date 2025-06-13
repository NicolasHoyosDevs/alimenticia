from fastapi import FastAPI
from .Routes import router

app = FastAPI()
app.include_router(router)