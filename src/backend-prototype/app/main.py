from fastapi import FastAPI
from app.api.routes import main_routes

app = FastAPI()

app.include_router(main_routes.router)
