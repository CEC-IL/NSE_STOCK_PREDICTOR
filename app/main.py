from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import Base, engine
from app.api.v1.stocks import router as stocks_router
from app.api.v1.predictions import router as predictions_router

#ensure all the tables exist
Base.metadata.create_all(bind=engine)

#initialize fastapi with metadata
app = FastAPI(
    title = settings.PROJECT_NAME,
    version = "1.0.0",
    description = "Backend serving NSE Stock data, features, and machine learning predictions"
)

#cors configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins = settings.CORS_ORIGINS,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

#mounting the api routers
app.include_router(stocks_router, prefix = settings.API_V1_STR)
app.include_router(predictions_router, prefix = settings.API_V1_STR)

#root endpoint
@app.get("/")
def root():
    return {
        "message": "Welcome to the NSE Stock Predictor API",
        "documentation": "/docs",
        "health":"operational"
    }