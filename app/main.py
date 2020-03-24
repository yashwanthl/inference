from fastapi import FastAPI
import uvicorn
from modules.dataset import Dataset
from models.spam import SpamOrHam
from loguru import logger
from starlette.middleware.cors import CORSMiddleware

from routers import spam, classifier, extract

app = FastAPI(title="Hubble Inference API", version='1.0', description="Hubble - Text analytics platform. Endpoints for all text analytics models")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initiate logging
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/data/log_files/logs.log', format=log_format, level='DEBUG', compression='zip')

app.include_router(
    spam.router,
    prefix="/spam",
    tags=["spam"],
    responses={404: {"description": "Not found"}}
)

app.include_router(
    classifier.router,
    prefix="/classifier",
    tags=["classifier"],
    responses={404: {"description": "Not found"}}
)

app.include_router(
    extract.router,
    prefix="/extract",
    tags=["extract"],
    responses={404: {"description": "Not found"}}
)

@app.get("/")
@app.get("/home")
def read_home():
    '''
    Home end point to test if API is up and running

    :return: Dict with key 'message' and value 'Hubble Inference API is live!. Listening you LOUD and CLEAR!'

    '''
    return {"message": "Hubble Inference API is live!. Listening you LOUD and CLEAR!"}

if __name__ == "__main__":
    uvicorn.run(app, port=80, host="0.0.0.0")

