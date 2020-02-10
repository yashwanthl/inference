from fastapi import FastAPI
import uvicorn
from modules.dataset import Dataset
from models.spam import SpamOrHam
from loguru import logger

from routers import spam

app = FastAPI(title="Hubble Inference API", version='1.0', description="Hubble - Text analytics platform. Endpoints for all text analytics models")

# Initiate logging
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/data/log_files/logs.log', format=log_format, level='DEBUG', compression='zip')

app.include_router(
    spam.router,
    prefix="/spam",
    tags=["spam"],
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

