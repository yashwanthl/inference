from fastapi import APIRouter
import sys
sys.path.append(".")
from app.modules.spam import Spam

router = APIRouter()

@router.get("/")
def status_spam():
    return {"message": "Spam model is up and running"}

@router.get("/train")
def train():
    '''
    End point to train spamorham mdoel
    '''
    module = Spam()
    response = module.train()
    return response

@router.get("/predict")
def predict(emailtext: str):
    '''
    End point to predict from trained spamorham model
    '''
    module = Spam()
    response = module.predict(emailtext)
    return response