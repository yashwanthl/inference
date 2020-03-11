from fastapi import APIRouter
import sys
sys.path.append(".")
from loguru import logger

router = APIRouter()

@router.get("/")
def get(text: str):
    '''
    End point to extract labels from a give text 

    parameters
    text: text from which labels has to be extracted
    '''
    return {"status": True, "text": text}