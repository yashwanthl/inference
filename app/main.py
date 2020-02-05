from fastapi import FastAPI
import uvicorn
from modules.dataset import Dataset
from models.spam import SpamOrHam
from loguru import logger

app = FastAPI(title="Hubble Inference API", version='1.0', description="Hubble - Text analytics platform. Endpoints for all text analytics models")

# Initiate logging
log_format = "{time} | {level} | {message} | {file} | {line} | {function} | {exception}"
logger.add(sink='app/data/log_files/logs.log', format=log_format, level='DEBUG', compression='zip')

@app.get("/")
@app.get("/home")
def read_home():
    '''
    Home end point to test if API is up and running

    :return: Dict with key 'message' and value 'Hubble Inference API is live!. Listening you LOUD and CLEAR!'

    '''
    return {"message": "Hubble Inference API is live!. Listening you LOUD and CLEAR!"}


@app.get("/train/spamorham")
def train_spamorham():
    '''
    End point to tran spamorham mdoel
    '''
    try:
        ds = Dataset()
        emails = ds.get_emailcorpus()
        model = SpamOrHam()
        model.train(emails)
        model.serialize("spamorham.model")
        logger.info("Model trained and saved successfully")
        print("[INFO]: Model trained and saved successfully")
        return {"status": "Success"}
    except Exception as e:
        logger.error("Error in training the model. Error " + str(e))
        print("[ERROR]: Error in training the model. Error " + str(e))
        return {"status": "Failure", "Error": str(e)}

@app.get("/predict/spamorham")
def preidct_spamorham(emailtext: str):
    '''
    End point to predict from trained spamorham model
    '''
    try:
        md = SpamOrHam.deserialize("spamorham.model")
        predict = md.predict([emailtext])
        if (predict[0] == 0):
            logger.info("Model predicted the text as NOT SPAM")
            print("[INFO]: Model predicted the text as NOT SPAM")
            return {"status": "Success", "result": {"spam": False}}
        if (predict[0] == 1):
            logger.info("Model predicted the text as SPAM")
            print("[INFO]: Model predicted the text as SPAM")
            return {"status": "Success", "result": {"spam": True}}
    except Exception as e:
        logger.error("Error in predicting from model. Error " + str(e))
        print("[ERROR]: Error in predicting from model. Error " + str(e))
        return {"status": "Failure", "Error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")

