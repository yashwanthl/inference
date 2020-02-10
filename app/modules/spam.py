import sys
sys.path.append(".")
from app.modules.dataset import Dataset
from app.models.spam import SpamOrHam
from loguru import logger

class Spam:
    def __init__(self):
        pass

    def train(self):
        '''
        train spamorham mdoel
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

    def predict(self, emailtext: str):
        '''
        predict from trained spamorham model
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

