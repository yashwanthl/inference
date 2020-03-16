import sys
sys.path.append(".")
import ast
import re
import json
from app.models.extractor import Extractor
from app.modules.dataset import Dataset
from typing import List
from loguru import logger
import json

class ExtractorModule:
    def __init__(self):
        pass

    def extract(self, text: str, name: str = None):
        '''
        Extract entities from spaCy ner model or pre trained spaCy ner model

        parameters
        @text - text from which entities should be extracted
        @name - Optional - name of per trained spaCy ner model
        '''
        try:
            doc = Extractor.extact(text, name)
            if (doc is not None):
                logger.info("Fetched entities")
                logger.info("Building each entity object")
                entities = []
                for ent in doc.ents:
                    thisEnt = {
                        "text": ent.text,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "label": ent.label_
                    }
                    logger.info(thisEnt)
                    entities.append(thisEnt)
                return {"status": True, "Entities": entities}
            else:
                return {"status": False, "Error": "Error in fetching entities"}
        except Exception as e:
            logger.error("Error in extracting entities. Error " + str(e))
            return {"status": False, "Error": str(e)}
    
    def train_spacy(self, name: str, belongsTo: str, data = None, iterations = 20):
        '''
        Train spaCy ner

        Parameters
        @data - Training data
        @iterations - default to 20
        '''
        try:
            extractor = Extractor(name, belongsTo)
            success = False
            if data is None:
                ds = Dataset()
                data = ds.get_sampleannotations()
                # ast.literal_eval converts data from a string to list of tuples 
                success = extractor.train_spacy(data = ast.literal_eval(data), iterations = iterations)
            else:
                success = extractor.train_spacy(data = data, iterations = iterations)
            if (success is not None and success):
                return {"status": True, "Message": "Successfully craeted and trained model"}
            else:
                return {"status": False, "Message": "Error in creating and training model"}
        except Exception as e:
            logger.error("Error in creating and training ner. Error " + str(e))
            return {"status": False, "Error": str(e)}
    
    def regex_match(self, words: List[str]):
        '''
        End point to match various regex types

        Regex types supported
        Date
        Phone Number
        '''
        try:
            response = []
            allEntities = ['DATE', 'PHONE']
            for eachWord in words:
                logger.info("Finding regex matches for: " + eachWord)
                thisResponse = {
                    "word": eachWord,
                    "matches": []
                }
                logger.info("Finding PHONE regex matches for: " + eachWord)
                if (self.regex_match_phone_number(eachWord)):
                    thisResponse["matches"].append("PHONE")
                logger.info("Finding DATE regex matches for: " + eachWord)
                if (self.regex_match_date(eachWord)):
                    thisResponse["matches"].append("DATE")
                response.append(thisResponse)
            response = json.loads( json.dumps(response))
            return {"status": True, "AllRegex": allEntities, "Entities": response}
        except Exception as e:
            logger.error("Error in matching regex. Error " + str(e))
            return {"status": False, "Error": str(e)}
        return True
    
    def regex_match_phone_number(self, text: str):
        '''
        check if text is of phone number regex match

        :returns: bool
        '''
        phoneNumRegex = re.compile(r'^[2-9]\d{2}-\d{3}-\d{4}$')
        match = phoneNumRegex.match(text)
        if match:
            return True
        return False

    def regex_match_date(self, text: str):
        '''
        check if text is of date format(s)

        :returns: bool
        '''
        # case: dd/mm/yyyy or mm-dd-yyyy or mm.dd.yyyy or mmm-dd-yyyy or mmm/dd/yyyy or mmm.dd.yyyy In all the cases year can be yy also
        # dateRegex = re.compile(r'^(?:(?:0?[13578]|1[02]|(?:Jan|Mar|May|Jul|Aug|Oct|Dec))(\/|-|\.)(?:0?[1-9]|1\d|2\d|3[0-1])\1|(?:0?[469]|1[1]|(?:Apr|Jun|Sep|Nov))(\/|-|\.)(?:0?[1-9]|1\d|2\d|30)\2)(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:0?2|(?:Feb))(\/|-|\.)29\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00)))$|^(?:0?2|(?:Feb))(\/|-|\.)(?:0?[1-9]|1\d|2[0-8])\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$')
        dateRegex = re.compile(r'^(?:(?:(?:(?:0?[13578]|1[02]|(?:Jan|Mar|May|Jul|Aug|Oct|Dec))(\/|-|\.)(?:0?[1-9]|1\d|2\d|3[0-1])\1|(?:0?[469]|1[1]|(?:Apr|Jun|Sep|Nov))(\/|-|\.)(?:0?[1-9]|1\d|2\d|30)\2)(?:(?:1[6-9]|[2-9]\d)?\d{2})|(?:0?2|(?:Feb))(\/|-|\.)29\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))|(?:0?2|(?:Feb))(\/|-|\.)(?:0?[1-9]|1\d|2[0-8])\4(?:(?:1[6-9]|[2-9]\d)?\d{2}))$')
        match = dateRegex.match(text)
        if match:
            return True
        
        # case: dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy or dd-mmm-yyyy or dd/mmm/yyyy or dd.mmm.yyyy In all the cases year can be yy also
        dateRegex = re.compile(r'^(?:(?:31(\/|-|\.)(?:0?[13578]|1[02]|(?:Jan|Mar|May|Jul|Aug|Oct|Dec)))\1|(?:(?:29|30)(\/|-|\.)(?:0?[1,3-9]|1[0-2]|(?:Jan|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\2))(?:(?:1[6-9]|[2-9]\d)?\d{2})$|^(?:29(\/|-|\.)(?:0?2|(?:Feb))\3(?:(?:(?:1[6-9]|[2-9]\d)?(?:0[48]|[2468][048]|[13579][26])|(?:(?:16|[2468][048]|[3579][26])00))))$|^(?:0?[1-9]|1\d|2[0-8])(\/|-|\.)(?:(?:0?[1-9]|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep))|(?:1[0-2]|(?:Oct|Nov|Dec)))\4(?:(?:1[6-9]|[2-9]\d)?\d{2})$')
        match = dateRegex.match(text)
        if match:
            return True

        # case: dd MMM yyyy
        dateRegex = re.compile(r'^((31(?!\ (Feb(ruary)?|Apr(il)?|June?|(Sep(?=\b|t)t?|Nov)(ember)?)))|((30|29)(?!\ Feb(ruary)?))|(29(?=\ Feb(ruary)?\ (((1[6-9]|[2-9]\d)(0[48]|[2468][048]|[13579][26])|((16|[2468][048]|[3579][26])00)))))|(0?[1-9])|1\d|2[0-8])\ (Jan(uary)?|Feb(ruary)?|Ma(r(ch)?|y)|Apr(il)?|Ju((ly?)|(ne?))|Aug(ust)?|Oct(ober)?|(Sep(?=\b|t)t?|Nov|Dec)(ember)?)\ ((1[6-9]|[2-9]\d)\d{2})$')
        match = dateRegex.match(text)
        if match:
            return True
        
        return False

