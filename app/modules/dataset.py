import pandas as pd
from loguru import logger

class Dataset:
    def __init__(self, data_dir='app/data/raw'):
        self.data_dir = data_dir

    def get_emailcorpus(self):
        '''
        read email.csv file from directory and return as panda data frame

        :return: emails as panda dataframe

        '''
        file = self.data_dir + '/emails.csv'
        logger.info("Fetching data from " + file)
        print("[INFO]: Fetching data from " + file)
        return pd.read_csv(file)