# Author: Harshdeep Gupta
# Date: 07 September, 2018
# Description: Splits the data into train and test using the leave the latest one out strategy 


import pandas as pd
import numpy as np
import os
import requests, zipfile, io
import logging
from tqdm import tqdm
import enlighten
import argparse

INPUT_PATH = 'data/ml-1m/'
INPUT_FILE = 'ratings.dat'

OUTPUT_PATH_TRAIN = 'movielens.train.rating'
OUTPUT_PATH_TEST = 'movielens.test.rating'
OUTPUT_PATH_TEST_NEGATIVES = 'movielens.test.rating'
USER_FIELD = 'userID'
dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

logger = logging.getLogger()


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument('--input_path', nargs='?', default=INPUT_PATH,
                        help='Input data path.')                    
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show info while running')
    return parser.parse_args()


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)
        self.flush()
        
        
def check_data(path):
    '''
    Checks if the needed data files to create the dataset are inside the provided path
    '''
    if(os.path.isfile(INPUT_PATH+INPUT_FILE)):
        return 0
    logging.info('Downloading file')
    oldhandlers = list(logger.handlers)
    response = requests.get(dataset_url,stream=True) # Need to add progress bar
    total_length = int(response.headers.get('content-length'))
    if total_length is None: # no content length header
        content = response.content
    else:
        content = bytearray(b'')
        # Setup progress bar
        logger.addHandler (TqdmLoggingHandler ())
        try:
            for x in tqdm(response.iter_content(chunk_size=4096),total = int((total_length/4096))):
                content+=x
        except:
            pass
    logger.handlers = oldhandlers
    z = zipfile.ZipFile(io.BytesIO(content))
    target = zipfile.ZipFile(io.BytesIO(content),mode= 'w')
    file = z.filelist[2]
    target.writestr('ratings.dat', z.read(file.filename))
    target.extract('ratings.dat',path)
    logging.info('Download finished')



def get_train_test_df(transactions):
    '''
    return train and test dataframe, with leave the latest one out strategy
    Args:
        transactions: the entire df of user/item transactions
    '''

    logging.info("Size of the entire dataset:{}".format(transactions.shape))
    transactions = transactions.sort_values(by = ['timestamp'])
    last_transaction_mask = transactions.duplicated(subset = {USER_FIELD}, keep = "last")
    # The last transaction mask has all the latest items of people
    # We want for the test dataset, items marked with a False
    train_df = transactions[last_transaction_mask]
    test_df = transactions[~last_transaction_mask]
    
    train_df = train_df.sort_values(by=["userID", 'timestamp'])
    test_df = test_df.sort_values(by=["userID", 'timestamp'])
    return train_df, test_df


def create_negatives():
    ##To be done
    return 0
    
def report_stats(transactions, train_df, test_df):
    whole_size = transactions.shape[0]*1.0
    train_size = train_df.shape[0]
    test_size = test_df.shape[0]
    logging.info("Total No. of Records = {}".format(whole_size))
    logging.info("Train size = {}, Test size = {}".format(train_size, test_size))
    logging.info("Train % = {}, Test % ={}".format(train_size/whole_size, test_size/whole_size))


def main():
    args = parse_args()
    if(args.verbose == 1):
        logger.setLevel (logging.INFO)
    else:
        logger.setLevel (logging.WARNING)
    check_data(INPUT_PATH)
    transactions = pd.read_csv(INPUT_PATH+INPUT_FILE, sep="::", names = ['userID', 'movieID', 'rating', 'timestamp'],engine='python')

    # convert to implicit scenario
    transactions['rating'] = 1
    
    # make the dataset
    train_df, test_df = get_train_test_df(transactions)
    train_df.to_csv(INPUT_PATH+OUTPUT_PATH_TRAIN, header = False,index = False)
    test_df.to_csv(INPUT_PATH+OUTPUT_PATH_TEST, header = False,index = False)
    report_stats(transactions, train_df, test_df)
    return 0



if __name__ == "__main__":
    main()