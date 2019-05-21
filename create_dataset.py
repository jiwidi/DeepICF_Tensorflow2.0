# Based on code from  Harshdeep Gupta
# Description: Splits the data into train and test using the leave the latest one out strategy 


import pandas as pd
import numpy as np
import os
import requests, zipfile, io
from random import sample 
import logging
from tqdm import tqdm
import argparse

INPUT_PATH = 'data/ml-1m/'
INPUT_FILE = 'ratings.dat'

OUTPUT_PATH_TRAIN = 'movielens.train.rating'
OUTPUT_PATH_TEST = 'movielens.test.rating'
OUTPUT_PATH_TEST_NEGATIVES = 'movielens.test.negative'
OUTPUT_PATH_TRAIN_NEGATIVES = 'movielens.train.negative'
USER_FIELD = 'userID'
dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

logger = logging.getLogger()


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Create dataset")
    parser.add_argument('--input_path', nargs='?', default=INPUT_PATH,
                        help='Input data path.')                    
    parser.add_argument('--num_neg_test', type=int, default=-1,
                        help='Number of negative instances to pair with a positive instance for the test set. If -1 no negatives will be created')
    parser.add_argument('--num_neg_train', type=int, default=-1,
                        help='Number of negative instances to pair with a positive instance for the train set. If -1 no negatives will be created')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show info while running')
    parser.add_argument('--force_download', type=bool, default=False,
                        help='Forces the script to redownload the data')
    return parser.parse_args()


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        msg = self.format(record)
        tqdm.write(msg)
        self.flush()
        
        
def check_data(path,force):
    '''
    Checks if the needed data files to create the dataset are inside the provided path
    '''
    if(force or os.path.isfile(INPUT_PATH+INPUT_FILE)):
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

def get_test_negatives(transactions,negatives):
    '''
    return a negative sample dataframe, creates 4 negatives samples for every user positive rating.
    Args:
        transactions: the entire df of user/item transactions
    '''
    #Really slow, need to improve
    users=[]
    movies=[]
    ratings=[]
    list_movies = transactions.movieID.unique()
    oldhandlers = list(logger.handlers)
    logger.addHandler (TqdmLoggingHandler ())
    logging.info('Creating negatives, this will take a while')
    for user in tqdm(transactions.userID.unique()):
        user_movies = transactions[transactions.userID == user].movieID.unique()
        unseen_movies = [item for item in list_movies if item not in user_movies]
        negative_movies = sample(unseen_movies,negatives)
        for movie in negative_movies:
            users.append(user)
            movies.append(movie)
            ratings.append(1)
    negatives = pd.DataFrame({
        "userID" : users,
        "movieID" : movies,
        "rating" : ratings}
    )
    logger.handlers = oldhandlers
    return negatives

def get_train_negatives(transactions,negatives):
    '''
    return a negative sample dataframe, creates 4 negatives samples for every user positive rating.
    Args:
        transactions: the entire df of user/item transactions
    '''
    #Really slow, need to improve
    users=[]
    movies=[]
    ratings=[]
    list_movies = transactions.movieID.unique()
    oldhandlers = list(logger.handlers)
    logger.addHandler (TqdmLoggingHandler ())
    logging.info('Creating negatives, this will take a while')
    for user in tqdm(transactions.userID):
        user_movies = transactions[transactions.userID == user].movieID.unique()
        unseen_movies = [item for item in list_movies if item not in user_movies]
        negative_movies = sample(unseen_movies,negatives)
        for movie in negative_movies:
            users.append(user)
            movies.append(movie)
            ratings.append(1)
    negatives = pd.DataFrame({
        "userID" : users,
        "movieID" : movies,
        "rating" : ratings}
    )
    logger.handlers = oldhandlers
    return negatives


def report_stats(transactions, train, test, negative_test):
    '''
    return stats for a series of dataframes
    Args:
        transactions: the entire df of user/item transactions
        train: Train dataframe
        test: test dataframe
        negative_test: negative_test dataframe
    '''
    whole_size = transactions.shape[0]*1.0
    train_size = train.shape[0]
    test_size = test.shape[0]
    negative_size = negative_test.shape[0]
    print("Total No. of Records = {}".format(whole_size))
    print("Train size = {}, Test size = {} Negative Test Size = {}".format(train_size, test_size, negative_size))
    print("Train % = {}, Test % ={}".format(train_size/whole_size, test_size/whole_size))

def create_mapping(values):
    value_to_id = {value:idx for idx, value in enumerate(values.unique())}
    return value_to_id

def clean_df(transactions):
    user_mapping = create_mapping(transactions["userID"])
    item_mapping = create_mapping(transactions["movieID"])
    transactions["userID"] = transactions["userID"].map(user_mapping.get)
    transactions["movieID"] = transactions["movieID"].map(item_mapping.get)
    return transactions

def main():
    args = parse_args()
    if(args.verbose == 1):
        logger.setLevel (logging.INFO)
    else:
        logger.setLevel (logging.WARNING)
    check_data(INPUT_PATH,args.force_download)
    transactions = pd.read_csv(INPUT_PATH+INPUT_FILE, sep="::", names = ['userID', 'movieID', 'rating', 'timestamp'],engine='python')
    transactions = clean_df(transactions)
    # convert to implicit scenario
    transactions['rating'] = 1
    
    # make the dataset
    train_df, test_df = get_train_test_df(transactions)
    train_df.columns = [len(transactions.userID.unique()),len(transactions.movieID.unique()),0,0]
    test_df.columns = [len(transactions.userID.unique()),len(transactions.movieID.unique()),0,0]
    train_df.to_csv(INPUT_PATH+OUTPUT_PATH_TRAIN,index = False)
    test_df.to_csv(INPUT_PATH+OUTPUT_PATH_TEST,index = False)
    if(args.num_neg_test>0):
        negative_test = get_test_negatives(transactions,args.num_neg_test)
        negative_test.columns = [len(transactions.userID.unique()),len(transactions.movieID.unique()),0]
        negative_test.to_csv(INPUT_PATH+OUTPUT_PATH_TEST_NEGATIVES,index = False)
        report_stats(transactions, train_df, test_df,negative_test)
    if(args.num_neg_train>0):
        negative_train = get_train_negatives(transactions,args.num_neg_train)
        negative_train.columns = [len(transactions.userID.unique()),len(transactions.movieID.unique()),0]
        negative_train.to_csv(INPUT_PATH+OUTPUT_PATH_TRAIN_NEGATIVES,index = False)
        report_stats(transactions, train_df, test_df,negative_test)
    
    return 0



if __name__ == "__main__":
    main()