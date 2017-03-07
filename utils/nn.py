import sys
import numpy as np
from tqdm import tqdm

from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from feature_engineering import gen_or_load_feats

def generate_ff_features(ids, dataset, embedding_matrix, tokenizer, EMBEDDING_DIM):
    h, b, y = [],[],[]

    stances = get_stances_from_ids(dataset, ids)

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(dataset.articles[stance['Body ID']])

    X = sum_words(h, b, embedding_matrix, tokenizer, EMBEDDING_DIM)

    return X,y

def get_stances_from_ids(dataset, ids):
    stances = []
    for stance in dataset.stances:
        if stance['Body ID'] in ids:
            stances.append(stance)
    return stances

def sum_words(headlines, bodies, embedding_matrix, tokenizer, EMBEDDING_DIM):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):

        clean_headline = tokenizer.texts_to_sequences(headline)
        clean_body = tokenizer.texts_to_sequences(body)
        headline_embedding = np.zeros((1, EMBEDDING_DIM))
        body_embedding = np.zeros((1, EMBEDDING_DIM))
        for token_id in clean_headline:
            if token_id:
                headline_embedding += embedding_matrix[token_id[0],:]
        for token_id in clean_body:
            if token_id:
                body_embedding += embedding_matrix[token_id[0],:]
        concatenated_embedding = np.concatenate((headline_embedding, body_embedding), axis=1)
        X.append(np.concatenate((headline_embedding, body_embedding), axis=1))
    return X
