import sys, os
import numpy as np
from tqdm import tqdm

from utils.score import LABELS

UNRELATED = 3

def generate_data(ids, dataset):
    X_headline, X_body, y = [], [], []

    stances = get_stances_from_ids(dataset, ids)

    for stance in stances:
        y.append(LABELS.index(stance['Stance']))
        X_headline.append(stance['Headline'])
        X_body.append(dataset.articles[stance['Body ID']])

    return X_headline, X_body, y

def get_stances_from_ids(dataset, ids):
    stances = []
    for stance in dataset.stances:
        if stance['Body ID'] in ids:
            stances.append(stance)
    return stances

def collapse_stances(labels):
    "['agree', 'disagree', 'discuss', 'unrelated'] => ['unrelated','related']"
    return [ 0 if label == UNRELATED else 1 for label in labels]
