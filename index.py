import json
import nltk
import pandas as pd
from flask import request
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
CACHE_FILENAME = "doc_cache.json"
S = set(stopwords.words('english'))

ps =PorterStemmer()



def open_cache():
        cache_file = open(CACHE_FILENAME, 'r')
        cache_contents = cache_file.read()
        cache_dict = json.loads(cache_contents)
        cache_file.close()
    return cache_dict


def remove_stopwords(tokens):
    tokens_stop_removed = []
    for token in tokens:
        token = ps.stem(token)
        if not token.lower() in S:
            tokens_stop_removed.append(token)
    return tokens_stop_removed

def corpus_index():
    cache_dict = open_cache()
    corpus = list(cache_dict.values())
    tokenized_corpus = [remove_stopwords(str(doc).split(" ")) for doc in corpus]
    bm25plus = BM25Plus(tokenized_corpus)
    return corpus, bm25plus, cache_dict
