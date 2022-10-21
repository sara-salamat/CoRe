import faiss
import pickle
import csv
import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
args = parser.parse_args()

with open('encodings/' + 'queries_'+ str(args.model_name) + 'relevant' + '.pkl', 'rb') as f:
    query_embeddings = pickle.load(f)
with open('encodings/' + 'corpus_'+ str(args.model_name) + 'relevant' + '.pkl', 'rb') as f:
    encoded_corpus = pickle.load(f)

N = len(encoded_corpus)
d = 768
print('corpus size N: '+ str(N))
print('vector size d: '+ str(d))
corpus_ids = []
corpus_embedding = np.zeros((len(encoded_corpus), d), dtype='float32')
c = 0
for key in encoded_corpus:
    corpus_ids.append(key)
    corpus_embedding[c,:] = encoded_corpus[key].reshape(-1,d)
    c = c + 1



index = faiss.IndexFlatL2(d)
index.add(corpus_embedding) # corpus_embedding is N by d

print(index.ntotal)

k = 1000 # No of similar vectors

def unique(list1):
  
    # initialize a null list
    unique_list = []
      
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # print list
    return unique_list

qrels_test = pd.read_csv('data/qrels/qrels.relevant.test.csv',header=None)




test_queries_embedding = np.zeros((len(qrels_test[0].unique()), d), dtype='float32')
c = 0
for qid in qrels_test[0].unique():
    test_queries_embedding[c,:] = query_embeddings[qid]
    c = c + 1
D, I = index.search(test_queries_embedding, k) 


f = open('data/results_' + str(args.model_name) + 'relevant', 'w')
writer = csv.writer(f, delimiter = ' ')
# writer.writerow(['qid','Q0' ,'docid', 'rank', 'score', 'STANDARD'])

qids = list(qrels_test[0].unique())
assert D.shape[0] == len(qids)
assert D.shape[1] == k
for i in range(len(qids)):
    for j in range(k):
        row = []
        row.append(qids[i])
        row.append(0)
        row.append(corpus_ids[I[i,j]])
        row.append(j)
        row.append(1000 - D[i,j])
        row.append('STANDARD')
        writer.writerow(row)
