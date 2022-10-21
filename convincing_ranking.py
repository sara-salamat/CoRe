import argparse
import csv
import pickle

import faiss
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
args = parser.parse_args()

corpus = pd.read_csv('data/comments.csv')

with open('encodings/queries_'+ str(args.model_name) + 'convincing' + '.pkl', 'rb') as f:
    query_embeddings = pickle.load(f)
with open('encodings/corpus_'+ str(args.model_name) + 'convincing' + '.pkl', 'rb') as f:
    corpus_embedding = pickle.load(f)
N = corpus_embedding

d = 768
assert len(corpus) == len(corpus_embedding)

qrels_test = pd.read_csv('data/qrels/qrels.relevant.test.csv', header=None)
# test_queries_embedding = torch.load('test_queries_embeddings_combined.pt').numpy()
queries = pd.read_csv('data/posts.csv')



dict_of_rel_docs = {}
for qid in qrels_test[0].unique():
    indices = corpus[corpus['post_id'] == qid]['comment_id'].to_list()
    rel_docs = np.zeros((len(indices), d))

    for c in range(len(indices)):
        rel_docs[c,:] = corpus_embedding[indices[c]].reshape(-1,d)
    dict_of_rel_docs[qid] = rel_docs

f = open('data/results_' + str(args.model_name) + 'convincing', 'w')
writer = csv.writer(f, delimiter = ' ')


for qid in qrels_test[0].unique():
    index = faiss.IndexFlatL2(d)
    index.add(np.float32(dict_of_rel_docs[qid]))

    D, I = index.search(query_embeddings[qid].reshape((-1,d)), dict_of_rel_docs[qid].shape[0])

    lst = corpus[corpus['post_id'] == qid]['comment_id'].to_list()
    for j in range(dict_of_rel_docs[qid].shape[0]):
        row = []
        row.append(qid)
        row.append(0)
        assert lst[I[0,j]] in corpus[corpus['post_id'] == qid]['comment_id'].values
        row.append(lst[I[0,j]]) 
        row.append(j)
        row.append(1000-D[0,j])
        row.append('STANDARD')
        writer.writerow(row)

