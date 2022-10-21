import argparse
import logging
import os
import pickle
import random
import sys
from datetime import datetime
from shutil import copyfile

import pandas as pd
from sentence_transformers import (InputExample, LoggingHandler,
                                   SentenceTransformer, losses,
                                   models)
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--epochs", default=10, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")

args = parser.parse_args()

logging.info(str(args))


# The  model we want to fine-tune
train_batch_size = args.train_batch_size          #Increasing the train batch size improves the model performance, but requires more GPU memory
model_name = args.model_name
max_seq_length = args.max_seq_length  
num_epochs = args.epochs         # Number of epochs we want to train

# Load our embedding model
if args.use_pre_trained_model:
    logging.info("use pretrained SBERT model")
    model = SentenceTransformer(model_name)
    model.max_seq_length = max_seq_length
else:
    logging.info("Create new SBERT model")
    word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), args.pooling)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

model_save_path = f'output/train-convincing-{model_name.replace("/", "-")}-batch_size_{train_batch_size}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

os.makedirs(model_save_path, exist_ok=True)
train_script_path = os.path.join(model_save_path, 'train_script.py')
copyfile(__file__, train_script_path)

with open(train_script_path, 'a') as fOut:
    fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
corpus = {}         #dict in the format: passage_id -> passage. Stores all existent passages

comments = pd.read_csv('data/comments.csv')
for comment_id,text in zip(comments['comment_id'].to_list(), comments['text'].to_list()):
    corpus[comment_id] = text

### Read the train queries, store in queries dict
queries = {}        #dict in the format: query_id -> query. Stores all training queries
train_queries = pd.read_csv('data/qrels/qrels.convincing.train.csv')
posts = pd.read_csv('data/posts.csv')
for qid in train_queries['0'].unique():
    q = posts[posts['post_id']==qid]['title'].values[0] + ' ' + posts[posts['post_id']==qid]['text'].values[0]
    queries[qid] = q

print('len ',len(train_queries))

print("making train queries dict")
# making train_queries dict
def get_pos_neg_ids(comments, qid):
    pos_id = []
    neg_id = []
    pos_levels = []
    for docid, conv_level in zip(comments[comments['post_id']==qid]['comment_id'], comments[comments['post_id']==qid]['conv_level']):
        if conv_level == 0:
            neg_id.append(docid)
        else:
            pos_id.append(docid)
            pos_levels.append(conv_level)
    return pos_id, neg_id, pos_levels

train_queries_dict = {}
cnt = 0
for qid in train_queries['0'].unique():
    if cnt % 100 ==0:
        print(qid)
    query = queries[qid]
    pos_id, neg_id, pos_levels = get_pos_neg_ids(comments, qid)
    sorted_pos_id = sorted(list(zip(pos_id,pos_levels)), key= lambda x: x[1])
    if len(neg_id) == 0:
        neg_id.append(sorted_pos_id[0][0])
    train_queries_dict[qid] = {
        'qid': qid,
        'query':query,
        'pos':pos_id,
        'neg': neg_id
    }
    cnt = cnt +1
with open('train_queries_dict_convincing.pkl', 'wb') as f:
    pickle.dump(train_queries_dict, f)
# with open('train_queries_dict_convincing.pkl', 'rb') as f:
#     train_queries_dict = pickle.load(f)

class CMVDataset(Dataset):
    def __init__(self, queries, corpus):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus

        for qid in self.queries:
            self.queries[qid]['pos'] = list(self.queries[qid]['pos'])
            self.queries[qid]['neg'] = list(self.queries[qid]['neg'])
            random.shuffle(self.queries[qid]['neg'])

    def __getitem__(self, item):
        query = self.queries[self.queries_ids[item]]
        query_text = query['query']
        qid = query['qid']

        if len(query['pos']) > 0:
            pos_id = query['pos'].pop(0)    #Pop positive and add at end
            pos_text = self.corpus[pos_id]
            query['pos'].append(pos_id)
        else:   #We only have negatives, use two negs
            pos_id = query['neg'].pop(0)    #Pop negative and add at end
            pos_text = self.corpus[pos_id]
            query['neg'].append(pos_id)
        
        #Get a negative passage
        neg_id = query['neg'].pop(0)    #Pop negative and add at end
        neg_text = self.corpus[neg_id]
        query['neg'].append(neg_id)

        pos_score = comments[comments['comment_id']==pos_id]['conv_level'].values[0]
        neg_score = 0

        return InputExample(texts=[query_text, pos_text, neg_text], label=float(pos_score-neg_score))

    def __len__(self):
        return len(self.queries)

print("defining datasets")
# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = CMVDataset(queries=train_queries_dict, corpus=corpus)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MarginMSELoss(model=model)

# Train the model
print("training model")
model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          warmup_steps=args.warmup_steps,
          use_amp=True,
          checkpoint_path=model_save_path,
          checkpoint_save_steps=10000,
          optimizer_params = {'lr': args.lr},
          )

# Train latest model

model.save(model_save_path)

encoded_corpus = {}
logging.info("encoding corpus")
for key,value in corpus.items():
    encoded_corpus[key] = model.encode(value)

with open('encodings/corpus_'+ str(args.model_name) + 'convincing' + '.pkl', 'wb') as f:
    pickle.dump(encoded_corpus, f)
logging.info("encoding queries")
encoded_qs = {}
all_qs = (posts['title'] + ' ' + posts['text']).to_list()
for qid, q in zip(posts['post_id'].to_list(), all_qs):
    encoded_qs[qid] = model.encode(q)

with open('encodings/queries_'+ str(args.model_name) + 'convincing' + '.pkl', 'wb') as f:
    pickle.dump(encoded_qs, f)
