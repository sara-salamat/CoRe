import sys

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
import os

from torch.utils.data import Dataset
import random
from shutil import copyfile
import pickle
import argparse
import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=16, type=int)
parser.add_argument("--max_seq_length", default=300, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--ce_model_name", required=True)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--warmup_steps", default=1000, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--num_negs_per_system", default=5, type=int)
parser.add_argument("--use_pre_trained_model", default=False, action="store_true")
parser.add_argument("--use_all_queries", default=False, action="store_true")
args = parser.parse_args()
#First, we define the transformer model we want to fine-tune
model_name = args.ce_model_name
train_batch_size = 16
num_epochs = 1
model_save_path = 'output/training_relevant-'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# We train the network with as a binary label task
# Given [query, passage] is the label 0 = irrelevant or 1 = relevant?
# We use a positive-to-negative ratio: For 1 positive sample (label 1) we include 4 negative samples (label 0)
# in our training setup. For the negative samples, we use the triplets provided by MS Marco that
# specify (query, positive sample, negative sample).
pos_neg_ration = 4

# Maximal number of training samples we want to use
max_train_samples = 2e7

#We set num_labels=1, which predicts a continous score between 0 and 1
ce_model = CrossEncoder(model_name, num_labels=1, max_length=512)


### Now we read the MS Marco dataset

#### Read the corpus files, that contain all the passages. Store them in the corpus dict
print('reading corpus')
corpus = {}
comments = pd.read_csv('data/comments.csv')
for comment_id,text in zip(comments['comment_id'].to_list(), comments['text'].to_list()):
    corpus[comment_id] = text

### Read the train queries, store in queries dict
queries = {}
print('reading train queries')
train_queries = pd.read_csv('data/qrels/qrels.convincing.train.csv')

print(len(train_queries))
posts = pd.read_csv('data/posts.csv')

for qid in train_queries['0'].unique():
    q = posts[posts['post_id']==qid]['title'].values[0] + ' ' + posts[posts['post_id']==qid]['text'].values[0]
    queries[qid] = q

### Now we create our training & dev data
train_samples = []
dev_samples = {}

dev_queries = pd.read_csv('data/qrels/qrels.convincing.dev.csv',header=None)
print('reading dev queries')
# (qid, pos_id, neg_id)
for qid in dev_queries[0].unique():
    dev_samples[qid] = {'query': posts[posts['post_id']==qid]['title'].values[0].strip() + ' ' + posts[posts['post_id']==qid]['text'].values[0].strip(), 'positive': set(), 'negative': set()}
    rel_docs = comments[comments['post_id']==qid]['text'].to_list()
    if len(rel_docs)>30:
        rel_docs = rel_docs[:30]
    dev_samples[qid]['positive'].update(rel_docs)
    dev_samples[qid]['negative'].update(comments[comments['post_id']!=qid].sample(20)['text'].to_list())


# training file
print('making train triplets')
train_triplets = []
for qid in train_queries['0'].unique():
    neg_samples = train_queries[train_queries['0']!=qid].sample(len(train_queries[train_queries['0']==qid]))['2'].to_list()
    for qid, pos_id, neg_id in zip(train_queries[train_queries['0']==qid]['0'].to_list(), train_queries[train_queries['0']==qid]['2'].to_list(), neg_samples):
        train_triplets.append((qid, pos_id, neg_id))


cnt = 0
for qid, pos_id, neg_id in train_triplets:
    query = queries[qid]
    if (cnt % (pos_neg_ration+1)) == 0:
        passage = corpus[pos_id]
        label = 1
    else:
        passage = corpus[neg_id]
        label = 0
    train_samples.append(InputExample(texts=[query, passage], label=label))
    cnt = cnt + 1

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
# We create a DataLoader to load our train samples
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# We add an evaluator, which evaluates the performance during training
# It performs a classification task and measures scores like F1 (finding relevant passages) and Average Precision
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

# Configure the training
warmup_steps = 5000
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
print('training cross-encoder starts')
ce_model.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=10000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=True)

print('fitting cross encoder done!')



logging.info(str(args))


# The  model we want to fine-tune
train_batch_size = args.train_batch_size          #Increasing the train batch size improves the model performance, but requires more GPU memory
model_name = args.model_name
max_passages = args.max_passages
max_seq_length = args.max_seq_length            #Max length for passages. Increasing it, requires more GPU memory


num_negs_per_system = args.num_negs_per_system  # We used different systems to mine hard negatives. Number of hard negatives to add from each system
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

model_save_path = f'output/train-{model_name.replace("/", "-")}-batch_size_{train_batch_size}-{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

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
# assert len(train_queries) == 101480
print('len ',len(train_queries))


    
logging.info("Load CrossEncoder scores dict")
# Load a dict (qid, pid) -> ce_score that maps query-ids (qid) and paragraph-ids (pid)
# to the CrossEncoder score computed by the a cross-encoder trained on cmv dataset
# with open('ce_scores.pkl', 'rb') as f:
#     ce_scores = pickle.load(f)
ce_scores = []
print("making train queries dict")
# making train_queries dict
train_queries_dict = {}
cnt = 0
for qid in train_queries['0'].to_list():
    if cnt % 100 ==0:
        print(qid)
    query = queries[qid]
    pos_id = comments[comments['post_id']==qid]['comment_id'].to_list()
    neg_id = set(comments[comments['post_id']!=qid].sample(len(pos_id))['comment_id'].to_list())
    train_queries_dict[qid] = {
        'qid': qid,
        'query':query,
        'pos':pos_id,
        'neg': neg_id
    }
    cnt = cnt +1
with open('train_queries_dict.pkl', 'wb') as f:
    pickle.dump(train_queries_dict, f)

# with open('train_queries_dict.pkl', 'rb') as f:
#     train_queries_dict = pickle.load(f)


class CMVDataset(Dataset):
    def __init__(self, queries, corpus, ce_scores):
        self.queries = queries
        self.queries_ids = list(queries.keys())
        self.corpus = corpus
        self.ce_scores = ce_scores

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

        pos_score = ce_model.predict([self.queries[qid]['query'],pos_text])
        neg_score = ce_model.predict([self.queries[qid]['query'],neg_text])

        return InputExample(texts=[query_text, pos_text, neg_text], label=pos_score-neg_score)

    def __len__(self):
        return len(self.queries)

print("defining datasets")
# For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
train_dataset = CMVDataset(queries=train_queries_dict, corpus=corpus, ce_scores=ce_scores)
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
with open('corpus_'+ str(args.model_name) + 'relevant' + '.pkl', 'wb') as f:
    pickle.dump(encoded_corpus, f)
logging.info("encoding queries")
encoded_qs = {}
all_qs = (posts['title'] + ' ' + posts['text']).to_list()
for qid, q in zip(posts['post_id'].to_list(), all_qs):
    encoded_qs[qid] = model.encode(q)

with open('queries_'+ str(args.model_name) + 'relevant' + '.pkl', 'wb') as f:
    pickle.dump(encoded_qs, f)