# Don’t Raise Your Voice, Improve Your Argument: Learning to Retrieve Convincing Arguments
We introduce CoRe dataset to enable researchers to explore concepts of convincingness and relevance in information retrieval tasks. We leveraged CMV community on reddit where people express their opinions and hear others' viewpoints in hope of changing beliefs or adopting new ones. We collected 7937 posts along with their subsequent arguments (comments). We have studied the concept of convincingness and trained neural rankers on both relevance and convincingness concepts to rank arguments based on both criteria seperately. In conclusion, we found that these two concepts are closely correlated. The more relevant an argument is, the more convincing  it is too.
## Dataset
We collected posts, comments and users' information from CMV subreddit. We present our dataset in three parts:

 1. Posts: opinions posted on CMV subreddit
 2. Comments: arguments under posts
 3. Users: Information of users who wrote a comment or post

The code for downloading data from reddit can be found in `./Crawler/` directory. You need to enter `client_id` and `client_secret` that can be found in you reddit account and start using the code for downloading posts, comments and users from CMV for your specified date. 

For information retrieval tasks, we use posts as queries and comments as corpus. We have downloaded all firs-level comments for each post along with other information. For each part of the dataset, we collected the following information:
| Data        | Fields           | 
| ------------- |:-------------:|
| Posts      | Post_id, title, text, user_id, score, time, post_flair |
| Comments      | comments_id, text, post_id, #delta, convincingness_level|
| Users |user_id, comment_karma, link_karma, registration_date |

Delta scoring system is a system that CMV subreddit use to acknowledge change of opinions. When a comment convinces someone to change at least one aspect of their opinion, a delta should be awarded by the user to the comment. We have leveraged these deltas and mapped them into 5 levels. the conv_level field in the comments dataset shows these levels of convincingness. 
These information can be used for further research on convincingness and information retrieval. 

We have splitted queries and documents to train/dev/test sets and built qrels files. You can download the dataset [here](https://drive.google.com/drive/folders/1EHHNFQpERm4TNrHIce5yVxuiKHajxB44?usp=sharing).
## Usage
We have trained our models on both relevance and convincingness notions. Using dense retrievers, we trained base models such that first, they learn how to retrieve relevant documents (all the first-level documents of a post are considered relevant). Second, they learn how to retrieve convincing documents (comments with more deltas are considered more convincing) and third, we use models trained on ms-marco (relevance notion) and train them on our dataset for convincingness notion. 
To train models on each concepts, use the following code with the name of the base model you want to train from Huggingface models:

**Convincingness:**

    python train_convincing.py --model_name model_name
**Relevance:**

    python train_relevant.py --model_name model_name --ce_model_name ce_model_name
Training files train a model for creating encodings and save encodings for queries and corpus in the encoding directory. 
Use the following commands to index and search within the corpus. The following python scripts generate search result files in Trec format for test queries. Use the name of the base model you trained to load the related encodings.

**Convincingness:**

    python convincing_ranking.py --model_name trained_model_name 
**Relevance:**

    python relevant_ranking.py --model_name trained_model_name
   

