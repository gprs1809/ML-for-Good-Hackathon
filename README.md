Our final codes (a notebook and a .py file) and outputs are in the folder Team 12 submission folder. 
We have described all the details, like our throught process, approaches, conclusions etc in the notebook 'Final_ML_for_good.ipynb' which can be found within the folder Final_Notebook_and_codes, which is within Team 12 submission folder. <br>

The IDEs we used are mainly Google Colab and VS.<br>

There are three 4 main functions: doc_preprocessing, extractive_text_summarization, preprocess_prolificacademic_and_feature_importances and sentiment_scores. Apart from these, there are loops and steps to execute and save the outputs. For more details, please see 'Final_ML_for_good.ipynb'<br>

The following packages need to be installed and imported (also can be seen in our codes).<br>

import pandas as pd <br>
import os <br>
import textract <br>
import re <br>
import gensim<br>
from gensim.summarization.summarizer import summarize <br>
from gensim.summarization import keywords <br>
from rake_nltk import Rake <br>
import nltk <br>
from sentence_transformers import SentenceTransformer <br>
#sbert_model = SentenceTransformer('bert-base-nli-mean-tokens') <br>
import tensorflow as tf <br>
import tensorflow_hub <br>
from sklearn.decomposition import PCA <br>
from sklearn.preprocessing import StandardScaler <br>
from sklearn.ensemble import RandomForestClassifier <br>
from summarizer import Summarizer <br>
import docx <br>
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer <br>

After this, the following downloads are necessary (also can be seen in our codes):<br>
nltk.download('stopwords') <br>
nltk.download('punkt') <br>
model = Summarizer() <br>
uni_encoder = tensorflow_hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',trainable=False)
