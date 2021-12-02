Our final codes (a notebook and a .py file) and outputs are in the folder Team 12 submission folder. 
We have described all the details, like our throught process, approaches, conclusions etc in the notebook 'Final_ML_for_good.ipynb' which can be found within the folder Final_Notebook_and_codes, which is within Team 12 submission folder. <br>

The IDEs we used are mainly Google Colab and VS.<br>

There are three 4 main functions: doc_preprocessing, extractive_text_summarization, preprocess_prolificacademic_and_feature_importances and sentiment_scores. Apart from these, there are loops and steps to execute and save the outputs. For more details, please see 'Final_ML_for_good.ipynb'<br>

The following packages need to be installed and imported (also can be seen in our codes).<br>

import pandas as pd <br>
import os <br>
import textract
import re
import gensim
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from rake_nltk import Rake
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sentence_transformers import SentenceTransformer
#sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
import tensorflow as tf
import tensorflow_hub
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from summarizer import Summarizer
import docx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
