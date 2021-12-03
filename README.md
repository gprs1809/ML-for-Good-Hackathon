This repository contains all the work that Team 12 did as an attempt to help the Child Mind Institute.<br>

Our final codes (a notebook and a .py file) and outputs are in the folder Team 12 submission folder. <br>
We have described all the details, like our throught process, approaches, difficulties faced, potential drawbacks, conclusions etc in the notebook 'Final_ML_for_good.ipynb' which can be found within the folder Final_Notebook_and_codes, which is within Team 12 submission folder. <br>

Below is a summary of what we have done, the notebook has more details:<br>

We worked on all the data provided independently, we didn't combine any data.

1) For the docx files, we implemented the following:<br>
             * doc_preprocessing function: This function takes the path of any docx file and processes it. The processing broadly involves replacing irrelevant substrings,    breaking down text into sentences, identifying the starting of any dialogue, removing not so important dialogues, combining dialogues of each speaker.<br>
             * extractive_text_summarization: This function takes the path of any docx files, processes it using the doc_preprocessing function and then, performs extractive text summarization. We have presented 4 types of extractive text summarization, 2 of which use the Gensim summarizer with ratio and word count as parameters respectively. The other two use the Bert extractive summarizer with ratio and number of sentences as parameters respectively.<br>

2) For each of the csv file under Prolific Academic (Nov 2020 and April 2021, as well as the files for these months under updated data), we implemented the following function:<br>
             * preprocess_prolificacademic_and_feature_importances: This function takes the path of the csv file, preprocesses it and then, tries to return insights in the form of feature importances. The feature importances are arrived at by fitting a Random Forest classifier where the target feature is taken as 'suspectedinfected'. Before setting the target, there are a bunch of preprocessing steps: remove features with more than 50% NaNs, find the top 19 or 20 features that are most correlated to suspectedidentified among each of int and float features, replace NaNs in the string features with 'Information not available', combine dataframes, removing rows with NaNs, set target as suspectedidentified, get embeddings for each string feature using the Universal Sentence Encoder pretrained model from tensorflow hub, perform PCA to reduce the dimension of embeddings for each string feature to 5, the new features as embeddings are given names as feature_name followed by an index.<br>

3) For the crisis logger csv file, we implemented the function sentiment_scores to get Positive, Negative and Neutral scores for each transcription. We further also return the Overall rating of each transcription with respect to scores, as Positive, Negative or Neutral. <br>



We mainly used Google Colab and Visual Studio for implementing. Python version we used is 3.8.5 when we used VS. <br>

The latest versions of the following packages need to be installed and imported or just imported (also can be seen in our codes). We used pip to install all the required packages that aren't built-in. The pip commands are also executed from the jupyter notebook, which can be seen in 'Final_ML_for_good.ipynb' <br>

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
model = Summarizer() (Bert summarizer) <br>
uni_encoder = tensorflow_hub.KerasLayer('https://tfhub.dev/google/universal-sentence-encoder/4',trainable=False) (downloading the pretrained Universal Sentence Encoder model from Tensorflow Hub.
