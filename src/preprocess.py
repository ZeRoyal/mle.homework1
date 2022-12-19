import configparser
import os
import pandas as pd
import sys
import traceback
import pickle

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation,TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from logger import Logger

import utils

SHOW_LOG = True
RANDOM_STATE = 26
TEST_SIZE = 0.2

class DataMaker():

    def __init__(self) -> None:
        """
        paths to the data, data constructor init
        """

        logger = Logger(SHOW_LOG)
        dataset_name = 'reviews'
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.project_path = os.path.join(os.getcwd(), "data")

        self.train_directory = os.path.join(self.project_path, "train")
        self.test_directory = os.path.join(self.project_path, "test")
        
        self.data_path = os.path.join(self.project_path, f"Video_Games_5.json")
        self.X_tfidf_path = os.path.join(self.project_path, f"{dataset_name}_X_tfidf.pkl")
        self.y_path = os.path.join(self.project_path, f"{dataset_name}_y.csv")
        
        self.Tf_Idf_m_path = os.path.join(self.project_path, f"Tf_Idf_m.pkl")
        self.train_path = [
            os.path.join(self.train_directory, f"{dataset_name}_X.pkl"),
            os.path.join(self.train_directory, f"{dataset_name}_y.csv")
        ]
        self.test_path = [
            os.path.join(self.test_directory, f"{dataset_name}_X.pkl"),
            os.path.join(self.test_directory, f"{dataset_name}_y.csv")
        ]
        
        self.log.info("DataMaker is ready")

    def get_data(self, utest=False) -> bool:
        """
        Reading dataset and getting tf-idf form for data
        """

        data = pd.read_json(self.data_path, lines = True)
        
        cols_of_interest = ['overall','reviewText', 'summary']

        data_cleaned = data[cols_of_interest]

        data_cleaned = data_cleaned.dropna()

        data = self.get_reviews(data_cleaned, 10000) if not utest else self.get_reviews(data_cleaned, 10)

        tfidf_m, tfidf_d = self.get_tf(data['reviewText'], use_idf=True, max_df=0.90, min_df=10)

        Y = data['target']

        with open(self.X_tfidf_path, 'wb') as f:
            pickle.dump(tfidf_d, f)
        Y.to_csv(self.y_path, index=True)

        with open(self.Tf_Idf_m_path, 'wb') as f:
            pickle.dump(tfidf_m, f)

        if os.path.isfile(self.X_tfidf_path) and os.path.isfile(self.y_path):
            self.log.info("X (tfidf form) and y data are ready")
            self.config["DATA"] = {'X_data_tfidf': self.X_tfidf_path,
                                   'y_data': self.y_path}

            return os.path.isfile(self.X_tfidf_path) and os.path.isfile(self.y_path)
        else:
            self.log.error("X (tfidf form) or y data are not ready")
            return False


    def get_tf(self, data, use_idf, max_df=1.0, min_df=1, ngram_range=(1,1)):
        """
        Getting tf representation
        """
        if use_idf:
            m = TfidfVectorizer(max_df=max_df, min_df=min_df, stop_words='english', ngram_range=ngram_range, tokenizer=utils.tokenize)
                    
        d = m.fit_transform(data)
        return m, d


    def cat_y(self, y) -> str:
        """
        Help function to convert labels
        """
        cat = ['bad','neutral','good']

        if y <= 2.:
            return cat[0]
        elif y >= 4.:
            return cat[2]
        else:
            return cat[1]

    def get_reviews(self, data, n_samples) -> pd.DataFrame:
        """
        Get review texts in usefull form
        """
        df = data.copy()
        df = df[df['reviewText'].apply(lambda x: len(x.split()) >= 45)]
        df['reviewText'] = df['reviewText'] + ' ' + df['summary']
        df['target'] = df['overall'].apply(self.cat_y)
        
        df = df.groupby('target').apply(lambda x: x.sample(n=n_samples))
        df = df.drop(['summary'], axis=1)
        return df

    def split_data(self, test_size=TEST_SIZE) -> bool:
        """
        Split dataset for train and test parts
        """
        self.get_data()
        try:
            with open(self.X_tfidf_path, 'rb') as f:
                X = pickle.load(f)
            y = pd.read_csv(self.y_path, index_col=0)
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE)
        
        if not os.path.exists(self.train_directory):
            os.mkdir(self.train_directory)

        self.save_splitted_data(X_train, self.train_path[0], isCsv=False)
        self.save_splitted_data(y_train, self.train_path[1])

        if not os.path.exists(self.test_directory):
            os.mkdir(self.test_directory)

        self.save_splitted_data(X_test, self.test_path[0], isCsv=False)
        self.save_splitted_data(y_test, self.test_path[1])

        self.config["SPLIT_DATA"] = {'X_train': self.train_path[0],
                                     'y_train': self.train_path[1],
                                     'X_test': self.test_path[0],
                                     'y_test': self.test_path[1]}

        self.config["TF_IDF"] = {'Tf_Idf_m': self.Tf_Idf_m_path}
        self.log.info("Train and test data are ready")
        with open(os.path.join(os.getcwd(), 'config.ini'), 'w') as configfile:
            self.config.write(configfile)
        return os.path.isfile(self.train_path[0]) and\
            os.path.isfile(self.train_path[1]) and\
            os.path.isfile(self.test_path[0]) and \
            os.path.isfile(self.test_path[1])

    def save_splitted_data(self, df, path, isCsv=True) -> bool:
        """
        Saving splitted data to csv and pickle files
        """
        if isCsv:
            df = df.reset_index(drop=True)
            df.to_csv(path, index=True)
            self.log.info(f'{path} is saved')
        else:
            with open(path, 'wb') as f:
                pickle.dump(df, f)
            self.log.info(f'{path} is saved')            
        return os.path.isfile(path)


if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()
