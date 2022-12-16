import configparser
import os
import pandas as pd
import numpy as np

import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import sys
import traceback

from logger import Logger

SHOW_LOG = True


class ReviewsModel():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), 'config.ini')
        self.config.read(self.config_path)

        with open(self.config["SPLIT_DATA"]["X_train"], 'rb') as f:
            self.X_train = pickle.load(f)

        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)

        with open(self.config["SPLIT_DATA"]["X_test"], 'rb') as f:
            self.X_test = pickle.load(f)

        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        print(self.y_train)
        self.y_train = self.y_train['target.1']
        self.y_test = self.y_test['target.1']

        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.model_path = os.path.join(self.project_path, "model.sav")
        self.log.info("Reviews Classifier is ready")


    def train(self, predict=False) -> bool:
        models = []
        cat = ['bad', 'neutral', 'good']
        try:
            for c in cat:
                y_adj = np.array(self.y_train == c)
                lm = LogisticRegression()
                lm_f = lm.fit(self.X_train, y_adj)
                models.append(lm_f)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        if predict:
            y_gt = []
            print('Model Predction')
            for m in range(0,3):
                y_adj = np.array(self.y_test == cat[m])
                print('Model ({M}): {P:.1%}'.format(M=cat[m], P=cross_val_score(models[m], self.X_test, y_adj, cv=10, scoring='accuracy').mean() ))

        params = {'path': self.model_path}
        return self.save_model(models, self.model_path, "MODEL", params)


    def save_model(self, model, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        os.remove(self.config_path)
        with open(self.config_path, 'w') as f:
            self.config.write(f)

        with open(path, 'wb') as f:
            pickle.dump(model, f)

        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    model = ReviewsModel()
    model.train(predict=True)
