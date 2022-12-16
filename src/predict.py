import argparse
import configparser
from datetime import datetime
import os
import json
import pickle
import shutil
import sys
import time
import traceback
import yaml

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

from logger import Logger

SHOW_LOG = True

class Predictor():

    def __init__(self) -> None:

        logger = Logger(SHOW_LOG)
        self.cat = ['bad', 'neutral', 'good']
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), 'config.ini')
        self.config.read(self.config_path)

        self.parser = argparse.ArgumentParser(description="Predictor")
        self.parser.add_argument("-t",
                                 "--tests",
                                 type=str,
                                 help="Select tests",
                                 required=True,
                                 default="smoke",
                                 const="smoke",
                                 nargs="?",
                                 choices=["smoke", "func"])

        with open(self.config["SPLIT_DATA"]["X_test"], 'rb') as f:
            self.X_test = pickle.load(f)

        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        self.y_test = self.y_test['target.1']

        with open(self.config["TF_IDF"]["Tf_Idf_m"], 'rb') as f:
            self.tfidf_m = pickle.load(f)


        try:
            with open(self.config["MODEL"]["path"], "rb") as f:
                self.model = pickle.load(f)
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        
        self.log.info("Predictor is ready")


    def predict(self) -> bool:

        args = self.parser.parse_args()
        
        if args.tests == "smoke":
            print('smoke!')
            try:
                for m in range(0,3):
                    y_adj = np.array(self.y_test == self.cat[m])
                    print('Model ({M}): {P:.1%}'.format(M=self.cat[m], P=cross_val_score(self.model[m], self.X_test, y_adj, cv=10, scoring='accuracy').mean() ))
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)
            self.log.info(
                f'{self.config["MODEL"]["path"]} passed smoke tests')

        elif args.tests == "func":

            tests_path = os.path.join(os.getcwd(), "tests")
            exp_path = os.path.join(os.getcwd(), "experiments")
            for test in os.listdir(tests_path):
                with open(os.path.join(tests_path, test)) as f:

                    try:
                        data = json.load(f)
                        X = [x['review'] for x in data['X']]
                        y = [y['target'] for y in data['y']]
                        print(X)
                        for text_ in X:
                            #y_adj = np.array(y == self.cat[m])
                            self.get_review_score(text_)
                    except Exception:
                        self.log.error(traceback.format_exc())
                        sys.exit(1)

                    self.log.info(
                        f'{self.config["MODEL"]["path"]} passed func test {f.name}')
                    exp_data = {
                        "model params": dict(self.config.items("MODEL")),
                        "tests": args.tests,
                        "X_test path": self.config["SPLIT_DATA"]["x_test"],
                        "y_test path": self.config["SPLIT_DATA"]["y_test"],
                    }

                    date_time = datetime.fromtimestamp(time.time())
                    str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                    
                    os.mkdir(exp_dir)
                    with open(os.path.join(exp_dir,"exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                    
                    shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir, "exp_logfile.log"))
                    shutil.copy(self.config["MODEL"]["path"], os.path.join(exp_dir, f'exp.sav'))
        
        return True
    
    def get_review_score(self, text: list) -> list:
        """
        interactive review scoring

        """
        test_str = [text]
        test_new = self.tfidf_m.transform(test_str)

        print('Review text: "{R}"\n'.format(R=test_str[0]))
        print('Model Predction')
        for m in range(0, 3):
            print('Model ({M}): {P:.1%}'.format(M=self.cat[m], P=self.model[m].predict_proba(test_new)[0][1]))
        #return self.classifier.predict(text)


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()