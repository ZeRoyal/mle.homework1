import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from predict import Predictor

config = configparser.ConfigParser()
config.read("config.ini")


class TestPredict(unittest.TestCase):

    def setUp(self) -> None:
        self.model = Predictor()

    # def test_predict(self):
    #     self.assertEqual(self.model.predict(), True)

    def test_get_review_score(self):
        self.assertEqual(self.model.get_review_score('This was very bad product!'), True)

if __name__ == "__main__":
    unittest.main()