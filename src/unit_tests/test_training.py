import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import ReviewsModel

config = configparser.ConfigParser()
config.read("config.ini")


class TestTrain(unittest.TestCase):

    def setUp(self) -> None:
        self.model = ReviewsModel()

    def test_train_model(self):
        self.assertEqual(self.model.train(), True)


if __name__ == "__main__":
    unittest.main()