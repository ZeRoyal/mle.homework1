import configparser
import os
import unittest
import pandas as pd
import sys
import pickle
sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from preprocess import DataMaker

config = configparser.ConfigParser()
config.read("config.ini")


class TestDataMaker(unittest.TestCase):

    def setUp(self) -> None:
        self.data_maker = DataMaker()

    def test_get_data(self):
        self.assertEqual(self.data_maker.get_data(utest=True), True)

    def test_split_data(self):
        self.assertEqual(self.data_maker.split_data(), True)

    def test_save_splitted_data(self):
        path = config["DATA"]["X_data_tfidf"]
        with open(path, 'rb') as f:
            df = pickle.load(f)
        self.assertEqual(self.data_maker.save_splitted_data(df, path, isCsv=False), True)


if __name__ == "__main__":
    unittest.main()