from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer


def tokenize(text):
    """
    tokenization for EN text ( for train/test process )
    """
    stemmer = SnowballStemmer("english")
    tokenizer = RegexpTokenizer("[a-z']+")
    tokens = tokenizer.tokenize(text)
    return [stemmer.stem(t) for t in tokens] 