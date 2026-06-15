from dotenv import load_dotenv
import os

from speeches2text import speeches2text
from corpus import append_corpus_data
from spelling_dataset import append_spelling_data


load_dotenv()

if __name__ == "__main__":
    DATASET_PATH = "dataset.jsonl"

    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
    CORPUSPATH = os.getenv("CORPUSPATH")
    SPEECHESPATH = os.getenv("SPEECHESPATH")
    LEMMAPATH = os.getenv("LEMMAPATH")
    FREQLISTPATH = os.getenv("FREQLISTPATH")
    assert CORPUSPATH != None and LEMMAPATH != None and SPEECHESPATH != None and FREQLISTPATH != None
    speeches2text(SPEECHESPATH, CORPUSPATH)
    append_corpus_data(CORPUSPATH, LEMMAPATH, FREQLISTPATH)

    SPELLINGPATH = os.getenv("SPELLINGPATH")
    assert SPELLINGPATH != None
    append_spelling_data(SPELLINGPATH)
