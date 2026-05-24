from dotenv import load_dotenv
import os

from speeches2text import speeches2text
from text import append_text_data
from spelling import append_spelling_data


load_dotenv()

if __name__ == "__main__":
    DATASET_PATH = "dataset.jsonl"

    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
    TEXTSPATH = os.getenv("TEXTSPATH")
    SPEECHESPATH = os.getenv("SPEECHESPATH")
    LEMMAPATH = os.getenv("LEMMAPATH")
    assert TEXTSPATH != None and LEMMAPATH != None and SPEECHESPATH != None
    speeches2text(SPEECHESPATH, TEXTSPATH)
    append_text_data(TEXTSPATH, LEMMAPATH)

    SPELLINGPATH = os.getenv("SPELLINGPATH")
    assert SPELLINGPATH != None
    append_spelling_data(SPELLINGPATH)
