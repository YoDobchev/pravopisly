from dotenv import load_dotenv
import os

from text import append_text_data
from spelling import append_spelling_data


load_dotenv()

if __name__ == "__main__":
    DATASET_PATH = "dataset.jsonl"

    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
    TEXTSPATH = os.getenv("TEXTSPATH")
    LEMMAPATH = os.getenv("LEMMAPATH")
    assert TEXTSPATH != None and LEMMAPATH != None
    append_text_data(TEXTSPATH, LEMMAPATH)

    SPELLINGPATH = os.getenv("SPELLINGPATH")
    assert SPELLINGPATH != None
    append_spelling_data(SPELLINGPATH)
