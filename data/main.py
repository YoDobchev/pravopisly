from dotenv import load_dotenv
import os

from wikipedia import append_wikipedia_sentences
from spelling import append_spelling_data


load_dotenv()

if __name__ == "__main__":
    DATASET_PATH = "dataset.jsonl"

    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
    WIKIPATH = os.getenv("WIKIPATH")
    LEMMAPATH = os.getenv("LEMMAPATH")
    assert WIKIPATH != None and LEMMAPATH != None
    append_wikipedia_sentences(WIKIPATH, LEMMAPATH)

    SPELLINGPATH = os.getenv("SPELLINGPATH")
    assert SPELLINGPATH != None
    append_spelling_data(SPELLINGPATH)
