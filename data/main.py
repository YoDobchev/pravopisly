from dotenv import load_dotenv
import os

from speeches2text import speeches2text
from corpus import append_corpus_data
from spelling_dataset import append_spelling_data


load_dotenv()

MAX_COMMA_ROWS = 500_000
MAX_SPELLING_ROWS = 500_000
MAX_GRAMMAR_ROWS = 300_000

if __name__ == "__main__":
    DATASET_PATH = "dataset.jsonl"

    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)

    CORPUSPATH = os.getenv("CORPUSPATH")
    SPEECHESPATH = os.getenv("SPEECHESPATH")
    LEMMAPATH = os.getenv("LEMMAPATH")
    FREQLISTPATH = os.getenv("FREQLISTPATH")
    WORDCORRECTIONCSVPATH = os.getenv("WORDCORRECTIONCSVPATH")

    assert CORPUSPATH != None
    assert LEMMAPATH != None
    assert SPEECHESPATH != None
    assert FREQLISTPATH != None

    speeches2text(SPEECHESPATH, CORPUSPATH)

    append_corpus_data(
        CORPUSPATH,
        LEMMAPATH,
        FREQLISTPATH,
        WORDCORRECTIONCSVPATH,
        output_path=DATASET_PATH,
        max_comma_rows=MAX_COMMA_ROWS,
        max_spelling_rows=MAX_SPELLING_ROWS,
        max_grammar_rows=MAX_GRAMMAR_ROWS,
    )

    SPELLINGPATH = os.getenv("SPELLINGPATH")
    assert SPELLINGPATH != None

    append_spelling_data(
        SPELLINGPATH,
        output_path=DATASET_PATH,
    )
