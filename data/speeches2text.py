import pandas as pd
from pathlib import Path


def speeches2text(speechesPath: str, corpusFolderPath):
    output_path = Path(corpusFolderPath) / "speeches"

    if output_path.exists():
        return

    sp = pd.read_parquet(speechesPath)

    speeches = (
        sp["speech"]
        .dropna()
        .str.replace(r"\n+", " ", regex=True)
        .str.strip()
    )

    with output_path.open("w", encoding="utf-8") as f:
        for speech in speeches:
            f.write(speech + "\n")
