import re


def build_replacements(lemmas_file):
    word_col = lemmas_file.columns[0]
    lemma_col = lemmas_file.columns[1]

    replacements = {}

    for _, group in lemmas_file.groupby(lemma_col):
        forms = (
            group[word_col]
            .dropna()
            .astype(str)
            .str.strip()
            .tolist()
        )

        forms = [form for form in forms if re.fullmatch(r"[А-Яа-я]+", form)]

        if len(forms) < 2:
            continue

        forms = list(dict.fromkeys(forms))

        for form in forms:
            key = form.lower()

            alternatives = [
                other for other in forms
                if other.lower() != key
            ]

            if alternatives:
                replacements.setdefault(key, []).extend(alternatives)

    for key in replacements:
        replacements[key] = list(dict.fromkeys(replacements[key]))

    return replacements