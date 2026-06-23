import re

import pb.pravopisly_pb2 as pravopisly_pb2
from symspellpy import Verbosity


WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁёЍѝ]+(?:-[A-Za-zА-Яа-яЁёЍѝ]+)?")


def preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()

    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]

    return replacement


def append_spelling_suggestions(
    sym_spell,
    reranker,
    suggestions: list,
    original_text: str,
    spelling_probs,
    min_confidence: float = 0.85,
):
    words = list(WORD_RE.finditer(original_text))

    for i, match in enumerate(words):
        if i >= len(spelling_probs):
            break

        if float(spelling_probs[i]) < min_confidence:
            continue

        original_word = match.group()
        word_lower = original_word.lower()

        if len(word_lower) < 3:
            continue

        symspell_results = sym_spell.lookup(
            word_lower,
            Verbosity.ALL,
            max_edit_distance=2,
            include_unknown=False,
        )

        candidates = []
        candidate_infos = []
        candidate_counts = {}

        for result in symspell_results:
            candidate = result.term

            if candidate == word_lower:
                continue

            candidate_counts[candidate] = result.count

            candidate = preserve_case(original_word, candidate)

            if candidate not in candidates:
                candidates.append(candidate)
                candidate_infos.append((
                    candidate,
                    result.distance,
                    result.count,
                ))

        if not candidates:
            continue

        candidate_infos.sort(key=lambda x: (x[1], -x[2]))

        candidate_words = [
            candidate
            for candidate, _, _ in candidate_infos[:7]
        ]

        print(candidate_words)

        ranked_candidates = reranker.rerank(
            original_text=original_text,
            start_index=match.start(),
            end_index=match.end(),
            candidates=candidate_words,
            original_word=word_lower,
            candidate_counts=candidate_counts,
        )

        replacements = [candidate for candidate, _ in ranked_candidates[:3]]

        if not replacements:
            continue

        suggestions.append(
            pravopisly_pb2.TextSuggestion(
                type=pravopisly_pb2.SPELLING,
                start_index=match.start(),
                end_index=match.end(),
                replacements=replacements,
            )
        )
