import re

import pb.pravopisly_pb2 as pravopisly_pb2


def append_comma_suggestions(
    suggestions: list,
    original_text: str,
    comma_probs,
    min_confidence: float = 0.8,
):
    words = list(re.finditer(r"\S+", original_text))

    for i, match in enumerate(words):
        if i >= len(comma_probs):
            break

        prob = float(comma_probs[i])
        end = match.end()

        comma_index = end if end < len(
            original_text) and original_text[end] == "," else -1

        if comma_index == -1:
            j = end
            while j < len(original_text) and original_text[j].isspace():
                j += 1
            if j < len(original_text) and original_text[j] == ",":
                comma_index = j

        if prob >= min_confidence and comma_index == -1:
            suggestions.append(pravopisly_pb2.TextSuggestion(
                type=pravopisly_pb2.COMMA,
                start_index=end,
                end_index=end,
                replacements=[","],
            ))

        elif prob < min_confidence and comma_index != -1:
            suggestions.append(pravopisly_pb2.TextSuggestion(
                type=pravopisly_pb2.COMMA,
                start_index=comma_index,
                end_index=comma_index+1,
                replacements=[""],
            ))
