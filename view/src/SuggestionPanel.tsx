import type { Suggestion } from "./SuggestionHighlight";
import "./SuggestionPanel.css";

type Props = {
    suggestions: Suggestion[];
    text: string;
    activeIndex: number | null;
    onActiveChange: (index: number | null) => void;
    onApplySuggestion: (index: number, replacement: string) => void;
};

const typeLabels: Record<number, string> = {
    1: "Запетаи",
    2: "Граматични грешки",
    3: "Правописни грешки",
};

function getOriginalText(text: string, suggestion: Suggestion) {
    if (suggestion.start_index === suggestion.end_index) {
        return "Добави";
    }

    return text.slice(suggestion.start_index, suggestion.end_index);
}

function groupSuggestions(suggestions: Suggestion[]) {
    const groups = new Map<number, { suggestion: Suggestion; index: number }[]>();

    suggestions.forEach((suggestion, index) => {
        if (!groups.has(suggestion.type)) {
            groups.set(suggestion.type, []);
        }

        groups.get(suggestion.type)!.push({ suggestion, index });
    });

    return [...groups.entries()];
}

export default function SuggestionPanel({
    suggestions,
    text,
    activeIndex,
    onActiveChange,
    onApplySuggestion,
}: Props) {
    const groups = groupSuggestions(suggestions);

    return (
        <div className="suggestion-panel">
            <h2>Предложения</h2>

            {suggestions.length === 0 && (
                <p className="suggestion-empty">Няма грешки в текста.</p>
            )}

            {groups.map(([type, items]) => (
                <section key={type} className="suggestion-group">
                    <h3>{typeLabels[type] ?? `TYPE ${type}`}</h3>
                    <hr />

                    {items.map(({ suggestion, index }) => {
                        const original = getOriginalText(text, suggestion);

                        return (
                            <div
                                key={`${suggestion.start_index}-${suggestion.end_index}-${index}`}
                                className={
                                    activeIndex === index
                                        ? "suggestion-card active"
                                        : "suggestion-card"
                                }
                                onMouseEnter={() => onActiveChange(index)}
                                onMouseLeave={() => onActiveChange(null)}
                                onFocus={() => onActiveChange(index)}
                                onBlur={() => onActiveChange(null)}
                            >
                                <div className="suggestion-row">
                                    <span className="suggestion-original">
                                        {original}
                                    </span>

                                </div>

                                {suggestion.replacements.length > 0 && (
                                    <div className="replacement-list">
                                        {suggestion.replacements.map(
                                            (replacement) => (
                                                <button
                                                    key={replacement}
                                                    type="button"
                                                    className="replacement-button"
                                                    onClick={() =>
                                                        onApplySuggestion(
                                                            index,
                                                            replacement,
                                                        )
                                                    }
                                                >
                                                    {replacement}
                                                </button>
                                            ),
                                        )}
                                    </div>
                                )}
                            </div>
                        );
                    })}
                </section>
            ))}
        </div>
    );
}
