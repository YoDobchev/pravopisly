import { Extension } from "@tiptap/core";
import type { Node as ProseMirrorNode } from "@tiptap/pm/model";
import { Plugin, PluginKey } from "@tiptap/pm/state";
import { Decoration, DecorationSet } from "@tiptap/pm/view";
import "./SuggestionHighlight.css";

export type Suggestion = {
    type: number;
    start_index: number;
    end_index: number;
    replacements: string[];
};

export type SuggestionHighlightMeta = {
    suggestions: Suggestion[];
    activeIndex: number | null;
};

export const suggestionHighlightKey = new PluginKey<DecorationSet>(
    "suggestionHighlight",
);

export function textIndexToDocPos(doc: ProseMirrorNode, index: number) {
    let current = 0;
    let result: number | null = null;

    doc.descendants((node, pos) => {
        if (!node.isText || result !== null) {
            return true;
        }

        const text = node.text ?? "";
        const next = current + text.length;

        if (index <= next) {
            result = pos + (index - current);
            return false;
        }

        current = next;
        return true;
    });

    return result ?? doc.content.size;
}

function buildDecorations(
    doc: ProseMirrorNode,
    suggestions: Suggestion[],
    activeIndex: number | null,
) {
    const decorations = suggestions.map((s, index) => {
        const from = textIndexToDocPos(doc, s.start_index);
        const to = textIndexToDocPos(doc, s.end_index);

        const className = [
            "suggestion",
            `suggestion-${s.type}`,
            activeIndex === index ? "suggestion-active" : "",
        ]
            .filter(Boolean)
            .join(" ");

        if (s.start_index === s.end_index) {
            const el = document.createElement("span");
            el.className = className;
            el.textContent = s.replacements[0] || "";

            return Decoration.widget(from, el);
        }

        return Decoration.inline(from, to, {
            class: className,
        });
    });

    return DecorationSet.create(doc, decorations);
}

export const SuggestionHighlight = Extension.create({
    name: "suggestionHighlight",

    addProseMirrorPlugins() {
        return [
            new Plugin<DecorationSet>({
                key: suggestionHighlightKey,

                state: {
                    init() {
                        return DecorationSet.empty;
                    },

                    apply(tr, old) {
                        const meta = tr.getMeta(suggestionHighlightKey) as
                            | SuggestionHighlightMeta
                            | undefined;

                        if (meta) {
                            return buildDecorations(
                                tr.doc,
                                meta.suggestions,
                                meta.activeIndex,
                            );
                        }

                        return old.map(tr.mapping, tr.doc);
                    },
                },

                props: {
                    decorations(state) {
                        return this.getState(state);
                    },
                },
            }),
        ];
    },
});
