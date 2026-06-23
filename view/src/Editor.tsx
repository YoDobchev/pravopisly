import "./Editor.css";

import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import Underline from "@tiptap/extension-underline";
import { EditorContent, useEditor } from "@tiptap/react";
import { useRef, useState } from "react";

import SuggestionPanel from "./SuggestionPanel";
import {
    type Suggestion,
    SuggestionHighlight,
    suggestionHighlightKey,
    textIndexToDocPos,
} from "./SuggestionHighlight";

type ModelResp = {
    suggestions: Suggestion[];
} | null;

function Editor() {
    const timeoutRef = useRef<number | null>(null);
    const requestIdRef = useRef(0);

    const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
    const [currentText, setCurrentText] = useState("");
    const [activeSuggestionIndex, setActiveSuggestionIndex] = useState<
        number | null
    >(null);

    const editor = useEditor({
        extensions: [Document, Paragraph, Text, Underline, SuggestionHighlight],
        content: `
            <p>Шампанско и сълзи е голямата хит който слушам.</p>
        `,

        onCreate({ editor }) {
            setCurrentText(editor.getText());
        },

        onUpdate({ editor }) {
            const text = editor.getText();
            const requestId = ++requestIdRef.current;

            setCurrentText(text);

            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }

            timeoutRef.current = window.setTimeout(() => {
                sendToModel(text, requestId);
            }, 900);
        },
    });

    function applySuggestionHighlights(
        nextSuggestions: Suggestion[],
        activeIndex: number | null,
    ) {
        if (!editor) {
            return;
        }

        editor.view.dispatch(
            editor.state.tr.setMeta(suggestionHighlightKey, {
                suggestions: nextSuggestions,
                activeIndex,
            }),
        );
    }

    function clearSuggestionState(editorInstance = editor) {
        if (!editorInstance) {
            return;
        }

        setSuggestions([]);
        setActiveSuggestionIndex(null);

        editorInstance.view.dispatch(
            editorInstance.state.tr.setMeta(suggestionHighlightKey, {
                suggestions: [],
                activeIndex: null,
            }),
        );
    }

    async function sendToModel(text: string, requestId: number) {
        try {
            const res = await fetch("/api/suggestions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ Text: text }),
            });

            if (!res.ok) {
                throw new Error(`err from server, status: ${res.status}`);
            }

            const content: ModelResp = await res.json();

            if (requestId !== requestIdRef.current) {
                return;
            }

            const nextSuggestions = content?.suggestions ?? [];

            setSuggestions(nextSuggestions);
            setActiveSuggestionIndex(null);
            applySuggestionHighlights(nextSuggestions, null);
        } catch (err) {
            console.log(err);
        }
    }

    function handleActiveSuggestionChange(index: number | null) {
        setActiveSuggestionIndex(index);
        applySuggestionHighlights(suggestions, index);
    }

    function handleApplySuggestion(index: number, replacement: string) {
        if (!editor) {
            return;
        }

        const suggestion = suggestions[index];

        if (!suggestion) {
            return;
        }

        const from = textIndexToDocPos(
            editor.state.doc,
            suggestion.start_index,
        );
        const to = textIndexToDocPos(editor.state.doc, suggestion.end_index);

        editor
            .chain()
            .focus()
            .insertContentAt(
                {
                    from,
                    to,
                },
                replacement,
            )
            .run();

        clearSuggestionState(editor);
        setCurrentText(editor.getText());
    }

    if (!editor) {
        return null;
    }

    return (
        <div className="editor-shell">
            <div className="editor">
                <EditorContent editor={editor} />
            </div>

            <SuggestionPanel
                suggestions={suggestions}
                text={currentText}
                activeIndex={activeSuggestionIndex}
                onActiveChange={handleActiveSuggestionChange}
                onApplySuggestion={handleApplySuggestion}
            />
        </div>
    );
}

export default Editor;
