import "./Editor.css";

import Document from "@tiptap/extension-document";
import Paragraph from "@tiptap/extension-paragraph";
import Text from "@tiptap/extension-text";
import Underline from "@tiptap/extension-underline";
import { EditorContent, useEditor } from "@tiptap/react";
import { useRef } from "react";

type modelResp = {
  commaProbs: number[]
  grammarProbs: number[]
} | null;

function Editor() {
    const timeoutRef = useRef<number | null>(null);

    const sendToModel = async (text: string) => {
        try {
            const res = await fetch("/api/suggestions", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ "Text": text }),
            });
            const content: modelResp = await res.json()
            console.log(content)
            if (!res.ok) {
                throw new Error(`err from server, status: ${res.status}`);
            }
        } catch (err) {
            console.log(err);
        }
    };
    const editor = useEditor({
        extensions: [Document, Paragraph, Text, Underline],
        content: `
        <p style="text-decoration: underline">Шампанско и сълзи е голямата хит който слушам.</p>
      `,
        onUpdate({ editor }) {
            if (timeoutRef.current) {
                clearTimeout(timeoutRef.current);
            }

            timeoutRef.current = window.setTimeout(() => {
                sendToModel(editor.getText());
            }, 900);
        },
    });

    if (!editor) {
        return null;
    }
    return (
        <>
            <EditorContent editor={editor} />
        </>
    );
}

export default Editor;
