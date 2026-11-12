"""
app.py: Gradio web app for Named Entity Recognition (CoNLL-2003).
Loads the best model chosen in the notebook (models/results.json), takes free text
and highlights the detected entities in real time: PER, ORG, LOC, MISC.
Run:  python app.py  -> opens http://127.0.0.1:7860
"""
import json
import os

import gradio as gr
import torch

import ner_utils as nu

with open("models/results.json") as f:
    info = json.load(f)
# DistilBERT weights (265 MB) are too big for GitHub. If they are missing,
# use the best of the from-scratch models that are in the repo.
ranked = sorted(info["results"], key=lambda m: info["results"][m]["f1"], reverse=True)
BEST = next(m for m in ranked if os.path.exists(
    "models/DistilBERT/config.json" if m == "DistilBERT" else f"models/{m}/weights.pt"))
F1 = info["results"][BEST]["f1"]
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Loading best model:", BEST)

if BEST == "DistilBERT":
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    hf_tok = AutoTokenizer.from_pretrained("models/DistilBERT")
    hf_model = AutoModelForTokenClassification.from_pretrained("models/DistilBERT").to(device).eval()
    id2tag = hf_model.config.id2label

    def predict(words):
        enc = hf_tok(words, is_split_into_words=True, truncation=True, max_length=512,
                     return_tensors="pt").to(device)
        with torch.no_grad():
            pred = hf_model(**enc).logits.argmax(-1)[0].tolist()
        tags, seen = ["O"] * len(words), set()
        for pos, wid in enumerate(enc.word_ids(0)):
            if wid is not None and wid not in seen:          # first sub-word decides the word's tag
                seen.add(wid)
                tags[wid] = id2tag[pred[pos]]
        return tags
else:
    model, word_vocab, char_vocab, TAGS = nu.load_rnn_model(f"models/{BEST.replace(' ', '')}", device)

    def predict(words):
        w, c, l = nu.encode_batch([words], word_vocab, char_vocab, device)
        return [TAGS[i] for i in model.predict(w, c, l)[0]]


def tag_text(text):
    if not text or not text.strip():
        return [], ""
    tokens = nu.tokenize_with_offsets(text)
    if not tokens:
        return [(text, None)], ""
    tags = predict([t for t, _, _ in tokens])
    spans = nu.tags_to_spans(tags)
    table = "\n".join(f"| {text[tokens[s][1]:tokens[e - 1][2]]} | {etype} |" for etype, s, e in spans)
    summary = ("| Entity | Type |\n|---|---|\n" + table) if spans else "_No entities found._"
    return nu.highlight_segments(text, tokens, tags), summary


with gr.Blocks(title="Named Entity Recognition") as demo:
    gr.Markdown(f"# Named Entity Recognition\nFinds **people (PER)**, **organizations (ORG)**, "
                f"**locations (LOC)** and **other names (MISC)** as you type.  \n"
                f"Model: **{BEST}**, test entity F1 **{F1:.3f}** on CoNLL-2003")
    inp = gr.Textbox(lines=5, label="Your text", placeholder="Type or paste English text...")
    out = gr.HighlightedText(label="Detected entities", color_map=nu.ENTITY_COLORS,
                             show_legend=True, combine_adjacent=False)
    table = gr.Markdown()
    inp.change(tag_text, inp, [out, table])               # updates on every keystroke = real time
    gr.Examples([
        ["Lionel Messi signed a new contract with Inter Miami in Florida on Tuesday."],
        ["The European Union and Germany criticised Microsoft over its Windows licensing in Brussels."],
        ["Egyptian striker Mohamed Salah scored twice as Liverpool beat Arsenal at Anfield."],
        ["Elon Musk said Tesla will open a new factory near Cairo, according to Reuters."],
    ], inputs=inp, outputs=[out, table], fn=tag_text, run_on_click=True)

if __name__ == "__main__":
    demo.launch(inbrowser=True)
