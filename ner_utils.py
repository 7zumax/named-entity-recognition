"""
ner_utils.py: shared code for the NER notebook and the Gradio app.
Keeping it in one file means the app processes text exactly like training did.

Contents
  1. Reading CoNLL-2003 files + IOB1 -> IOB2 conversion
  2. Vocabularies (word + character) and GloVe loading
  3. The three from-scratch models: LSTM, BiLSTM, BiLSTM + CRF
  4. Helpers to turn raw text into tokens and predicted tags into entity spans
"""
import json
import os
import re
import zipfile

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD, UNK = "<PAD>", "<UNK>"          # index 0 = padding, index 1 = unknown word / char


# =====================================================================
# 1. DATA
# =====================================================================
def read_conll(path):
    """Read a CoNLL-2003 file -> list of sentences, each = (list_of_words, list_of_tags).
    Each line is 'word POS chunk NER'. A blank line ends a sentence.
    '-DOCSTART-' lines only mark where a news article starts, so we skip them."""
    sentences, words, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("-DOCSTART-"):
                if words:
                    sentences.append((words, tags))
                    words, tags = [], []
                continue
            parts = line.split()
            words.append(parts[0])
            tags.append(parts[-1])           # the NER tag is the last column
    if words:
        sentences.append((words, tags))
    return sentences


def iob1_to_iob2(tags):
    """The original CoNLL files use IOB1: an entity starts with I- and B- appears only
    when two entities of the same type touch. IOB2 (the standard today) ALWAYS starts
    an entity with B-. Rule: an I-X becomes B-X when the previous tag is not X."""
    out = []
    for i, t in enumerate(tags):
        if t.startswith("I-"):
            prev = out[i - 1] if i > 0 else "O"
            if prev == "O" or prev[2:] != t[2:]:
                t = "B-" + t[2:]
        out.append(t)
    return out


# =====================================================================
# 2. VOCABULARIES + GLOVE
# =====================================================================
def norm_word(w):
    """Word key used for the word-embedding lookup: lowercase + all digits -> 0
    ('1996' and '2001' share one vector). Capital letters are NOT lost: the
    character-level part of the model still sees the original word."""
    return re.sub(r"\d", "0", w.lower())


class Vocab:
    """Simple token <-> id mapping. Index 0 = <PAD>, 1 = <UNK>."""
    def __init__(self, tokens=()):
        self.itos = [PAD, UNK]
        self.stoi = {PAD: 0, UNK: 1}
        for t in tokens:
            self.add(t)

    def add(self, t):
        if t not in self.stoi:
            self.stoi[t] = len(self.itos)
            self.itos.append(t)

    def __len__(self):
        return len(self.itos)

    def get(self, t):
        return self.stoi.get(t, 1)

    def to_dict(self):
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d):
        v = cls()
        v.itos = d["itos"]
        v.stoi = {t: i for i, t in enumerate(v.itos)}
        return v


GLOVE_ZIP_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"


def get_glove_file(dim=100, folder="data"):
    """Return the path of glove.6B.<dim>d.txt, downloading it only if it is missing.
    The official zip is 862 MB (4 sizes inside). With 'remotezip' we download ONLY
    the one file we need (~130 MB compressed for 100d) using HTTP range requests."""
    import shutil
    name = f"glove.6B.{dim}d.txt"
    path = os.path.join(folder, name)
    expected_lines = 400_000                    # GloVe 6B has exactly 400k words
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            if sum(1 for _ in f) == expected_lines:
                return path
        print("Existing GloVe file is incomplete -> downloading again")
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".part"                        # write to a temp name, rename only when complete
    try:
        from remotezip import RemoteZip
        print(f"Downloading only {name} from the GloVe zip (partial download)...")
        with RemoteZip(GLOVE_ZIP_URL) as z, z.open(name) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
    except Exception as e:                      # fallback: full zip (slower, same result)
        print("Partial download failed:", repr(e), "-> downloading the full zip instead")
        import urllib.request
        zpath = os.path.join(folder, "glove.6B.zip")
        urllib.request.urlretrieve(GLOVE_ZIP_URL, zpath)
        with zipfile.ZipFile(zpath) as z, z.open(name) as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        os.remove(zpath)
    os.replace(tmp, path)
    return path


def load_glove(path, wanted_words):
    """Read only the GloVe vectors for words we actually use (saves memory)."""
    vectors = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, rest = line.split(" ", 1)
            if word in wanted_words:
                vectors[word] = np.asarray(rest.split(), dtype="float32")
    return vectors


# =====================================================================
# 3. MODELS (built from scratch in PyTorch)
# =====================================================================
class CharCNN(nn.Module):
    """Builds a vector for a word from its LETTERS.
    Why: a word never seen in training (OOV) still has letters, capitals, suffixes
    ('-ville', '-son', 'Inc'), so the model can still guess it is a place/person/org."""
    def __init__(self, n_chars, char_dim=30, n_filters=50, kernel=3):
        super().__init__()
        self.emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_dim, n_filters, kernel_size=kernel, padding=1)
        self.out_dim = n_filters

    def forward(self, chars):                       # chars: (batch, seq_len, word_len)
        b, s, w = chars.shape
        x = self.emb(chars.view(b * s, w))          # (b*s, w, char_dim)
        x = self.conv(x.transpose(1, 2))            # (b*s, n_filters, w)
        x = torch.relu(x).max(dim=2).values         # strongest signal per filter -> (b*s, n_filters)
        return x.view(b, s, -1)


class RNNTagger(nn.Module):
    """One class for all three from-scratch models:
        bidirectional=False, use_crf=False -> LSTM
        bidirectional=True,  use_crf=False -> BiLSTM
        bidirectional=True,  use_crf=True  -> BiLSTM + CRF
    Pipeline per word: [GloVe word vector ; char-CNN vector] -> (Bi)LSTM -> scores per tag (-> CRF)."""
    def __init__(self, n_words, n_chars, n_tags, emb_matrix=None, word_dim=100,
                 hidden=200, bidirectional=True, use_crf=False, dropout=0.5):
        super().__init__()
        self.config = dict(n_words=n_words, n_chars=n_chars, n_tags=n_tags, word_dim=word_dim,
                           hidden=hidden, bidirectional=bidirectional, use_crf=use_crf, dropout=dropout)
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        if emb_matrix is not None:
            self.word_emb.weight.data.copy_(torch.as_tensor(emb_matrix))
        self.char_cnn = CharCNN(n_chars)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(word_dim + self.char_cnn.out_dim, hidden, batch_first=True,
                            bidirectional=bidirectional)
        self.to_tags = nn.Linear(hidden * (2 if bidirectional else 1), n_tags)
        self.crf = None
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(n_tags, batch_first=True)

    def emissions(self, words, chars, lengths):
        x = torch.cat([self.word_emb(words), self.char_cnn(chars)], dim=-1)
        x = self.dropout(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=words.shape[1])
        return self.to_tags(self.dropout(out))       # (batch, seq_len, n_tags)

    def loss(self, words, chars, lengths, tags):
        em = self.emissions(words, chars, lengths)
        mask = words != 0
        if self.crf is not None:
            return -self.crf(em, tags, mask=mask, reduction="mean")   # negative log-likelihood of the whole tag sequence
        return nn.functional.cross_entropy(em[mask], tags[mask])     # independent decision per word

    @torch.no_grad()
    def predict(self, words, chars, lengths):
        """Returns a list of tag-id lists (one per sentence, true length)."""
        em = self.emissions(words, chars, lengths)
        mask = words != 0
        if self.crf is not None:
            return self.crf.decode(em, mask=mask)                 # Viterbi: best VALID sequence
        best = em.argmax(-1).cpu().tolist()
        return [p[:l] for p, l in zip(best, lengths.tolist())]


# ---- Turning sentences into padded tensors ----
MAX_WORD_LEN = 20


def encode_batch(sentences, word_vocab, char_vocab, device="cpu"):
    """sentences: list of word lists -> (word_ids, char_ids, lengths) tensors, zero-padded."""
    lengths = [len(s) for s in sentences]
    L = max(lengths)
    words = torch.zeros(len(sentences), L, dtype=torch.long)
    chars = torch.zeros(len(sentences), L, MAX_WORD_LEN, dtype=torch.long)
    for i, sent in enumerate(sentences):
        for j, w in enumerate(sent):
            words[i, j] = word_vocab.get(norm_word(w))
            for k, c in enumerate(w[:MAX_WORD_LEN]):
                chars[i, j, k] = char_vocab.get(c)
    return words.to(device), chars.to(device), torch.tensor(lengths).to(device)


def save_rnn_model(model, word_vocab, char_vocab, tag_list, folder):
    os.makedirs(folder, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(folder, "weights.pt"))
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"config": model.config, "words": word_vocab.to_dict(),
                   "chars": char_vocab.to_dict(), "tags": tag_list}, f)


def load_rnn_model(folder, device="cpu"):
    with open(os.path.join(folder, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    model = RNNTagger(**meta["config"])
    model.load_state_dict(torch.load(os.path.join(folder, "weights.pt"), map_location=device))
    return (model.to(device).eval(), Vocab.from_dict(meta["words"]),
            Vocab.from_dict(meta["chars"]), meta["tags"])


# =====================================================================
# 4. RAW TEXT <-> ENTITY SPANS
# =====================================================================
TOKEN_RE = re.compile(r"(?:[A-Za-z]\.){2,}|\w+(?:[-'.]\w+)*|[^\w\s]")


def tokenize_with_offsets(text):
    """Split raw text into CoNLL-style tokens ('U.S.', "O'Neil", ',' ...) and remember
    where each token sits in the original text, so we can highlight it later."""
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]


def tags_to_spans(tags):
    """IOB2 tags -> list of (entity_type, first_token, last_token_exclusive)."""
    spans, start, etype = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != etype):
            if start is not None:
                spans.append((etype, start, i))
                start, etype = None, None
            if t != "O":                       # B-X, or an I-X that starts a new entity
                start, etype = i, t[2:]
    return spans


def highlight_segments(text, tokens, tags):
    """Build the (text_piece, label_or_None) list that gr.HighlightedText expects,
    keeping the user's original spacing."""
    segments, pos = [], 0
    for etype, s, e in tags_to_spans(tags):
        a, b = tokens[s][1], tokens[e - 1][2]
        if a > pos:
            segments.append((text[pos:a], None))
        segments.append((text[a:b], etype))
        pos = b
    if pos < len(text):
        segments.append((text[pos:], None))
    return segments


ENTITY_COLORS = {"PER": "#ff9f9f", "ORG": "#9fc5ff", "LOC": "#9fe0a8", "MISC": "#ffd27f"}


def spans_to_html(words, tags):
    """Color-coded HTML for the notebook (same colors as the app)."""
    html, i = [], 0
    spans = {s: (etype, e) for etype, s, e in tags_to_spans(tags)}
    while i < len(words):
        if i in spans:
            etype, e = spans[i]
            html.append(f'<span style="background:{ENTITY_COLORS[etype]};padding:2px 4px;'
                        f'border-radius:4px;margin:0 1px">{" ".join(words[i:e])} '
                        f'<b style="font-size:0.7em">{etype}</b></span>')
            i = e
        else:
            html.append(words[i])
            i += 1
    return '<div style="line-height:2.2;font-size:15px">' + " ".join(html) + "</div>"
