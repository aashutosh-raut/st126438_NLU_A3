from flask import Flask, render_template, request
import torch
import sentencepiece as spm

from model import Encoder, Decoder, Seq2Seq

# --------------------------------------------------
# App setup
# --------------------------------------------------
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# SentencePiece
# --------------------------------------------------
SRC_SP = spm.SentencePieceProcessor(model_file="../spm/sp_en.model")
TGT_SP = spm.SentencePieceProcessor(model_file="../spm/sp_ne.model")

SRC_PAD_IDX = SRC_SP.pad_id()
BOS_IDX = TGT_SP.bos_id()
EOS_IDX = TGT_SP.eos_id()

# --------------------------------------------------
# Hyperparameters (MUST match training)
# --------------------------------------------------
EMB_DIM = 256
ENC_HID = 512
DEC_HID = 512

SRC_VOCAB = SRC_SP.get_piece_size()
TGT_VOCAB = TGT_SP.get_piece_size()

BOS_ID = TGT_SP.bos_id()
EOS_ID = TGT_SP.eos_id()

# --------------------------------------------------
# Build model
# --------------------------------------------------
encoder = Encoder(
    vocab_size=SRC_VOCAB,
    emb_dim=EMB_DIM,
    hid_dim=ENC_HID,
    pad_idx=SRC_PAD_IDX
)

decoder = Decoder(
    vocab_size=TGT_VOCAB,
    emb_dim=EMB_DIM,
    enc_hid_dim=ENC_HID,
    dec_hid_dim=DEC_HID,
    pad_idx=TGT_SP.pad_id()
)

model = Seq2Seq(encoder, decoder, SRC_PAD_IDX).to(device)

# --------------------------------------------------
# Load trained weights
# --------------------------------------------------
state_dict = torch.load("../models/mt_dot.pt", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# --------------------------------------------------
# Translation helper
# --------------------------------------------------
def remove_repeats(tokens):
    out = []
    for t in tokens:
        if len(out) == 0 or out[-1] != t:
            out.append(t)
    return out

# def translate(text, max_len=50):
#     text = text.lower()
#     src_ids = SRC_SP.encode(text, out_type=int)
#     src_len = torch.tensor([len(src_ids)])
#     src = torch.tensor(src_ids).unsqueeze(0).to(device)
#     out_ids = remove_repeats(out_ids) 

#     out_ids = model.translate(src, src_len, BOS_IDX, EOS_IDX, max_len)
#     return TGT_SP.decode(out_ids)


def translate(text):
    text = text.lower()

    src_ids = SRC_SP.encode(text, out_type=int)
    src = torch.tensor(src_ids).unsqueeze(0).to(device)
    src_len = torch.tensor([len(src_ids)]).to(device)

    with torch.no_grad():
        out_ids = model.translate(
            src,
            src_len,
            BOS_ID,
            EOS_ID,
            max_len=64
        )

    out_ids = remove_repeats(out_ids)
    return TGT_SP.decode(out_ids)


# --------------------------------------------------
# Routes
# # --------------------------------------------------
# @app.route("/", methods=["GET", "POST"])
# def index():
#     translation = ""
#     if request.method == "POST":
#         translation = translate(request.form["text"])
#     return render_template("index.html", translation=translation)

# # --------------------------------------------------
# # Run
# # --------------------------------------------------
# if __name__ == "__main__":
#     app.run(debug=True)

@app.route("/", methods=["GET", "POST"])
def index():
    translation = ""
    src_text = ""

    if request.method == "POST":
        src_text = request.form["text"]
        translation = translate(src_text)

    return render_template(
        "index.html",
        translation=translation,
        src_text=src_text
    )

if __name__ == "__main__":
    app.run(debug=True)
