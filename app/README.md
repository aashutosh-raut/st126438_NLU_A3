# Machine Translation Web App

A simple Streamlit app to demo English → Nepali translation using the Seq2Seq + Attention model trained in the notebook.

## Prerequisites
- Train the models in `mt_nepali_english_attention.ipynb` and ensure the following files exist in the project root:
  - `sp_en.model`, `sp_ne.model` (SentencePiece models)
  - `mt_dot.pt` or `mt_additive.pt` (trained checkpoints)

## Install
```bash
pip install streamlit torch sentencepiece
```

## Run
```bash
streamlit run app/app.py
```

Use the dropdown to select the attention type and load the corresponding checkpoint. Enter an English sentence and click Translate to see the Nepali output.
