import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ✅ Fix: ensure MODEL_PATH points to the folder containing tokenizer files
MODEL_PATH = "bert_saved_model"

# Load tokenizer and model from folder
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown(
    """
    <style>
    .result-box {
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
    .fake {background-color: #ffdddd; color: #b30000;}
    .real {background-color: #ddffdd; color: #006600;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📰 Fake News Detector (BERT)")
st.write("Paste text below or upload a file to check if it's Real or Fake.")

user_input = st.text_area("✍️ Enter News Text:", height=150)

if st.button("🔍 Analyze Text"):
    if user_input.strip():
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        if prediction == 1:
            st.markdown('<div class="result-box fake">🚨 FAKE NEWS DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box real">✅ This looks REAL</div>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please enter some text first.")

st.subheader("📂 Bulk Prediction from File")
uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

if uploaded_file:
    results_df = None
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        if "text" in df.columns:
            preds = []
            for text in df["text"]:
                inputs = tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    preds.append(torch.argmax(outputs.logits, dim=1).item())
            df["Prediction"] = ["Fake" if p == 1 else "Real" for p in preds]
            results_df = df
            st.dataframe(df)

            if "label" in df.columns:
                y_true = df["label"].tolist()
                y_pred = preds
                acc = accuracy_score(y_true, y_pred)
                prec = precision_score(y_true, y_pred)
                rec = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)

                st.markdown(f"""
                ### 📊 Model Performance
                - **Accuracy:** {acc*100:.2f}%
                - **Precision:** {prec*100:.2f}%
                - **Recall:** {rec*100:.2f}%
                - **F1-score:** {f1*100:.2f}%
                """)

                cm_df = pd.DataFrame(
                    cm,
                    index=["True Real", "True Fake"],
                    columns=["Pred Real", "Pred Fake"]
                )
                st.write("### Confusion Matrix")
                st.dataframe(cm_df)
        else:
            st.error("CSV must have a column named 'text'.")
    else:
        lines = uploaded_file.read().decode("utf-8").splitlines()
        results = []
        for line in lines:
            inputs = tokenizer(line, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
                results.append({"Text": line, "Prediction": "Fake" if prediction == 1 else "Real"})
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)

    if results_df is not None:
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

