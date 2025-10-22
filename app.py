import streamlit as st
import os
import gdown
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

MODEL_DIR = "bert_saved_model"
FOLDER_URL = "https://drive.google.com/drive/folders/1FrOAKgTjQuxXCylSKUiGwfcjF2Hp71ZU?usp=sharing"

@st.cache_resource
def load_model():
    try:
        if not check_model_folder(MODEL_DIR):
            st.info("Downloading BERT model folder from Google Drive...")
            gdown.download_folder(url=FOLDER_URL, output=MODEL_DIR, use_cookies=False)
            st.success("Model folder downloaded successfully!")
        
        tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
        model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def check_model_folder(folder_path):
    required_files = ["pytorch_model.bin", "config.json", "vocab.txt"]
    if not os.path.exists(folder_path):
        return False
    files_present = os.listdir(folder_path)
    return all(f in files_present for f in required_files)

def predict_batch(texts, tokenizer, model, batch_size=16):
    predictions = []
    confidences = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(outputs.logits, dim=1).tolist()
            confs = torch.max(probs, dim=1).values.tolist()
            
            predictions.extend(preds)
            confidences.extend(confs)
    
    return predictions, confidences

tokenizer, model = load_model()

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
        margin: 10px 0;
    }
    .fake {background-color: #ffdddd; color: #b30000;}
    .real {background-color: #ddffdd; color: #006600;}
    .confidence {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Fake News Detector (BERT)")
st.write("Paste text below or upload a file to check if it's Real or Fake.")

user_input = st.text_area("Enter News Text:", height=150, placeholder="Paste your news article here...")

if st.button("Analyze Text", type="primary"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            try:
                inputs = tokenizer(
                    user_input, 
                    return_tensors="pt", 
                    truncation=True, 
                    padding=True, 
                    max_length=512
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    prediction = torch.argmax(outputs.logits, dim=1).item()
                    confidence = torch.max(probs).item()
                
                st.write(f"Debug - Raw prediction: {prediction}, Logits: {outputs.logits.tolist()}, Probs: {probs.tolist()}")
                
                if prediction == 0:
                    st.markdown(
                        f'<div class="result-box fake">FAKE NEWS DETECTED<br>'
                        f'<span class="confidence">Confidence: {confidence*100:.1f}%</span></div>', 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="result-box real">This looks REAL<br>'
                        f'<span class="confidence">Confidence: {confidence*100:.1f}%</span></div>', 
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.warning("Please enter some text first.")

st.divider()
st.subheader("Bulk Prediction from File")
st.caption("Upload a CSV file with a 'text' column, or a TXT file with one article per line")

uploaded_file = st.file_uploader("Upload a CSV or TXT file", type=["csv", "txt"])

if uploaded_file:
    try:
        results_df = None
        
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            
            if "text" not in df.columns:
                st.error("CSV must have a column named 'text'.")
            else:
                with st.spinner(f"Processing {len(df)} articles..."):
                    texts = [str(text) if pd.notna(text) else "" for text in df["text"]]
                    
                    preds, confs = predict_batch(texts, tokenizer, model, batch_size=16)
                    
                    df["Prediction"] = ["Fake" if p == 1 else "Real" for p in preds]
                    df["Confidence"] = [f"{c*100:.1f}%" for c in confs]
                    results_df = df
                    
                    st.success(f"Processed {len(df)} articles!")
                    st.dataframe(df, use_container_width=True)
                    
                    fake_count = sum(1 for p in preds if p == 1)
                    real_count = len(preds) - fake_count
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Articles", len(preds))
                    col2.metric("Real News", real_count, delta=f"{real_count/len(preds)*100:.1f}%")
                    col3.metric("Fake News", fake_count, delta=f"{fake_count/len(preds)*100:.1f}%")
                    
                    if "label" in df.columns:
                        st.divider()
                        y_true = df["label"].tolist()
                        y_pred = preds
                        
                        acc = accuracy_score(y_true, y_pred)
                        prec = precision_score(y_true, y_pred, zero_division=0)
                        rec = recall_score(y_true, y_pred, zero_division=0)
                        f1 = f1_score(y_true, y_pred, zero_division=0)
                        cm = confusion_matrix(y_true, y_pred)
                        
                        st.markdown("### Model Performance")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Accuracy", f"{acc*100:.2f}%")
                        col2.metric("Precision", f"{prec*100:.2f}%")
                        col3.metric("Recall", f"{rec*100:.2f}%")
                        col4.metric("F1-Score", f"{f1*100:.2f}%")
                        
                        cm_df = pd.DataFrame(
                            cm,
                            index=["True Real (0)", "True Fake (1)"],
                            columns=["Pred Real (0)", "Pred Fake (1)"]
                        )
                        st.write("#### Confusion Matrix")
                        st.dataframe(cm_df, use_container_width=True)
        
        else:
            content = uploaded_file.read().decode("utf-8")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            
            if not lines:
                st.warning("The file is empty or contains no valid text.")
            else:
                with st.spinner(f"Processing {len(lines)} lines..."):
                    preds, confs = predict_batch(lines, tokenizer, model, batch_size=16)
                    
                    results = []
                    for line, pred, conf in zip(lines, preds, confs):
                        results.append({
                            "Text": line[:100] + "..." if len(line) > 100 else line,
                            "Prediction": "Fake" if pred == 1 else "Real",
                            "Confidence": f"{conf*100:.1f}%"
                        })
                    
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
        
        if results_df is not None:
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Predictions as CSV", 
                data=csv, 
                file_name="fake_news_predictions.csv", 
                mime="text/csv",
                use_container_width=True
            )
    
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please ensure your file is properly formatted.")

st.divider()
st.caption("Powered by BERT | Built with Streamlit")
