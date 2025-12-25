import streamlit as st
import fitz
import google.generativeai as genai
from PIL import Image
import io
import tempfile
import os
import re
from transformers import pipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Multi-Model PDF Summarizer",
    layout="wide"
)

st.title("Multi-Model PDF Summarizer")

# --------------------------------------------------
# LOAD API KEY FROM STREAMLIT CLOUD SECRETS
# --------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY not found in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --------------------------------------------------
# SAFE MODEL FOR STREAMLIT CLOUD
# --------------------------------------------------
model = genai.GenerativeModel("gemini-1.5-flash")

# --------------------------------------------------
# DOMAINS
# --------------------------------------------------
DOMAINS = [
    "Academic / Research",
    "Legal",
    "Medical / Healthcare",
    "Business / Finance",
    "Technical / Engineering",
    "AI / Machine Learning",
    "Government / Policy",
    "Education",
    "General",
    "Patent / Intellectual Property"
]

# --------------------------------------------------
# LOAD BERT (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_bert():
    return pipeline("summarization", model="facebook/bart-large-cnn")

bert_summarizer = load_bert()

# --------------------------------------------------
# SAFE GEMINI CALL
# --------------------------------------------------
def safe_generate(prompt):
    try:
        return model.generate_content(prompt).text
    except Exception:
        return ""

# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------
def extract_pdf_components(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    images = []
    tables = []
    formulas = []

    for page_no, page in enumerate(doc, start=1):
        page_text = page.get_text()
        text += page_text

        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            img_bytes = doc.extract_image(xref)["image"]
            image = Image.open(io.BytesIO(img_bytes))
            images.append((page_no, img_idx + 1, image, page_text))

        for block in page.get_text("blocks"):
            if "\t" in block[4]:
                tables.append((page_no, block[4]))

        formulas.extend(re.findall(r"\$.*?\$|\\\[.*?\\\]", page_text))

    return text, images, tables, formulas

# --------------------------------------------------
# IMAGE EXPLANATION (TEXT-BASED)
# --------------------------------------------------
def explain_image_from_text(page_text, domain):
    prompt = f"""
    You are a domain expert in {domain}.
    Based on the following page text, infer the purpose of the image
    and explain its role briefly.

    Page text:
    {page_text[:1200]}
    """
    return safe_generate(prompt)

# --------------------------------------------------
# SUMMARY FUNCTIONS (≈ 3000 CHARS EACH)
# --------------------------------------------------
def bert_summary(text, target_chars=3000):
    chunk = text[:3500]
    summary = bert_summarizer(
        chunk,
        max_length=220,
        min_length=150,
        do_sample=False
    )[0]["summary_text"]
    return summary[:target_chars]

def bilstm_style_summary(text, domain, target_chars=3000):
    limited_text = text[:12000]

    prompt = f"""
    Generate an abstractive summary similar to a BiLSTM-based sequence model.
    Domain: {domain}
    Single continuous text.
    Target length: approximately {target_chars} characters.

    Document:
    {limited_text}
    """

    return safe_generate(prompt)[:target_chars]

def hybrid_summary(bert_sum, bilstm_sum, domain, target_chars=3000):
    prompt = f"""
    Combine the two summaries below into one coherent summary.
    Domain: {domain}
    Single continuous text.
    Target length: approximately {target_chars} characters.

    Summary A:
    {bert_sum}

    Summary B:
    {bilstm_sum}
    """

    return safe_generate(prompt)[:target_chars]

# --------------------------------------------------
# UI FLOW
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload PDF document", type="pdf")

if uploaded_file:
    domain = st.selectbox("Select summary domain", DOMAINS)

    if st.button("Generate Output"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Extracting content"):
            text, images, tables, formulas = extract_pdf_components(pdf_path)

        st.success("Extraction completed")

        # ---------------- IMAGES ----------------
        st.header("Extracted Images and Context-Based Explanations")
        for p, i, img, page_text in images:
            st.image(img, caption=f"Page {p}, Image {i}")
            st.write(explain_image_from_text(page_text, domain))

        # ---------------- TABLES ----------------
        st.header("Extracted Tables")
        for p, table in tables:
            st.code(table)

        # ---------------- FORMULAS ----------------
        st.header("Extracted Formulas")
        for f in formulas:
            st.latex(f)

        # ---------------- SUMMARIES ----------------
        st.header("Generated Summaries")

        with st.spinner("Generating BERT-style summary"):
            s_bert = bert_summary(text)

        with st.spinner("Generating BiLSTM-style summary"):
            s_bilstm = bilstm_style_summary(text, domain)

        with st.spinner("Generating Hybrid summary"):
            s_hybrid = hybrid_summary(s_bert, s_bilstm, domain)

        st.subheader("BERT-style Summary")
        st.write(f"Characters: {len(s_bert)}")
        st.write(s_bert)

        st.subheader("BiLSTM-style Summary")
        st.write(f"Characters: {len(s_bilstm)}")
        st.write(s_bilstm)

        st.subheader("Hybrid Summary (BERT + BiLSTM)")
        st.write(f"Characters: {len(s_hybrid)}")
        st.write(s_hybrid)

        os.remove(pdf_path)
