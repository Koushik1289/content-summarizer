import streamlit as st
import fitz  # PyMuPDF
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
# GEMINI MODEL (TEXT ONLY)
# --------------------------------------------------
model = genai.GenerativeModel("gemini-1.5-pro")

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
# LOAD BERT SUMMARIZER (CACHED)
# --------------------------------------------------
@st.cache_resource
def load_bert():
    return pipeline("summarization", model="facebook/bart-large-cnn")

bert_summarizer = load_bert()

# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------
def extract_pdf_components(pdf_path):
    doc = fitz.open(pdf_path)

    full_text = ""
    images = []
    tables = []
    formulas = []

    for page_no, page in enumerate(doc, start=1):
        page_text = page.get_text()
        full_text += page_text

        # Images
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            img_bytes = doc.extract_image(xref)["image"]
            image = Image.open(io.BytesIO(img_bytes))
            images.append((page_no, img_idx + 1, image))

        # Tables (raw text blocks)
        for block in page.get_text("blocks"):
            if "\t" in block[4]:
                tables.append((page_no, block[4]))

        # Formulas (raw patterns)
        formulas.extend(re.findall(r"\$.*?\$|\\\[.*?\\\]", page_text))

    return full_text, images, tables, formulas

# --------------------------------------------------
# SUMMARY GENERATORS (<= 5000 CHARS EACH)
# --------------------------------------------------
def bert_summary(text, max_chars=5000):
    chunk = text[:3000]  # BERT input limit
    summary = bert_summarizer(
        chunk,
        max_length=180,
        min_length=80,
        do_sample=False
    )[0]["summary_text"]
    return summary[:max_chars]

def bilstm_style_summary(text, domain, max_chars=5000):
    limited_text = text[:12000]

    prompt = f"""
    You are generating a summary in the style of a BiLSTM-based sequence model.
    Domain: {domain}

    Characteristics:
    - Abstractive
    - Smooth sequential flow
    - Captures context across the document
    - No headings or bullet points

    Document:
    {limited_text}
    """

    response = model.generate_content(prompt)
    return response.text[:max_chars]

def hybrid_summary(bert_sum, bilstm_sum, domain, max_chars=5000):
    prompt = f"""
    Combine the two summaries below into a single coherent summary.
    Domain: {domain}

    Requirements:
    - Preserve factual precision from BERT-style summary
    - Preserve contextual flow from BiLSTM-style summary
    - Single continuous text
    - No headings or lists
    - Maximum {max_chars} characters

    BERT-style summary:
    {bert_sum}

    BiLSTM-style summary:
    {bilstm_sum}
    """

    response = model.generate_content(prompt)
    return response.text[:max_chars]

# --------------------------------------------------
# UI FLOW
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload PDF document", type="pdf")

if uploaded_file:
    domain = st.selectbox("Select summary domain", DOMAINS)

    if st.button("Generate Summaries"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Extracting content"):
            text, images, tables, formulas = extract_pdf_components(pdf_path)

        st.success("Extraction completed")

        # ---------------- DISPLAY EXTRACTED CONTENT ----------------
        st.header("Extracted Images")
        if images:
            for p, i, img in images:
                st.image(img, caption=f"Page {p}, Image {i}")
        else:
            st.write("No images found.")

        st.header("Extracted Tables")
        if tables:
            for p, table in tables:
                st.code(table)
        else:
            st.write("No tables found.")

        st.header("Extracted Formulas")
        if formulas:
            for f in formulas:
                st.latex(f)
        else:
            st.write("No formulas found.")

        # ---------------- SUMMARIES ----------------
        st.header("Generated Summaries")

        with st.spinner("Generating BERT-style summary"):
            summary_bert = bert_summary(text)

        with st.spinner("Generating BiLSTM-style summary"):
            summary_bilstm = bilstm_style_summary(text, domain)

        with st.spinner("Generating Hybrid summary"):
            summary_hybrid = hybrid_summary(summary_bert, summary_bilstm, domain)

        st.subheader("BERT-style Summary")
        st.write(f"Length: {len(summary_bert)} characters")
        st.write(summary_bert)

        st.subheader("BiLSTM-style Summary")
        st.write(f"Length: {len(summary_bilstm)} characters")
        st.write(summary_bilstm)

        st.subheader("Hybrid (BERT + BiLSTM) Summary")
        st.write(f"Length: {len(summary_hybrid)} characters")
        st.write(summary_hybrid)

        os.remove(pdf_path)
