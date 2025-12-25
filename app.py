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
model = genai.GenerativeModel("gemini-2.5-flash")

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
        return "Explanation unavailable due to API limitations."

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

        # Images with page context
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            img_bytes = doc.extract_image(xref)["image"]
            image = Image.open(io.BytesIO(img_bytes))
            images.append((page_no, img_idx + 1, image, page_text))

        # Tables
        for block in page.get_text("blocks"):
            if "\t" in block[4]:
                tables.append((page_no, block[4]))

        # Formulas
        formulas.extend(re.findall(r"\$.*?\$|\\\[.*?\\\]", page_text))

    return text, images, tables, formulas

# --------------------------------------------------
# IMAGE EXPLANATION (TEXT-BASED, SAFE)
# --------------------------------------------------
def explain_image_from_text(page_text, domain):
    context = page_text[:1500]

    prompt = f"""
    You are a domain expert in {domain}.

    The following text appears on a page that contains a figure or image:

    {context}

    Based on this text:
    - Infer what the image most likely represents
    - Explain its purpose in the document
    - Describe the insight it conveys

    Do not mention that the image was inferred.
    Write clearly and confidently.
    """

    return safe_generate(prompt)

# --------------------------------------------------
# SUMMARY FUNCTIONS (<= 5000 CHARS)
# --------------------------------------------------
def bert_summary(text, max_chars=5000):
    chunk = text[:3000]
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
    Generate an abstractive summary in the style of a BiLSTM-based sequence model.
    Domain: {domain}
    Single continuous text, no headings.

    Document:
    {limited_text}
    """

    return safe_generate(prompt)[:max_chars]

def hybrid_summary(bert_sum, bilstm_sum, domain, max_chars=5000):
    prompt = f"""
    Combine the two summaries below into one coherent summary.
    Domain: {domain}
    No headings or bullet points.

    BERT summary:
    {bert_sum}

    BiLSTM summary:
    {bilstm_sum}
    """

    return safe_generate(prompt)[:max_chars]

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

        # ---------------- IMAGES + EXPLANATIONS ----------------
        st.header("Extracted Images and Inferred Explanations")
        if images:
            for p, i, img, page_text in images:
                st.image(img, caption=f"Page {p}, Image {i}")
                st.write(explain_image_from_text(page_text, domain))
        else:
            st.write("No images found.")

        # ---------------- TABLES ----------------
        st.header("Extracted Tables")
        if tables:
            for p, table in tables:
                st.code(table)
        else:
            st.write("No tables found.")

        # ---------------- FORMULAS ----------------
        st.header("Extracted Formulas")
        if formulas:
            for f in formulas:
                st.latex(f)
        else:
            st.write("No formulas found.")

        # ---------------- SUMMARIES ----------------
        st.header("Generated Summaries")

        with st.spinner("Generating BERT-style summary"):
            s_bert = bert_summary(text)

        with st.spinner("Generating BiLSTM-style summary"):
            s_bilstm = bilstm_style_summary(text, domain)

        with st.spinner("Generating Hybrid summary"):
            s_hybrid = hybrid_summary(s_bert, s_bilstm, domain)

        st.subheader("BERT-style Summary")
        st.write(f"Length: {len(s_bert)} characters")
        st.write(s_bert)

        st.subheader("BiLSTM-style Summary")
        st.write(f"Length: {len(s_bilstm)} characters")
        st.write(s_bilstm)

        st.subheader("Hybrid Summary (BERT + BiLSTM)")
        st.write(f"Length: {len(s_hybrid)} characters")
        st.write(s_hybrid)

        os.remove(pdf_path)
