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
    page_title="Domain-Aware PDF Intelligence System",
    layout="wide"
)

# --------------------------------------------------
# LOAD API KEY FROM STREAMLIT CLOUD SECRETS
# --------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY not found in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --------------------------------------------------
# USE ONE STABLE MODEL ONLY
# --------------------------------------------------
MODEL_NAME = "gemini-1.5-pro"
model = genai.GenerativeModel(MODEL_NAME)

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
# UTIL
# --------------------------------------------------
def word_count(text):
    return len(text.split())

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
            images.append((page_no, img_idx + 1, image))

        for block in page.get_text("blocks"):
            if "\t" in block[4]:
                tables.append((page_no, block[4]))

        formulas.extend(re.findall(r"\$.*?\$|\\\[.*?\\\]", page_text))

    return text, images, tables, formulas

# --------------------------------------------------
# EXPLANATIONS (SAFE, NO VISION MODEL)
# --------------------------------------------------
def explain_image(image, domain):
    try:
        prompt = f"""
        You are a domain expert in {domain}.
        Explain the image in detail.
        Describe what it represents, its components,
        patterns, and its relevance to the document.
        """
        response = model.generate_content([prompt, image])
        return response.text
    except Exception:
        return "Image explanation unavailable due to API or quota limitations."

def explain_table(table_text, domain):
    prompt = f"""
    Explain the following table for a {domain} audience.
    Discuss variables, trends, relationships, and implications.

    Table:
    {table_text}
    """
    return model.generate_content(prompt).text

def explain_formula(formula, domain):
    prompt = f"""
    Explain this formula for a {domain} audience.
    Describe variables, logic, and usage.

    Formula:
    {formula}
    """
    return model.generate_content(prompt).text

# --------------------------------------------------
# SUMMARIZATION
# --------------------------------------------------
def bert_summary(text):
    return bert_summarizer(
        text[:3000],
        max_length=180,
        min_length=80,
        do_sample=False
    )[0]["summary_text"]

def bilstm_style_summary(text, domain):
    prompt = f"""
    Generate an abstractive summary similar to a BiLSTM-LSTM model.
    Domain: {domain}

    Text:
    {text}
    """
    return model.generate_content(prompt).text

def gemini_domain_summary(text, domain):
    prompt = f"""
    You are a senior expert in {domain}.
    Produce a structured, domain-aware summary.

    Text:
    {text}
    """
    return model.generate_content(prompt).text

def ensemble_summary(bert, bilstm, gemini, domain):
    prompt = f"""
    Combine the following summaries into one unified summary.

    Domain: {domain}

    BERT Summary:
    {bert}

    BiLSTM Summary:
    {bilstm}

    Gemini Summary:
    {gemini}
    """
    return model.generate_content(prompt).text

# --------------------------------------------------
# LONG SUMMARY (>= 4000 WORDS)
# --------------------------------------------------
def generate_long_summary(base_summary, source_text, domain, min_words=4000):
    final = base_summary

    sections = [
        "Introduction and Background",
        "Context and Motivation",
        "Detailed Content Explanation",
        "Technical or Conceptual Analysis",
        "Domain-Specific Interpretation",
        "Examples and Use Cases",
        "Implications and Impact",
        "Limitations and Challenges",
        "Future Scope and Recommendations",
        "Conclusion"
    ]

    for section in sections:
        if word_count(final) >= min_words:
            break

        prompt = f"""
        Write a detailed section titled '{section}' for a {domain} audience.
        Minimum 500 words.
        Avoid repetition.

        Reference:
        {source_text[:6000]}
        """
        final += "\n\n## " + section + "\n" + model.generate_content(prompt).text

    return final

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Domain-Aware PDF Intelligence System")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    domain = st.selectbox("Select summary domain", DOMAINS)

    if st.button("Process Document"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Extracting content"):
            text, images, tables, formulas = extract_pdf_components(pdf_path)

        st.success("Extraction completed")

        st.header("Image Explanations")
        for p, i, img in images:
            st.image(img, caption=f"Page {p}, Image {i}")
            st.write(explain_image(img, domain))

        st.header("Table Explanations")
        for p, table in tables:
            st.code(table)
            st.write(explain_table(table, domain))

        st.header("Formula Explanations")
        for f in formulas:
            st.latex(f)
            st.write(explain_formula(f, domain))

        st.header("Summarization")

        bert = bert_summary(text)
        bilstm = bilstm_style_summary(text, domain)
        gemini = gemini_domain_summary(text, domain)

        base = ensemble_summary(bert, bilstm, gemini, domain)

        with st.spinner("Generating long summary"):
            final_summary = generate_long_summary(base, text, domain)

        st.write("Word count:", word_count(final_summary))
        st.write(final_summary)

        os.remove(pdf_path)
