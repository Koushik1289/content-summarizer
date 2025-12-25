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
    page_title="Domain-Aware PDF Summarizer",
    layout="wide"
)

st.title("Domain-Aware PDF Summarizer")

# --------------------------------------------------
# LOAD API KEY FROM STREAMLIT CLOUD SECRETS
# --------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY not found in Streamlit Cloud secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --------------------------------------------------
# SINGLE STABLE MODEL
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
# UTIL
# --------------------------------------------------
def word_count(text):
    return len(text.split())

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

        # Tables (raw blocks)
        for block in page.get_text("blocks"):
            if "\t" in block[4]:
                tables.append((page_no, block[4]))

        # Formulas (raw)
        formulas.extend(re.findall(r"\$.*?\$|\\\[.*?\\\]", page_text))

    return full_text, images, tables, formulas

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
    You are a domain expert in {domain}.
    Generate a detailed domain-aware summary.

    Text:
    {text}
    """
    return model.generate_content(prompt).text

def ensemble_summary(bert, bilstm, gemini, domain):
    prompt = f"""
    Combine the following summaries into a single coherent summary.

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
    final_summary = base_summary

    sections = [
        "Introduction and Background",
        "Context and Motivation",
        "Detailed Content Overview",
        "Conceptual and Technical Analysis",
        "Domain-Specific Perspective",
        "Practical Implications",
        "Challenges and Limitations",
        "Future Scope",
        "Conclusion"
    ]

    for section in sections:
        if word_count(final_summary) >= min_words:
            break

        prompt = f"""
        Write a detailed section titled '{section}' for a {domain} audience.
        Minimum 500 words.
        Do not repeat previous content.

        Reference text:
        {source_text[:6000]}
        """
        section_text = model.generate_content(prompt).text
        final_summary += f"\n\n## {section}\n{section_text}"

    return final_summary

# --------------------------------------------------
# UI FLOW
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload PDF document", type="pdf")

if uploaded_file:
    domain = st.selectbox("Select summary domain", DOMAINS)

    if st.button("Generate Summary"):
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

        # ---------------- SUMMARY ----------------
        st.header("Generated Summary")

        bert = bert_summary(text)
        bilstm = bilstm_style_summary(text, domain)
        gemini = gemini_domain_summary(text, domain)

        base_summary = ensemble_summary(bert, bilstm, gemini, domain)

        with st.spinner("Generating long summary"):
            final_summary = generate_long_summary(base_summary, text, domain)

        st.write(f"Total word count: {word_count(final_summary)}")
        st.write(final_summary)

        os.remove(pdf_path)
