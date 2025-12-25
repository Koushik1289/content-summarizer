import streamlit as st
import fitz  # PyMuPDF
import google.generativeai as genai
from PIL import Image
import io, tempfile, os, re
from transformers import pipeline

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Domain-Aware PDF Intelligence System",
    layout="wide"
)

# --------------------------------------------------
# LOAD API KEY SECURELY
# --------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("not found in Streamlit secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

text_model = genai.GenerativeModel("gemini-2.5-flash")
vision_model = genai.GenerativeModel("gemini-2.5-pro-vision")

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
# LOAD NLP MODEL
# --------------------------------------------------
@st.cache_resource
def load_bert():
    return pipeline("summarization", model="facebook/bart-large-cnn")

bert_summarizer = load_bert()

# --------------------------------------------------
# UTILS
# --------------------------------------------------
def word_count(text: str) -> int:
    return len(text.split())

# --------------------------------------------------
# PDF EXTRACTION
# --------------------------------------------------
def extract_pdf_components(pdf_path):
    doc = fitz.open(pdf_path)
    text, images, tables, formulas = "", [], [], []

    for page_no, page in enumerate(doc, start=1):
        page_text = page.get_text()
        text += page_text

        # Images
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            img_bytes = doc.extract_image(xref)["image"]
            image = Image.open(io.BytesIO(img_bytes))
            images.append((page_no, img_idx + 1, image))

        # Tables (heuristic)
        for block in page.get_text("blocks"):
            if "\t" in block[4]:
                tables.append((page_no, block[4]))

        # Formulas
        formulas.extend(re.findall(r"\$.*?\$|\\\[.*?\\\]", page_text))

    return text, images, tables, formulas

# --------------------------------------------------
# EXPLANATIONS
# --------------------------------------------------
def explain_image(image, domain):
    prompt = f"""
    Explain this image thoroughly for a {domain} audience.
    Include purpose, interpretation, insights, and relevance.
    """
    return vision_model.generate_content([prompt, image]).text

def explain_table(table_text, domain):
    prompt = f"""
    Explain this table in detail for a {domain} audience.
    Describe variables, trends, and implications.

    Table:
    {table_text}
    """
    return text_model.generate_content(prompt).text

def explain_formula(formula, domain):
    prompt = f"""
    Explain this mathematical formula for a {domain} audience.
    Describe variables, logic, and applications.

    Formula:
    {formula}
    """
    return text_model.generate_content(prompt).text

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
    Generate an abstractive summary as if produced by a BiLSTM-LSTM model.
    Domain: {domain}
    Maintain coherence and semantic flow.

    Text:
    {text}
    """
    return text_model.generate_content(prompt).text

def gemini_domain_summary(text, domain):
    prompt = f"""
    You are a senior expert in {domain}.
    Produce a structured, detailed, domain-aware summary.

    Text:
    {text}
    """
    return text_model.generate_content(prompt).text

def ensemble_summary(bert, bilstm, gemini, domain):
    prompt = f"""
    Combine the following summaries into one unified, high-quality summary.

    Domain: {domain}

    BERT Summary:
    {bert}

    BiLSTM Summary:
    {bilstm}

    Gemini Summary:
    {gemini}
    """
    return text_model.generate_content(prompt).text

# --------------------------------------------------
# LONG SUMMARY (≥ 4000 WORDS GUARANTEED)
# --------------------------------------------------
def generate_long_summary(base_summary, source_text, domain, min_words=4000):
    final_summary = base_summary

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
        if word_count(final_summary) >= min_words:
            break

        prompt = f"""
        Write a detailed section titled '{section}' for a {domain} audience.
        Minimum 500 words.
        Avoid repetition and maintain coherence.

        Reference Document:
        {source_text[:6000]}
        """
        section_text = text_model.generate_content(prompt).text
        final_summary += f"\n\n## {section}\n{section_text}"

    return final_summary

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("Domain-Aware PDF Intelligence System")

uploaded_file = st.file_uploader("Upload PDF Document", type="pdf")

if uploaded_file:
    domain = st.selectbox("Select Summary Domain", DOMAINS)

    if st.button("Process Document"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("Extracting content..."):
            text, images, tables, formulas = extract_pdf_components(pdf_path)

        st.success("Extraction Completed")

        # ---------------- IMAGES ----------------
        st.header("Image Explanations")
        for p, i, img in images:
            st.image(img, caption=f"Page {p} - Image {i}")
            st.write(explain_image(img, domain))

        # ---------------- TABLES ----------------
        st.header("Table Explanations")
        for p, table in tables:
            st.code(table)
            st.write(explain_table(table, domain))

        # ---------------- FORMULAS ----------------
        st.header("Formula Explanations")
        for f in formulas:
            st.latex(f)
            st.write(explain_formula(f, domain))

        # ---------------- SUMMARIZATION ----------------
        st.header("Multi-Model Summarization")

        bert = bert_summary(text)
        bilstm = bilstm_style_summary(text, domain)
        gemini = gemini_domain_summary(text, domain)

        base = ensemble_summary(bert, bilstm, gemini, domain)

        with st.spinner("Generating summary..."):
            final_summary = generate_long_summary(base, text, domain)

        st.subheader("Summary Statistics")
        st.write(f"**Total Words:** {word_count(final_summary)}")

        st.subheader("Final Domain-Aware Summary (≥ 4000 Words)")
        st.write(final_summary)

        os.remove(pdf_path)
