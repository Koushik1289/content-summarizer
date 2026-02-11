import streamlit as st
import fitz
import google.generativeai as genai
from PIL import Image
import io
import tempfile
import os
import re

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Multi-Model PDF Summarizer",
    layout="wide"
)

st.title("📄 Multi-Model PDF Summarizer")

# --------------------------------------------------
# LOAD API KEY FROM STREAMLIT SECRETS
# --------------------------------------------------
if "API_KEY" not in st.secrets:
    st.error("System configuration error. Please contact administrator.")
    st.stop()

genai.configure(api_key=st.secrets["API_KEY"])

# Internal AI model (not exposed in UI)
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
# SAFE AI CALL
# --------------------------------------------------
def safe_generate(prompt):
    try:
        response = model.generate_content(prompt)
        if response and hasattr(response, "text") and response.text:
            return response.text.strip()
        return "No output generated."
    except Exception:
        return "Unable to generate output at this time."


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
        text += page_text + "\n"

        # Extract Images
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            image = Image.open(io.BytesIO(img_bytes))
            images.append((page_no, img_idx + 1, image, page_text))

        # Extract Tables (basic heuristic)
        for block in page.get_text("blocks"):
            if "\t" in block[4]:
                tables.append((page_no, block[4]))

        # Extract Formulas
        formulas.extend(re.findall(r"\$.*?\$|\\\[.*?\\\]", page_text))

    doc.close()
    return text, images, tables, formulas


# --------------------------------------------------
# IMAGE EXPLANATION
# --------------------------------------------------
def explain_image_from_text(page_text, domain):
    prompt = f"""
    You are an expert in {domain}.
    Based on the following page content, explain the likely purpose 
    and role of the extracted image in 3–5 concise sentences.

    Page content:
    {page_text[:1500]}
    """
    return safe_generate(prompt)


# --------------------------------------------------
# SUMMARY FUNCTIONS
# --------------------------------------------------
def summary_style_a(text, domain, target_chars=3000):
    limited_text = text[:15000]

    prompt = f"""
    Generate an extractive-style summary similar to a transformer-based summarizer.
    Domain: {domain}
    Single continuous text.
    Target length: approximately {target_chars} characters.

    Document:
    {limited_text}
    """
    return safe_generate(prompt)[:target_chars]


def summary_style_b(text, domain, target_chars=3000):
    limited_text = text[:15000]

    prompt = f"""
    Generate an abstractive summary similar to a sequence-to-sequence model.
    Domain: {domain}
    Single continuous text.
    Target length: approximately {target_chars} characters.

    Document:
    {limited_text}
    """
    return safe_generate(prompt)[:target_chars]


def combined_summary(summary_a, summary_b, domain, target_chars=3000):
    prompt = f"""
    Combine the two summaries below into one coherent, high-quality summary.
    Domain: {domain}
    Single continuous text.
    Target length: approximately {target_chars} characters.

    Summary A:
    {summary_a}

    Summary B:
    {summary_b}
    """
    return safe_generate(prompt)[:target_chars]


# --------------------------------------------------
# UI FLOW
# --------------------------------------------------
uploaded_file = st.file_uploader("📤 Upload PDF document", type="pdf")

if uploaded_file:
    domain = st.selectbox("📚 Select Summary Domain", DOMAINS)

    if st.button("🚀 Generate Output"):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        with st.spinner("🔍 Extracting content from PDF..."):
            text, images, tables, formulas = extract_pdf_components(pdf_path)

        st.success("Extraction completed successfully.")

        # ---------------- IMAGES ----------------
        if images:
            st.header("🖼 Extracted Images & Context-Based Explanation")
            for p, i, img, page_text in images:
                st.image(img, caption=f"Page {p}, Image {i}")
                explanation = explain_image_from_text(page_text, domain)
                st.write(explanation)
        else:
            st.info("No images found.")

        # ---------------- TABLES ----------------
        if tables:
            st.header("📊 Extracted Tables")
            for p, table in tables:
                st.code(table)
        else:
            st.info("No tables detected.")

        # ---------------- FORMULAS ----------------
        if formulas:
            st.header("🧮 Extracted Formulas")
            for f in formulas:
                st.latex(f)
        else:
            st.info("No formulas detected.")

        # ---------------- SUMMARIES ----------------
        st.header("📝 Generated Summaries")

        with st.spinner("Generating Summary A..."):
            s_a = summary_style_a(text, domain)

        with st.spinner("Generating Summary B..."):
            s_b = summary_style_b(text, domain)

        with st.spinner("Generating Final Summary..."):
            s_final = combined_summary(s_a, s_b, domain)

        st.subheader("🔹 Summary A")
        st.write(f"Characters: {len(s_a)}")
        st.write(s_a)

        st.subheader("🔹 Summary B")
        st.write(f"Characters: {len(s_b)}")
        st.write(s_b)

        st.subheader("🔹 Final Combined Summary")
        st.write(f"Characters: {len(s_final)}")
        st.write(s_final)

        os.remove(pdf_path)
