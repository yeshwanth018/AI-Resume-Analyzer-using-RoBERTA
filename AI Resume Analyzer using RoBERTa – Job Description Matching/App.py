from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoTokenizer, AutoModel
import PyPDF2
import docx
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

tokenizer = None
model = None

def load_model():
    global tokenizer, model
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-roberta-large-v1")
        model = AutoModel.from_pretrained("sentence-transformers/all-roberta-large-v1")
        model.eval()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_embedding(text):
    encoded = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)
    embedding = mean_pooling(output, encoded["attention_mask"])
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding.numpy()

def extract_text_from_pdf(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n".join([p.text for p in doc.paragraphs])



def find_matching_keywords(resume_text, jd_text):
    import re
    resume_text = resume_text.lower().strip()
    jd_text = jd_text.lower().strip()
    jd_words = set(re.findall(r'\b[a-zA-Z0-9+#]+\b', jd_text))
    matched = []
    missing = []
    for word in jd_words:
        if word in resume_text:
            matched.append(word)
        else:
            missing.append(word)
    return matched, missing

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    load_model()

    job_description = request.form.get("job_description", "").strip()
    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    resume_file = request.files.get("resume")
    resume_text = request.form.get("resume_text", "").strip()

    if resume_file:
        filename = resume_file.filename.lower()
        file_bytes = resume_file.read()
        if filename.endswith(".pdf"):
            resume_text = extract_text_from_pdf(file_bytes)
        elif filename.endswith(".docx"):
            resume_text = extract_text_from_docx(file_bytes)
        elif filename.endswith(".txt"):
            resume_text = file_bytes.decode("utf-8", errors="ignore")
        else:
            return jsonify({"error": "Unsupported file format. Use PDF, DOCX, or TXT"}), 400

    if not resume_text:
        return jsonify({"error": "Resume content is required"}), 400

    resume_emb = get_embedding(resume_text)
    jd_emb = get_embedding(job_description)
    score = float(cosine_similarity(resume_emb, jd_emb)[0][0])
    score_pct = round(score * 100, 1)

    matched_kw, missing_kw = find_matching_keywords(resume_text, job_description)

    if score_pct >= 75:
        verdict = "Strong Match"
        verdict_class = "strong"
    elif score_pct >= 50:
        verdict = "Moderate Match"
        verdict_class = "moderate"
    else:
        verdict = "Weak Match"
        verdict_class = "weak"

    return jsonify({
        "score": score_pct,
        "verdict": verdict,
        "verdict_class": verdict_class,
        "matched_keywords": matched_kw[:8],
        "missing_keywords": missing_kw[:8],
        "resume_length": len(resume_text.split()),
        "jd_length": len(job_description.split())
    })

if __name__ == "__main__":
    app.run(debug=True)