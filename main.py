"""
AI CV / Resume Parser — FastAPI Backend
========================================
Endpoint: POST /parse-cv/
Accepts:  PDF file upload
Returns:  Excel (.xlsx) file with extracted structured data
"""

import os
import io
import json
import tempfile

import pdfplumber
import pandas as pd
from groq import Groq
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # Reads GROQ_API_KEY from .env file

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = "llama-3.3-70b-versatile"

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY environment variable not set. "
        "Create a .env file with: GROQ_API_KEY=your_key_here"
    )

client = Groq(api_key=GROQ_API_KEY)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI CV Parser API",
    description="Extract structured data from PDF resumes using Groq LLaMA 3 70B",
    version="1.0.0",
)

# CORS — allow requests from your Netlify frontend and localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500",   # VS Code Live Server
        "http://localhost:5500",
        "https://*.netlify.app",    # All Netlify preview/prod deployments
        # Add your specific Netlify domain below after deploying:
        # "https://your-app-name.netlify.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core Pipeline Functions
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract all text from PDF bytes using pdfplumber.
    Works with in-memory bytes — no temp file needed.
    """
    full_text = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)

    combined = "\n".join(full_text).strip()

    if not combined:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be image-based — try a text-based PDF."
        )

    return combined


def call_groq_for_extraction(resume_text: str) -> dict:
    """
    Send resume text to Groq API and parse the structured JSON response.
    Uses temperature=0 for deterministic, consistent extraction.
    """
    system_prompt = """
You are an expert HR data extraction engine. Your sole purpose is to extract 
structured information from resume text and return it as a valid JSON object.

STRICT OUTPUT RULES:
- Return ONLY a raw JSON object. No markdown. No code fences. No explanation.
- The response must start with '{' and end with '}'
- Do not include ANY text before or after the JSON object

REQUIRED JSON SCHEMA (all fields are mandatory):
{
  "Full_Name": "string — candidate's full name",
  "Email": "string — email address or 'Not Found'",
  "Phone_Number": "string — phone number or 'Not Found'",
  "Job_Title": "string — most recent or primary job title",
  "Years_of_Experience": 0,
  "Technical_Skills": ["array", "of", "skill", "strings"]
}

Rules:
- Years_of_Experience must be an integer (e.g., 5, not "5 years")
- Technical_Skills must be an array of strings, never a single string
- If a field cannot be found, use 'Not Found' for strings, 0 for numbers, [] for arrays
"""

    user_prompt = f"Extract the structured data from this resume:\n\n{resume_text}"

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
    )

    raw_content = response.choices[0].message.content.strip()

    # Clean any accidental markdown fences
    if raw_content.startswith("```"):
        lines = raw_content.split("\n")
        raw_content = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Groq returned invalid JSON: {e}. "
            f"Raw response: {raw_content[:300]}"
        )

    return parsed


def build_excel_bytes(data: dict) -> bytes:
    """
    Convert extracted CV data dict to Excel file bytes.
    Returns raw bytes suitable for streaming as a download.
    """
    flat_data = data.copy()

    # Flatten Technical_Skills list → comma-separated string for Excel
    if isinstance(flat_data.get("Technical_Skills"), list):
        flat_data["Technical_Skills"] = ", ".join(flat_data["Technical_Skills"])

    df = pd.DataFrame([flat_data])

    # Write to in-memory buffer (no disk I/O needed)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Parsed CV")

        # Auto-size columns
        worksheet = writer.sheets["Parsed CV"]
        for col in worksheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = col[0].column_letter
            worksheet.column_dimensions[col_letter].width = min(max_len + 4, 60)

    buffer.seek(0)
    return buffer.read()


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "AI CV Parser API",
        "model": MODEL_NAME,
        "version": "1.0.0",
    }


@app.post("/parse-cv/", tags=["CV Parsing"])
async def parse_cv(file: UploadFile = File(...)):
    """
    Parse a PDF resume and return structured data as an Excel file.

    - **file**: PDF resume file (multipart/form-data)

    Returns an Excel (.xlsx) file with fields:
    Full_Name, Email, Phone_Number, Job_Title, Years_of_Experience, Technical_Skills
    """
    # ---- Validate file type ----
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted. Please upload a .pdf file.",
        )

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        # Some browsers send octet-stream for PDFs — we allow it
        if file.content_type and "pdf" not in file.content_type.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content type: {file.content_type}. Expected application/pdf.",
            )

    # ---- Read file bytes ----
    try:
        pdf_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read uploaded file: {e}")

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(pdf_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 10 MB.")

    # ---- Extract text from PDF ----
    try:
        resume_text = extract_text_from_pdf(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF extraction failed: {e}")

    # ---- Call Groq API ----
    try:
        extracted_data = call_groq_for_extraction(resume_text)
    except ValueError as e:
        raise HTTPException(status_code=502, detail=f"AI extraction error: {e}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Groq API unavailable: {e}")

    # ---- Build Excel output ----
    try:
        excel_bytes = build_excel_bytes(extracted_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Excel generation failed: {e}")

    # ---- Generate output filename based on candidate name ----
    candidate_name = extracted_data.get("Full_Name", "parsed_cv")
    safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "" for c in candidate_name)
    safe_name = safe_name.strip().replace(" ", "_") or "parsed_cv"
    output_filename = f"{safe_name}_parsed.xlsx"

    # ---- Return Excel as downloadable response ----
    return StreamingResponse(
        io.BytesIO(excel_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{output_filename}"',
            "X-Candidate-Name": extracted_data.get("Full_Name", ""),
            "X-Job-Title": extracted_data.get("Job_Title", ""),
        },
    )


# ---------------------------------------------------------------------------
# Run with: uvicorn main:app --reload --port 8000
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
