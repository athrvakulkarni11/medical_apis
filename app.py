import os
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from fastapi import FastAPI, File, UploadFile
from typing import List, Dict
from shutil import rmtree
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
app = FastAPI()

OUTPUT_DIR = "output"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_images_and_text(pdf_bytes: bytes, output_dir: str):
    """Extract text and images from the uploaded PDF file."""
    
    # Clear output directory
    if os.path.exists(output_dir):
        rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Save PDF as a temporary file
    temp_pdf_path = os.path.join(output_dir, "temp.pdf")
    with open(temp_pdf_path, "wb") as temp_pdf:
        temp_pdf.write(pdf_bytes)

    # Extract text from PDF
    doc = fitz.open(temp_pdf_path)
    full_text = "\n".join([page.get_text("text") for page in doc])

    # Extract images from PDF
    images = convert_from_bytes(pdf_bytes)
    image_paths = []
    for i, image in enumerate(images):
        image_path = os.path.join(images_dir, f"image_{i+1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)

    return full_text, image_paths

def extract_important_terms(text: str) -> Dict:
    """Use Groq to extract key medical terms from text."""
    prompt = f"""
    Extract and return only the key medical metrics from the following text, including:
    - Blood Pressure (BP)
    - Heart Rate
    - Risk Factors (e.g., hypertension, cardiovascular risk, diabetes, etc.)
    - Key medical findings
    
    Provide the response as a structured JSON format with fields: 'blood_pressure', 'heart_rate', 'risk_factors', and 'key_findings'.
    
    Text: {text}
    
    Do not add any extra text other than the mentioned json format
    """
    
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )

    return response.choices[0].message.content  # JSON format from Groq

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    extracted_text, images = extract_images_and_text(pdf_bytes, OUTPUT_DIR)

    # Process extracted text with Groq to get key terms
    important_terms = extract_important_terms(extracted_text)

    return {
        "message": "PDF processed successfully",
        "important_terms": important_terms,
        "image_files": images
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
