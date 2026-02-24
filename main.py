import io
import os
import tempfile
import subprocess
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from pypdf import PdfReader
from PIL import Image
import pytesseract

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "service": "heylisa-pdf-ocr"}

@app.get("/health")
def health():
    return {"ok": True}

def try_text_layer(pdf_path: str) -> str:
    """Extract selectable text if PDF has a text layer."""
    try:
        reader = PdfReader(pdf_path)
        chunks = []
        for page in reader.pages:
            t = page.extract_text() or ""
            t = t.strip()
            if t:
                chunks.append(t)
        return "\n\n".join(chunks).strip()
    except Exception:
        return ""

def pdf_to_images(pdf_path: str, out_dir: str) -> list[str]:
    """
    Convert PDF pages to PNG using pdftoppm (poppler).
    Returns list of image paths.
    """
    # creates out_dir/page-1.png, page-2.png...
    prefix = os.path.join(out_dir, "page")
    cmd = ["pdftoppm", "-png", "-r", "300", pdf_path, prefix]
    subprocess.check_call(cmd)
    images = []
    for name in sorted(os.listdir(out_dir)):
        if name.startswith("page") and name.endswith(".png"):
            images.append(os.path.join(out_dir, name))
    return images

def ocr_images(image_paths: list[str], lang: str) -> str:
    chunks = []
    for p in image_paths:
        img = Image.open(p)
        txt = pytesseract.image_to_string(img, lang=lang) or ""
        txt = txt.strip()
        if txt:
            chunks.append(txt)
    return "\n\n".join(chunks).strip()

@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    password: Optional[str] = Form(default=None),
    lang: str = Form(default="fra+eng"),
):
    # Save uploaded file
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, "doc.pdf")
        content = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(content)

        # 1) Try text layer first (fast)
        text = try_text_layer(pdf_path)
        if text:
            return {
                "ok": True,
                "mode": "text_layer",
                "text": text,
                "error": None,
            }

        # 2) OCR path (works for scanned + “simple” PDFs)
        try:
            img_dir = os.path.join(tmp, "imgs")
            os.makedirs(img_dir, exist_ok=True)
            imgs = pdf_to_images(pdf_path, img_dir)
            if not imgs:
                return {
                    "ok": False,
                    "mode": "ocr",
                    "text": "",
                    "error": "NO_IMAGES_FROM_PDF",
                }
            ocr_text = ocr_images(imgs, lang=lang)
            if not ocr_text:
                return {
                    "ok": False,
                    "mode": "ocr",
                    "text": "",
                    "error": "OCR_EMPTY",
                }
            return {
                "ok": True,
                "mode": "ocr",
                "text": ocr_text,
                "error": None,
            }
        except subprocess.CalledProcessError as e:
            return {"ok": False, "mode": "ocr", "text": "", "error": f"POPPLER_FAIL:{e}"}
        except Exception as e:
            return {"ok": False, "mode": "ocr", "text": "", "error": f"OCR_FAIL:{type(e).__name__}:{e}"}
