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


def ocr_single_image(image_path: str, lang: str) -> str:
    img = Image.open(image_path)
    txt = pytesseract.image_to_string(img, lang=lang) or ""
    return txt.strip()


def detect_file_type(upload: UploadFile) -> str:
    content_type = (upload.content_type or "").lower()
    filename = (upload.filename or "").lower()

    if content_type == "application/pdf" or filename.endswith(".pdf"):
        return "pdf"

    image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif")
    if content_type.startswith("image/") or filename.endswith(image_exts):
        return "image"

    return "unknown"


@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    password: Optional[str] = Form(default=None),  # kept for future PDF password handling
    lang: str = Form(default="fra+eng"),
):
    file_type = detect_file_type(file)

    with tempfile.TemporaryDirectory() as tmp:
        content = await file.read()

        # -------------------------
        # PDF path
        # -------------------------
        if file_type == "pdf":
            pdf_path = os.path.join(tmp, "doc.pdf")
            with open(pdf_path, "wb") as f:
                f.write(content)

            # 1) Try text layer first
            text = try_text_layer(pdf_path)
            if text:
                return {
                    "ok": True,
                    "mode": "text_layer",
                    "file_type": "pdf",
                    "text": text,
                    "error": None,
                }

            # 2) OCR from rendered PDF pages
            try:
                img_dir = os.path.join(tmp, "imgs")
                os.makedirs(img_dir, exist_ok=True)

                imgs = pdf_to_images(pdf_path, img_dir)
                if not imgs:
                    return {
                        "ok": False,
                        "mode": "ocr",
                        "file_type": "pdf",
                        "text": "",
                        "error": "NO_IMAGES_FROM_PDF",
                    }

                ocr_text = ocr_images(imgs, lang=lang)
                if not ocr_text:
                    return {
                        "ok": False,
                        "mode": "ocr",
                        "file_type": "pdf",
                        "text": "",
                        "error": "OCR_EMPTY",
                    }

                return {
                    "ok": True,
                    "mode": "ocr",
                    "file_type": "pdf",
                    "text": ocr_text,
                    "error": None,
                }

            except subprocess.CalledProcessError as e:
                return {
                    "ok": False,
                    "mode": "ocr",
                    "file_type": "pdf",
                    "text": "",
                    "error": f"POPPLER_FAIL:{e}",
                }
            except Exception as e:
                return {
                    "ok": False,
                    "mode": "ocr",
                    "file_type": "pdf",
                    "text": "",
                    "error": f"OCR_FAIL:{type(e).__name__}:{e}",
                }

        # -------------------------
        # Image path
        # -------------------------
        if file_type == "image":
            original_name = file.filename or "upload_image"
            image_path = os.path.join(tmp, original_name)

            with open(image_path, "wb") as f:
                f.write(content)

            try:
                ocr_text = ocr_single_image(image_path, lang)
                if not ocr_text:
                    return {
                        "ok": False,
                        "mode": "ocr",
                        "file_type": "image",
                        "text": "",
                        "error": "OCR_EMPTY",
                    }

                return {
                    "ok": True,
                    "mode": "ocr",
                    "file_type": "image",
                    "text": ocr_text,
                    "error": None,
                }

            except Exception as e:
                return {
                    "ok": False,
                    "mode": "ocr",
                    "file_type": "image",
                    "text": "",
                    "error": f"IMAGE_OCR_FAIL:{type(e).__name__}:{e}",
                }

        # -------------------------
        # Unsupported
        # -------------------------
        return {
            "ok": False,
            "mode": "unsupported",
            "file_type": "unknown",
            "text": "",
            "error": f"UNSUPPORTED_FILE_TYPE:{file.content_type or 'unknown'}:{file.filename or 'no_filename'}",
        }
