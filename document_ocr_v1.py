"""Versioned, service-to-service document extraction contract.

The legacy ``/extract`` route intentionally remains in ``main.py``. This module
does not reuse its filename-based detection or its HTTP-200 error responses.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import subprocess
import tempfile
import time
import uuid
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import fitz
import pytesseract
from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger("document_ocr.v1")
router = APIRouter()


def _positive_int_setting(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning("document_ocr configuration_invalid name=%s", name)
        return default
    return value


MAX_FILE_SIZE_BYTES = _positive_int_setting(
    "DOCUMENT_OCR_MAX_FILE_BYTES", 15 * 1024 * 1024
)
MAX_PDF_PAGES = _positive_int_setting("DOCUMENT_OCR_MAX_PDF_PAGES", 20)
MAX_IMAGE_FRAMES = _positive_int_setting("DOCUMENT_OCR_MAX_IMAGE_FRAMES", 20)
MAX_IMAGE_PIXELS = _positive_int_setting("DOCUMENT_OCR_MAX_IMAGE_PIXELS", 40_000_000)
PDF_CONVERSION_TIMEOUT_SECONDS = _positive_int_setting(
    "DOCUMENT_OCR_PDF_TIMEOUT_SECONDS", 120
)
OCR_TIMEOUT_SECONDS = _positive_int_setting("DOCUMENT_OCR_OCR_TIMEOUT_SECONDS", 60)
ENGINE_VERSION_TIMEOUT_SECONDS = 3
UPLOAD_CHUNK_SIZE_BYTES = 1024 * 1024
PDF_RENDER_DPI = 300

ALLOWED_LANGUAGES = frozenset({"fra", "eng", "fra+eng", "eng+fra"})
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "BMP", "TIFF"})
IMAGE_FORMAT_TO_MIMES = {
    "JPEG": frozenset({"image/jpeg", "image/jpg"}),
    "PNG": frozenset({"image/png"}),
    "WEBP": frozenset({"image/webp"}),
    "BMP": frozenset({"image/bmp", "image/x-ms-bmp"}),
    "TIFF": frozenset({"image/tiff"}),
}
EXTENSION_CLAIMS = {
    ".pdf": "PDF",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
    ".webp": "WEBP",
    ".bmp": "BMP",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".heic": "HEIC",
    ".heif": "HEIC",
}
HEIC_MIMES = frozenset({"image/heic", "image/heif"})
HEIC_BRANDS = frozenset({b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis"})


class V1Error(Exception):
    def __init__(
        self,
        status_code: int,
        error_code: str,
        message: str,
        retryable: bool,
        *,
        stage: str,
    ) -> None:
        super().__init__(error_code)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        self.retryable = retryable
        self.stage = stage


class PdfConversionTimeout(Exception):
    pass


class PdfConversionFailure(Exception):
    pass


class OcrTimeout(Exception):
    pass


class OcrFailure(Exception):
    pass


@dataclass(frozen=True)
class DetectedDocument:
    file_type: str
    format_name: str
    page_count: int


def _ensure_request_context(request: Request) -> str:
    request_id = getattr(request.state, "request_id", None)
    if request_id is None:
        request_id = uuid.uuid4().hex
        request.state.request_id = request_id
    if getattr(request.state, "started_at", None) is None:
        request.state.started_at = time.monotonic()
    return request_id


def _duration_ms(request: Request) -> int:
    started_at = getattr(request.state, "started_at", time.monotonic())
    return max(0, round((time.monotonic() - started_at) * 1000))


async def v1_error_handler(request: Request, exc: V1Error) -> JSONResponse:
    request_id = _ensure_request_context(request)
    duration_ms = _duration_ms(request)
    logger.warning(
        "document_ocr request_id=%s result=failure error_code=%s "
        "retryable=%s stage=%s file_type=%s size_bytes=%s duration_ms=%s",
        request_id,
        exc.error_code,
        exc.retryable,
        exc.stage,
        getattr(request.state, "file_type", "unknown"),
        getattr(request.state, "size_bytes", "unknown"),
        duration_ms,
    )
    response_headers = {"X-Request-ID": request_id}
    if exc.status_code == 401:
        response_headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "error_code": exc.error_code,
            "message": exc.message,
            "retryable": exc.retryable,
        },
        headers=response_headers,
    )


async def v1_request_validation_error_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    if request.url.path != "/v1/extract":
        return await request_validation_exception_handler(request, exc)
    return await v1_error_handler(
        request,
        V1Error(
            422,
            "INVALID_DOCUMENT",
            "The multipart document upload is invalid.",
            False,
            stage="request_validation",
        ),
    )


def require_v1_bearer(request: Request, authorization: str | None) -> None:
    _ensure_request_context(request)
    configured_token = os.getenv("DOCUMENT_OCR_SERVICE_TOKEN")
    if not configured_token:
        raise V1Error(
            503,
            "SERVICE_NOT_CONFIGURED",
            "The document extraction service is not configured.",
            False,
            stage="auth",
        )
    if authorization is None:
        raise V1Error(
            401,
            "AUTH_REQUIRED",
            "Bearer authentication is required.",
            False,
            stage="auth",
        )
    scheme, separator, candidate = authorization.partition(" ")
    if separator == "" or scheme.lower() != "bearer" or not candidate:
        raise V1Error(
            401,
            "AUTH_INVALID",
            "Bearer authentication is invalid.",
            False,
            stage="auth",
        )
    if not secrets.compare_digest(candidate, configured_token):
        raise V1Error(
            401,
            "AUTH_INVALID",
            "Bearer authentication is invalid.",
            False,
            stage="auth",
        )


async def v1_auth_middleware(request: Request, call_next):
    if request.url.path != "/v1/extract":
        return await call_next(request)
    try:
        require_v1_bearer(request, request.headers.get("authorization"))
    except V1Error as exc:
        return await v1_error_handler(request, exc)
    return await call_next(request)


def _safe_declared_mime(upload: UploadFile) -> str:
    return (upload.content_type or "").split(";", 1)[0].strip().lower()


async def _store_upload(upload: UploadFile, target_path: str) -> int:
    size_bytes = 0
    with open(target_path, "wb") as output:
        while True:
            chunk = await upload.read(UPLOAD_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            size_bytes += len(chunk)
            if size_bytes > MAX_FILE_SIZE_BYTES:
                raise V1Error(
                    413,
                    "FILE_TOO_LARGE",
                    "The uploaded document exceeds the configured size limit.",
                    False,
                    stage="upload",
                )
            output.write(chunk)
    if size_bytes == 0:
        raise V1Error(
            422,
            "INVALID_DOCUMENT",
            "The uploaded document is empty or invalid.",
            False,
            stage="upload",
        )
    return size_bytes


def _looks_like_heic(header: bytes) -> bool:
    return len(header) >= 12 and header[4:8] == b"ftyp" and header[8:12] in HEIC_BRANDS


def _filename_claim(filename: str | None) -> str | None:
    if not filename:
        return None
    return EXTENSION_CLAIMS.get(Path(filename).suffix.lower())


def _claimed_supported_type(declared_mime: str, filename: str | None) -> bool:
    if declared_mime == "application/pdf":
        return True
    if any(declared_mime in mimes for mimes in IMAGE_FORMAT_TO_MIMES.values()):
        return True
    return _filename_claim(filename) in ALLOWED_IMAGE_FORMATS | {"PDF"}


def _validate_pdf(path: str) -> DetectedDocument:
    try:
        with fitz.open(path) as document:
            if not document.is_pdf or document.needs_pass or document.page_count <= 0:
                raise V1Error(
                    422,
                    "INVALID_DOCUMENT",
                    "The PDF document is invalid or password-protected.",
                    False,
                    stage="validation",
                )
            page_count = document.page_count
    except V1Error:
        raise
    except Exception:
        raise V1Error(
            422,
            "INVALID_DOCUMENT",
            "The PDF document is invalid or unreadable.",
            False,
            stage="validation",
        ) from None

    if page_count > MAX_PDF_PAGES:
        raise V1Error(
            413,
            "PDF_PAGE_LIMIT_EXCEEDED",
            "The PDF document exceeds the configured page limit.",
            False,
            stage="validation",
        )
    return DetectedDocument("pdf", "PDF", page_count)


def _validate_image(path: str) -> DetectedDocument:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                format_name = (image.format or "").upper()
                if format_name not in ALLOWED_IMAGE_FORMATS:
                    raise V1Error(
                        415,
                        "UNSUPPORTED_FILE_TYPE",
                        "The actual document type is not supported.",
                        False,
                        stage="validation",
                    )
                frame_count = int(getattr(image, "n_frames", 1))
                if frame_count > MAX_IMAGE_FRAMES:
                    raise V1Error(
                        413,
                        "IMAGE_FRAME_LIMIT_EXCEEDED",
                        "The image document exceeds the configured frame limit.",
                        False,
                        stage="validation",
                    )
                total_pixels = 0
                for frame_index in range(frame_count):
                    image.seek(frame_index)
                    width, height = image.size
                    total_pixels += width * height
                    if total_pixels > MAX_IMAGE_PIXELS:
                        raise V1Error(
                            413,
                            "IMAGE_PIXEL_LIMIT_EXCEEDED",
                            "The decoded image exceeds the configured pixel limit.",
                            False,
                            stage="validation",
                        )
                    image.load()
    except V1Error:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        EOFError,
        Image.DecompressionBombWarning,
        Image.DecompressionBombError,
    ):
        raise V1Error(
            422,
            "INVALID_DOCUMENT",
            "The image document is invalid or unreadable.",
            False,
            stage="validation",
        ) from None
    return DetectedDocument("image", format_name, frame_count)


def detect_document(
    path: str, declared_mime: str, original_filename: str | None
) -> DetectedDocument:
    with open(path, "rb") as source:
        header = source.read(32)

    if _looks_like_heic(header):
        raise V1Error(
            415,
            "UNSUPPORTED_FILE_TYPE",
            "HEIC and HEIF documents are not supported.",
            False,
            stage="validation",
        )

    if header.lstrip().startswith(b"%PDF-"):
        detected = _validate_pdf(path)
    else:
        try:
            detected = _validate_image(path)
        except V1Error as exc:
            if exc.error_code != "INVALID_DOCUMENT":
                raise
            if declared_mime in HEIC_MIMES or _filename_claim(original_filename) == "HEIC":
                raise V1Error(
                    415,
                    "UNSUPPORTED_FILE_TYPE",
                    "HEIC and HEIF documents are not supported.",
                    False,
                    stage="validation",
                ) from None
            if _claimed_supported_type(declared_mime, original_filename):
                raise
            raise V1Error(
                415,
                "UNSUPPORTED_FILE_TYPE",
                "The actual document type is not supported.",
                False,
                stage="validation",
            ) from None

    _validate_declared_type(detected, declared_mime, original_filename)
    return detected


def _validate_declared_type(
    detected: DetectedDocument, declared_mime: str, original_filename: str | None
) -> None:
    if declared_mime and declared_mime != "application/octet-stream":
        if detected.format_name == "PDF":
            mime_matches = declared_mime == "application/pdf"
        else:
            mime_matches = declared_mime in IMAGE_FORMAT_TO_MIMES[detected.format_name]
        if not mime_matches:
            raise V1Error(
                415,
                "MIME_MISMATCH",
                "The declared MIME type does not match the actual document type.",
                False,
                stage="validation",
            )

    extension_claim = _filename_claim(original_filename)
    if extension_claim is not None and extension_claim != detected.format_name:
        raise V1Error(
            415,
            "MIME_MISMATCH",
            "The filename extension does not match the actual document type.",
            False,
            stage="validation",
        )


def _layout_page(page: fitz.Page, page_number: int) -> dict[str, Any]:
    rect = page.rect
    full_text_chunks: list[str] = []
    text_blocks: list[dict[str, Any]] = []
    for block_index, block in enumerate(page.get_text("blocks") or []):
        x0, y0, x1, y1, text, *_ = block
        clean_text = (text or "").strip()
        if not clean_text:
            continue
        full_text_chunks.append(clean_text)
        text_blocks.append(
            {
                "id": f"page_{page_number}_block_{block_index + 1}",
                "text": clean_text,
                "x": round(float(x0), 2),
                "y": round(float(y0), 2),
                "width": round(float(x1 - x0), 2),
                "height": round(float(y1 - y0), 2),
                "page_number": page_number,
            }
        )

    text_spans: list[dict[str, Any]] = []
    span_index = 0
    for block in (page.get_text("dict") or {}).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                clean_text = (span.get("text") or "").strip()
                if not clean_text:
                    continue
                x0, y0, x1, y1 = span.get("bbox", [0, 0, 0, 0])
                span_index += 1
                text_spans.append(
                    {
                        "id": f"page_{page_number}_span_{span_index}",
                        "text": clean_text,
                        "x": round(float(x0), 2),
                        "y": round(float(y0), 2),
                        "width": round(float(x1 - x0), 2),
                        "height": round(float(y1 - y0), 2),
                        "page_number": page_number,
                    }
                )

    return {
        "page_number": page_number,
        "mode": "layout",
        "width": round(float(rect.width), 2),
        "height": round(float(rect.height), 2),
        "text": "\n\n".join(full_text_chunks).strip(),
        "text_blocks": text_blocks,
        "text_spans": text_spans,
    }


def render_pdf_to_images(pdf_path: str, output_dir: str) -> list[str]:
    prefix = os.path.join(output_dir, "page")
    command = [
        "pdftoppm",
        "-png",
        "-r",
        str(PDF_RENDER_DPI),
        pdf_path,
        prefix,
    ]
    try:
        subprocess.run(
            command,
            check=True,
            timeout=PDF_CONVERSION_TIMEOUT_SECONDS,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise PdfConversionTimeout from None
    except (subprocess.CalledProcessError, OSError):
        raise PdfConversionFailure from None

    page_number_pattern = re.compile(r"page-(\d+)\.png$")
    numbered_images: list[tuple[int, str]] = []
    for name in os.listdir(output_dir):
        match = page_number_pattern.fullmatch(name)
        if match:
            numbered_images.append((int(match.group(1)), os.path.join(output_dir, name)))
    return [path for _, path in sorted(numbered_images)]


def _ocr_pillow_image(image: Image.Image, language: str) -> str:
    try:
        text = pytesseract.image_to_string(
            image, lang=language, timeout=OCR_TIMEOUT_SECONDS
        )
    except RuntimeError as exc:
        if "timeout" in str(exc).lower():
            raise OcrTimeout from None
        raise OcrFailure from None
    except Exception:
        raise OcrFailure from None
    return (text or "").strip()


def ocr_image_file(image_path: str, language: str) -> str:
    try:
        with Image.open(image_path) as image:
            image.load()
            normalized = image.convert("RGB")
            try:
                return _ocr_pillow_image(normalized, language)
            finally:
                normalized.close()
    except (OcrTimeout, OcrFailure):
        raise
    except Exception:
        raise OcrFailure from None


def ocr_image_document(image_path: str, language: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    try:
        with Image.open(image_path) as image:
            frame_count = int(getattr(image, "n_frames", 1))
            for frame_index in range(frame_count):
                image.seek(frame_index)
                normalized = image.convert("RGB")
                try:
                    text = _ocr_pillow_image(normalized, language)
                finally:
                    normalized.close()
                pages.append(
                    {
                        "page_number": frame_index + 1,
                        "mode": "ocr",
                        "text": text,
                    }
                )
    except (OcrTimeout, OcrFailure):
        raise
    except Exception:
        raise OcrFailure from None
    return pages


@lru_cache(maxsize=1)
def tesseract_version() -> str:
    try:
        result = subprocess.run(
            ["tesseract", "--version"],
            check=True,
            timeout=ENGINE_VERSION_TIMEOUT_SECONDS,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        first_line = result.stdout.splitlines()[0]
        return first_line.removeprefix("tesseract ").strip() or "unknown"
    except (subprocess.SubprocessError, OSError, IndexError):
        return "unknown"


def _process_pdf(path: str, page_count: int, language: str) -> dict[str, Any]:
    try:
        with fitz.open(path) as document:
            pages = [
                _layout_page(page, page_index + 1)
                for page_index, page in enumerate(document)
            ]
    except Exception:
        raise V1Error(
            500,
            "INTERNAL_EXTRACTION_ERROR",
            "The PDF layout extraction failed.",
            True,
            stage="layout",
        ) from None

    pages_requiring_ocr = [page for page in pages if not page["text"]]
    warnings_out: list[str] = []
    if not pages_requiring_ocr:
        return {
            "mode": "layout",
            "text": "\n\n".join(page["text"] for page in pages).strip(),
            "pages": pages,
            "page_count": page_count,
            "warnings": warnings_out,
            "engine": "pymupdf",
            "engine_version": str(fitz.VersionBind),
        }

    with tempfile.TemporaryDirectory(prefix="document-ocr-render-") as image_dir:
        try:
            image_paths = render_pdf_to_images(path, image_dir)
        except PdfConversionTimeout:
            raise V1Error(
                504,
                "PDF_CONVERSION_TIMEOUT",
                "PDF rendering exceeded the configured timeout.",
                True,
                stage="pdf_conversion",
            ) from None
        except PdfConversionFailure:
            raise V1Error(
                500,
                "INTERNAL_EXTRACTION_ERROR",
                "PDF rendering failed.",
                True,
                stage="pdf_conversion",
            ) from None

        if len(image_paths) != page_count:
            raise V1Error(
                422,
                "NO_IMAGES_FROM_PDF",
                "PDF rendering did not produce the expected page images.",
                False,
                stage="pdf_conversion",
            )

        empty_ocr_page = False
        for page in pages_requiring_ocr:
            page_number = page["page_number"]
            try:
                ocr_text = ocr_image_file(image_paths[page_number - 1], language)
            except OcrTimeout:
                raise V1Error(
                    504,
                    "OCR_TIMEOUT",
                    "OCR processing exceeded the configured timeout.",
                    True,
                    stage="ocr",
                ) from None
            except OcrFailure:
                raise V1Error(
                    500,
                    "INTERNAL_EXTRACTION_ERROR",
                    "OCR processing failed.",
                    True,
                    stage="ocr",
                ) from None
            page.clear()
            page.update(
                {
                    "page_number": page_number,
                    "mode": "ocr",
                    "text": ocr_text,
                }
            )
            if not ocr_text:
                empty_ocr_page = True

    full_text = "\n\n".join(page["text"] for page in pages if page["text"]).strip()
    if not full_text:
        raise V1Error(
            422,
            "OCR_EMPTY",
            "No readable text was extracted from the document.",
            False,
            stage="ocr",
        )

    warnings_out.append("OCR_TEXT_ONLY_NO_COORDINATES")
    if len(pages_requiring_ocr) != page_count:
        warnings_out.append("MIXED_PDF_NATIVE_AND_OCR")
    if empty_ocr_page:
        warnings_out.append("OCR_EMPTY_PAGES")
    return {
        "mode": "ocr",
        "text": full_text,
        "pages": pages,
        "page_count": page_count,
        "warnings": warnings_out,
        "engine": "pymupdf+tesseract"
        if len(pages_requiring_ocr) != page_count
        else "tesseract",
        "engine_version": (
            f"PyMuPDF {fitz.VersionBind}; Tesseract {tesseract_version()}"
            if len(pages_requiring_ocr) != page_count
            else tesseract_version()
        ),
    }


def _process_image(path: str, page_count: int, language: str) -> dict[str, Any]:
    try:
        pages = ocr_image_document(path, language)
    except OcrTimeout:
        raise V1Error(
            504,
            "OCR_TIMEOUT",
            "OCR processing exceeded the configured timeout.",
            True,
            stage="ocr",
        ) from None
    except OcrFailure:
        raise V1Error(
            500,
            "INTERNAL_EXTRACTION_ERROR",
            "OCR processing failed.",
            True,
            stage="ocr",
        ) from None

    full_text = "\n\n".join(page["text"] for page in pages if page["text"]).strip()
    if not full_text:
        raise V1Error(
            422,
            "OCR_EMPTY",
            "No readable text was extracted from the document.",
            False,
            stage="ocr",
        )
    return {
        "mode": "ocr",
        "text": full_text,
        "pages": pages,
        "page_count": page_count,
        "warnings": ["OCR_TEXT_ONLY_NO_COORDINATES"],
        "engine": "tesseract",
        "engine_version": tesseract_version(),
    }


def process_document(
    path: str, detected: DetectedDocument, language: str
) -> dict[str, Any]:
    if detected.file_type == "pdf":
        return _process_pdf(path, detected.page_count, language)
    return _process_image(path, detected.page_count, language)


@router.post("/v1/extract")
async def extract_v1(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
    lang: str = Form(default="fra+eng"),
) -> dict[str, Any]:
    request_id = _ensure_request_context(request)
    response.headers["X-Request-ID"] = request_id

    if lang not in ALLOWED_LANGUAGES:
        raise V1Error(
            422,
            "INVALID_LANGUAGE",
            "The requested OCR language is not supported.",
            False,
            stage="request_validation",
        )

    declared_mime = _safe_declared_mime(file)
    try:
        with tempfile.TemporaryDirectory(prefix="document-ocr-request-") as temp_dir:
            upload_path = os.path.join(temp_dir, "document.bin")
            size_bytes = await _store_upload(file, upload_path)
            request.state.size_bytes = size_bytes
            detected = await run_in_threadpool(
                detect_document, upload_path, declared_mime, file.filename
            )
            request.state.file_type = detected.file_type
            extracted = await run_in_threadpool(
                process_document, upload_path, detected, lang
            )
    except V1Error:
        raise
    except Exception:
        raise V1Error(
            500,
            "INTERNAL_EXTRACTION_ERROR",
            "Document extraction failed unexpectedly.",
            True,
            stage="internal",
        ) from None
    finally:
        await file.close()

    duration_ms = _duration_ms(request)
    result = {
        "ok": True,
        "mode": extracted["mode"],
        "file_type": detected.file_type,
        "text": extracted["text"],
        "pages": extracted["pages"],
        "page_count": extracted["page_count"],
        "warnings": extracted["warnings"],
        "engine": extracted["engine"],
        "engine_version": extracted["engine_version"],
        "duration_ms": duration_ms,
    }
    logger.info(
        "document_ocr request_id=%s result=success mode=%s file_type=%s "
        "page_count=%s size_bytes=%s duration_ms=%s",
        request_id,
        result["mode"],
        result["file_type"],
        result["page_count"],
        size_bytes,
        duration_ms,
    )
    return result
