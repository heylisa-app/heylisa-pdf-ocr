# HeyLisa PDF OCR

FastAPI service for native PDF layout extraction and Tesseract OCR. It remains
a document reader: it does not parse invoices, receipts, accounting fields, or
other business data.

## Routes

- `GET /` and `GET /health` keep the historical public contract.
- `POST /extract` is the unchanged SmartFormPDF-compatible legacy route.
- `POST /v1/extract` is the authenticated, versioned document-reader contract.

`POST /v1/extract` expects a multipart `file`, plus an optional `lang` restricted
to `fra`, `eng`, `fra+eng`, or `eng+fra`. It requires
`Authorization: Bearer <token>` and fails closed when
`DOCUMENT_OCR_SERVICE_TOKEN` is absent.

The v1 route accepts valid PDF, JPEG, PNG, WEBP, BMP, TIFF, and HEIC/HEIF content. The
actual bytes are validated with PyMuPDF or Pillow; the multipart MIME and
filename extension are only consistency claims. HEIC/HEIF is decoded in memory
through `pillow-heif`, then normalized to RGB for the existing Tesseract path.

Success responses contain `ok`, `mode`, `file_type`, `text`, `pages`,
`page_count`, `warnings`, `engine`, `engine_version`, and `duration_ms`. OCR
pages contain text and page association only; coordinates and confidence are
not fabricated.

Failures use a non-2xx HTTP status and this stable envelope:

```json
{
  "ok": false,
  "error_code": "INVALID_DOCUMENT",
  "message": "The uploaded document is invalid.",
  "retryable": false
}
```

## Limits and timeouts

Defaults are centralized and may be changed at process startup:

- `DOCUMENT_OCR_MAX_FILE_BYTES`: 15 MiB
- `DOCUMENT_OCR_MAX_PDF_PAGES`: 20
- `DOCUMENT_OCR_MAX_IMAGE_FRAMES`: 20
- `DOCUMENT_OCR_MAX_IMAGE_PIXELS`: 40 million cumulative decoded pixels
- `DOCUMENT_OCR_PDF_TIMEOUT_SECONDS`: 120 seconds per Poppler conversion
- `DOCUMENT_OCR_OCR_TIMEOUT_SECONDS`: 60 seconds per Tesseract page/frame

Uploads are copied by chunks to an isolated temporary directory under a
server-generated filename, then removed after the request. Poppler and
Tesseract subprocesses have explicit timeouts. Pillow has no cancellable native
decode timeout, so image work is bounded by byte, frame, and decoded-pixel caps.
Process isolation for a hard Pillow wall-clock deadline remains a follow-up.

Mixed PDFs are handled page by page when a page has no native text: native
pages retain PyMuPDF layout while blank-text pages go through OCR. A page that
contains sparse native text over a scanned background is still treated as a
layout page; improving that heuristic remains a follow-up.

## Local validation

Install test-only dependencies separately from the production image:

```sh
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q
```
