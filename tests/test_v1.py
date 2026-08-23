from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import document_ocr_v1 as v1
import main as legacy
from main import app


TOKEN = "synthetic-test-token"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DOCUMENT_OCR_SERVICE_TOKEN", TOKEN)
    with TestClient(app) as test_client:
        yield test_client


def auth_headers(token: str = TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def pdf_bytes(page_texts: list[str]) -> bytes:
    document = fitz.open()
    try:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        return document.tobytes()
    finally:
        document.close()


def image_bytes(format_name: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (48, 32), "white").save(output, format=format_name)
    return output.getvalue()


def heic_bytes(*, frames: int = 1, size: tuple[int, int] = (48, 32)) -> bytes:
    output = io.BytesIO()
    images = [Image.new("RGB", size, "white") for _ in range(frames)]
    try:
        images[0].save(
            output,
            format="HEIF",
            save_all=frames > 1,
            append_images=images[1:],
        )
        return output.getvalue()
    finally:
        for image in images:
            image.close()


def upload(
    client: TestClient,
    content: bytes,
    *,
    filename: str,
    mime: str,
    headers: dict[str, str] | None = None,
):
    return client.post(
        "/v1/extract",
        files={"file": (filename, content, mime)},
        headers=headers if headers is not None else auth_headers(),
    )


def assert_error(response, status: int, code: str, retryable: bool) -> None:
    assert response.status_code == status
    assert response.json() == {
        "ok": False,
        "error_code": code,
        "message": response.json()["message"],
        "retryable": retryable,
    }
    assert response.json()["message"]
    assert response.headers["x-request-id"]


def test_health_remains_public_and_minimal(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_extract_requires_authorization(client: TestClient) -> None:
    response = upload(
        client,
        pdf_bytes(["hello"]),
        filename="doc.pdf",
        mime="application/pdf",
        headers={},
    )
    assert_error(response, 401, "AUTH_REQUIRED", False)
    assert response.headers["www-authenticate"] == "Bearer"


def test_auth_is_rejected_before_upload_processing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def must_not_store(*_args, **_kwargs):
        raise AssertionError("upload processing must not run before authentication")

    monkeypatch.setattr(v1, "_store_upload", must_not_store)
    response = upload(
        client,
        pdf_bytes(["hello"]),
        filename="doc.pdf",
        mime="application/pdf",
        headers={},
    )
    assert_error(response, 401, "AUTH_REQUIRED", False)


@pytest.mark.parametrize("authorization", ["Basic nope", "Bearer", "not-bearer"])
def test_extract_rejects_invalid_authorization(
    client: TestClient, authorization: str
) -> None:
    response = upload(
        client,
        pdf_bytes(["hello"]),
        filename="doc.pdf",
        mime="application/pdf",
        headers={"Authorization": authorization},
    )
    assert_error(response, 401, "AUTH_INVALID", False)


def test_extract_rejects_wrong_bearer(client: TestClient) -> None:
    response = upload(
        client,
        pdf_bytes(["hello"]),
        filename="doc.pdf",
        mime="application/pdf",
        headers=auth_headers("wrong-token"),
    )
    assert_error(response, 401, "AUTH_INVALID", False)


def test_extract_fails_closed_when_server_token_is_absent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DOCUMENT_OCR_SERVICE_TOKEN", raising=False)
    response = upload(
        client,
        pdf_bytes(["hello"]),
        filename="doc.pdf",
        mime="application/pdf",
    )
    assert_error(response, 503, "SERVICE_NOT_CONFIGURED", False)


def test_layout_success_has_stable_contract(client: TestClient) -> None:
    response = upload(
        client,
        pdf_bytes(["Synthetic layout text"]),
        filename="doc.pdf",
        mime="application/pdf",
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "ok",
        "mode",
        "file_type",
        "text",
        "pages",
        "page_count",
        "warnings",
        "engine",
        "engine_version",
        "duration_ms",
    }
    assert body["ok"] is True
    assert body["mode"] == "layout"
    assert body["file_type"] == "pdf"
    assert body["text"] == "Synthetic layout text"
    assert body["page_count"] == 1
    assert body["warnings"] == []
    assert body["engine"] == "pymupdf"
    assert body["pages"][0]["mode"] == "layout"
    assert body["pages"][0]["text_blocks"]
    assert body["pages"][0]["text_spans"]
    assert isinstance(body["duration_ms"], int)


def test_pdf_ocr_success_has_text_only_pages(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_render(_pdf_path: str, output_dir: str) -> list[str]:
        image_path = os.path.join(output_dir, "page-1.png")
        Image.new("RGB", (20, 20), "white").save(image_path)
        return [image_path]

    monkeypatch.setattr(v1, "render_pdf_to_images", fake_render)
    monkeypatch.setattr(v1, "ocr_image_file", lambda _path, _lang: "OCR result")
    monkeypatch.setattr(v1, "tesseract_version", lambda: "test-version")

    response = upload(
        client,
        pdf_bytes([""]),
        filename="scan.pdf",
        mime="application/pdf",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "ocr"
    assert body["file_type"] == "pdf"
    assert body["text"] == "OCR result"
    assert body["pages"] == [
        {"page_number": 1, "mode": "ocr", "text": "OCR result"}
    ]
    assert body["warnings"] == ["OCR_TEXT_ONLY_NO_COORDINATES"]
    assert body["engine"] == "tesseract"
    assert body["engine_version"] == "test-version"


def test_mixed_pdf_uses_layout_and_ocr_per_page(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_render(_pdf_path: str, output_dir: str) -> list[str]:
        paths = []
        for page_number in (1, 2):
            image_path = os.path.join(output_dir, f"page-{page_number}.png")
            Image.new("RGB", (20, 20), "white").save(image_path)
            paths.append(image_path)
        return paths

    monkeypatch.setattr(v1, "render_pdf_to_images", fake_render)
    monkeypatch.setattr(v1, "ocr_image_file", lambda _path, _lang: "Scanned page")
    monkeypatch.setattr(v1, "tesseract_version", lambda: "test-version")

    response = upload(
        client,
        pdf_bytes(["Native page", ""]),
        filename="mixed.pdf",
        mime="application/pdf",
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "ocr"
    assert [page["mode"] for page in body["pages"]] == ["layout", "ocr"]
    assert body["text"] == "Native page\n\nScanned page"
    assert body["warnings"] == [
        "OCR_TEXT_ONLY_NO_COORDINATES",
        "MIXED_PDF_NATIVE_AND_OCR",
    ]
    assert body["engine"] == "pymupdf+tesseract"


@pytest.mark.parametrize(
    ("format_name", "filename", "mime"),
    [
        ("JPEG", "image.jpg", "image/jpeg"),
        ("PNG", "image.png", "image/png"),
        ("WEBP", "image.webp", "image/webp"),
        ("BMP", "image.bmp", "image/bmp"),
        ("TIFF", "image.tiff", "image/tiff"),
    ],
)
def test_allowed_image_types(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    filename: str,
    mime: str,
) -> None:
    monkeypatch.setattr(
        v1,
        "ocr_image_document",
        lambda _path, _lang: [{"page_number": 1, "mode": "ocr", "text": "read"}],
    )
    monkeypatch.setattr(v1, "tesseract_version", lambda: "test-version")
    response = upload(
        client,
        image_bytes(format_name),
        filename=filename,
        mime=mime,
    )
    assert response.status_code == 200
    assert response.json()["file_type"] == "image"
    assert response.json()["mode"] == "ocr"


def test_unsupported_type_is_415(client: TestClient) -> None:
    response = upload(
        client,
        b"synthetic executable bytes",
        filename="malware.exe",
        mime="application/octet-stream",
    )
    assert_error(response, 415, "UNSUPPORTED_FILE_TYPE", False)


def test_heic_is_decoded_and_uses_the_existing_ocr_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        v1,
        "ocr_image_document",
        lambda _path, _lang: [{"page_number": 1, "mode": "ocr", "text": "HEIC read"}],
    )
    monkeypatch.setattr(v1, "tesseract_version", lambda: "test-version")
    response = upload(
        client,
        heic_bytes(),
        filename="photo.heic",
        mime="image/heic",
    )
    assert response.status_code == 200
    assert response.json()["file_type"] == "image"
    assert response.json()["text"] == "HEIC read"


def test_corrupt_heic_is_invalid_document(client: TestClient) -> None:
    response = upload(
        client,
        b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00corrupt",
        filename="photo.heic",
        mime="image/heic",
    )
    assert_error(response, 422, "INVALID_DOCUMENT", False)


def test_heic_pixel_limit_is_enforced_after_decode(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v1, "MAX_IMAGE_PIXELS", 100)
    response = upload(
        client,
        heic_bytes(size=(20, 20)),
        filename="photo.heic",
        mime="image/heic",
    )
    assert_error(response, 413, "IMAGE_PIXEL_LIMIT_EXCEEDED", False)


def test_heic_frame_limit_is_enforced(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v1, "MAX_IMAGE_FRAMES", 1)
    response = upload(
        client,
        heic_bytes(frames=2),
        filename="burst.heic",
        mime="image/heic",
    )
    assert_error(response, 413, "IMAGE_FRAME_LIMIT_EXCEEDED", False)


def test_false_mime_is_rejected(client: TestClient) -> None:
    response = upload(
        client,
        image_bytes("PNG"),
        filename="image.png",
        mime="application/pdf",
    )
    assert_error(response, 415, "MIME_MISMATCH", False)


def test_misleading_extension_is_rejected(client: TestClient) -> None:
    response = upload(
        client,
        image_bytes("PNG"),
        filename="image.pdf",
        mime="image/png",
    )
    assert_error(response, 415, "MIME_MISMATCH", False)


def test_file_size_limit_is_checked_before_detection(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v1, "MAX_FILE_SIZE_BYTES", 16)
    response = upload(
        client,
        b"x" * 17,
        filename="doc.pdf",
        mime="application/pdf",
    )
    assert_error(response, 413, "FILE_TOO_LARGE", False)


def test_pdf_page_limit_is_enforced(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v1, "MAX_PDF_PAGES", 2)
    response = upload(
        client,
        pdf_bytes(["one", "two", "three"]),
        filename="doc.pdf",
        mime="application/pdf",
    )
    assert_error(response, 413, "PDF_PAGE_LIMIT_EXCEEDED", False)


@pytest.mark.parametrize(
    "filename",
    ["../../secret.pdf", "/absolute/secret.pdf", "C:\\..\\odd<>name.pdf"],
)
def test_user_filename_never_controls_temp_path(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    filename: str,
) -> None:
    request_dir = tmp_path / "isolated-request"

    class PreservedTemporaryDirectory:
        def __init__(self, *args, **kwargs) -> None:
            request_dir.mkdir(exist_ok=True)

        def __enter__(self) -> str:
            return str(request_dir)

        def __exit__(self, exc_type, exc, traceback) -> bool:
            return False

    monkeypatch.setattr(v1.tempfile, "TemporaryDirectory", PreservedTemporaryDirectory)
    response = upload(
        client,
        pdf_bytes(["Safe synthetic text"]),
        filename=filename,
        mime="application/pdf",
    )
    assert response.status_code == 200
    assert [path.name for path in request_dir.iterdir()] == ["document.bin"]
    assert not (tmp_path / "secret.pdf").exists()


def test_ocr_empty_is_non_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        v1,
        "ocr_image_document",
        lambda _path, _lang: [{"page_number": 1, "mode": "ocr", "text": ""}],
    )
    response = upload(
        client,
        image_bytes("PNG"),
        filename="empty.png",
        mime="image/png",
    )
    assert_error(response, 422, "OCR_EMPTY", False)


def test_no_images_from_pdf_is_non_200(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v1, "render_pdf_to_images", lambda _path, _dir: [])
    response = upload(
        client,
        pdf_bytes([""]),
        filename="scan.pdf",
        mime="application/pdf",
    )
    assert_error(response, 422, "NO_IMAGES_FROM_PDF", False)


def test_poppler_failure_is_structured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_render(_path: str, _directory: str) -> list[str]:
        raise v1.PdfConversionFailure

    monkeypatch.setattr(v1, "render_pdf_to_images", fail_render)
    response = upload(
        client,
        pdf_bytes([""]),
        filename="scan.pdf",
        mime="application/pdf",
    )
    assert_error(response, 500, "INTERNAL_EXTRACTION_ERROR", True)


def test_pdf_conversion_timeout_is_504(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout_render(_path: str, _directory: str) -> list[str]:
        raise v1.PdfConversionTimeout

    monkeypatch.setattr(v1, "render_pdf_to_images", timeout_render)
    response = upload(
        client,
        pdf_bytes([""]),
        filename="scan.pdf",
        mime="application/pdf",
    )
    assert_error(response, 504, "PDF_CONVERSION_TIMEOUT", True)


def test_ocr_failure_is_structured(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_ocr(_path: str, _language: str):
        raise v1.OcrFailure

    monkeypatch.setattr(v1, "ocr_image_document", fail_ocr)
    response = upload(
        client,
        image_bytes("PNG"),
        filename="image.png",
        mime="image/png",
    )
    assert_error(response, 500, "INTERNAL_EXTRACTION_ERROR", True)


def test_ocr_timeout_is_504(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timeout_ocr(_path: str, _language: str):
        raise v1.OcrTimeout

    monkeypatch.setattr(v1, "ocr_image_document", timeout_ocr)
    response = upload(
        client,
        image_bytes("PNG"),
        filename="image.png",
        mime="image/png",
    )
    assert_error(response, 504, "OCR_TIMEOUT", True)


def test_invalid_pdf_is_422(client: TestClient) -> None:
    response = upload(
        client,
        b"%PDF-1.7\nnot a valid PDF",
        filename="broken.pdf",
        mime="application/pdf",
    )
    assert_error(response, 422, "INVALID_DOCUMENT", False)


def test_missing_document_has_stable_v1_error(client: TestClient) -> None:
    response = client.post("/v1/extract", headers=auth_headers())
    assert_error(response, 422, "INVALID_DOCUMENT", False)


def test_sensitive_values_are_not_logged(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="document_ocr.v1")
    secret = "never-log-this-token"
    response = upload(
        client,
        pdf_bytes(["never-log-this-document-text"]),
        filename="never-log-this-filename.pdf",
        mime="application/pdf",
        headers=auth_headers(secret),
    )
    assert response.status_code == 401
    assert secret not in caplog.text
    assert "never-log-this-document-text" not in caplog.text
    assert "never-log-this-filename.pdf" not in caplog.text


def test_legacy_extract_contract_is_unchanged(client: TestClient) -> None:
    assert client.get("/").json() == {"ok": True, "service": "heylisa-pdf-ocr"}
    response = client.post(
        "/extract",
        files={"file": ("legacy.txt", b"legacy", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": False,
        "mode": "unsupported",
        "file_type": "unknown",
        "text": "",
        "error": "UNSUPPORTED_FILE_TYPE:text/plain:legacy.txt",
    }


def test_legacy_layout_success_contract_is_unchanged(client: TestClient) -> None:
    response = client.post(
        "/extract",
        files={
            "file": (
                "legacy.pdf",
                pdf_bytes(["Legacy synthetic text"]),
                "application/pdf",
            )
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "ok",
        "mode",
        "file_type",
        "text",
        "pages",
        "page_count",
        "error",
    }
    assert body["ok"] is True
    assert body["mode"] == "layout"
    assert body["file_type"] == "pdf"
    assert body["text"] == "Legacy synthetic text"
    assert body["error"] is None


def test_legacy_ocr_success_contract_is_unchanged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        legacy,
        "extract_pdf_layout",
        lambda _path: {"text": "", "pages": [], "page_count": 1},
    )
    monkeypatch.setattr(legacy, "pdf_to_images", lambda _path, _directory: ["page.png"])
    monkeypatch.setattr(legacy, "ocr_images", lambda _paths, lang: "Legacy OCR")
    response = client.post(
        "/extract",
        files={"file": ("legacy.pdf", b"legacy-pdf", "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "mode": "ocr",
        "file_type": "pdf",
        "text": "Legacy OCR",
        "error": None,
    }


def test_legacy_pypdf_helper_still_extracts_text(tmp_path: Path) -> None:
    path = tmp_path / "legacy-helper.pdf"
    path.write_bytes(pdf_bytes(["Legacy helper text"]))
    assert legacy.try_text_layer(str(path)) == "Legacy helper text"
