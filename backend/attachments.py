"""Turn an uploaded file or image into text the concierge can read.

Archives are opened and their members are read. Nothing in an upload is
executed. Unknown binary files are acknowledged by name instead of being
dumped into the conversation.
"""

from __future__ import annotations

import base64
import io
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_EXTRACT_CHARS = 6000
MAX_ZIP_MEMBERS = 40
MAX_ZIP_UNCOMPRESSED = 20 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_ZIP_DEPTH = 1

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif",
}
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".mpeg", ".mpga", ".flac", ".ogg", ".opus", ".aac", ".m4a",
}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi"}
OFFICE_EXTENSIONS = {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".xml", ".html", ".htm", ".log",
    ".rtf", ".yaml", ".yml", ".ini", ".cfg", ".text", ".eml", ".ics", ".svg",
}


class AttachmentTooLarge(Exception):
    """Raised when an upload exceeds the size cap."""


@dataclass
class ExtractedAttachment:
    filename: str
    kind: str
    text: str = ""
    warning: str = ""
    parts: list[str] = field(default_factory=list)


def safe_filename(filename: str | None) -> str:
    name = os.path.basename((filename or "upload").replace("\\", "/")).strip()
    name = re.sub(r"[\x00-\x1f]", "", name) or "upload"
    return name[:180]


def build_user_message(caption: str, extracted: ExtractedAttachment) -> str:
    """Caption plus whatever could be read from the attachment."""
    lines: list[str] = []
    caption = (caption or "").strip()
    if caption:
        lines.append(caption)
    label = extracted.kind if extracted.kind in {"image", "audio", "video"} else "file"
    lines.append(f"[Attached {label}: {extracted.filename}]")
    if extracted.text.strip():
        lines.append(extracted.text.strip())
    if extracted.warning.strip():
        lines.append(extracted.warning.strip())
    text = "\n".join(lines).strip()
    if len(text) > MAX_EXTRACT_CHARS:
        text = text[:MAX_EXTRACT_CHARS].rstrip() + "\n[Attachment truncated]"
    return text


def extract_upload(
    filename: str | None,
    data: bytes,
    *,
    force_image: bool = False,
    force_audio: bool = False,
    depth: int = 0,
) -> ExtractedAttachment:
    name = safe_filename(filename)
    if len(data) > MAX_UPLOAD_BYTES:
        raise AttachmentTooLarge(f"{name} is larger than 15 MB.")
    if not data:
        kind = "image" if force_image else "file"
        return ExtractedAttachment(name, kind, warning="The attachment is empty.")

    ext = os.path.splitext(name)[1].lower()
    known_text = ext in TEXT_EXTENSIONS or ext in {".pdf", ".rtf"} or ext in OFFICE_EXTENSIONS
    is_zip = ext == ".zip" or (
        data.startswith(b"PK") and ext not in OFFICE_EXTENSIONS and ext not in TEXT_EXTENSIONS and not force_image
    )
    if is_zip and depth > MAX_ZIP_DEPTH:
        return ExtractedAttachment(name, "file", warning=f"Skipped nested archive {name}.")

    if force_audio or ext in AUDIO_EXTENSIONS or ext in VIDEO_EXTENSIONS or _looks_like_audio(data):
        if ext in VIDEO_EXTENSIONS:
            kind = "video"
        else:
            kind = "audio"
        return _extract_media(name, data, kind)

    looks_like_image = ext in IMAGE_EXTENSIONS or (not known_text and _sniff_image_mime(data) is not None)
    if not is_zip and (force_image or looks_like_image):
        return _extract_image(name, data)

    if ext == ".pdf" or data.startswith(b"%PDF"):
        return _extract_pdf(name, data)
    if ext in OFFICE_EXTENSIONS or _is_ooxml(data, ext):
        return _extract_office(name, data, ext)
    if is_zip:
        return _extract_zip(name, data, depth=depth)
    if ext == ".rtf" or data.startswith(b"{\\rtf"):
        return ExtractedAttachment(name, "file", text=_rtf_to_text(data))
    if ext in {".html", ".htm"} or data.lstrip()[:14].lower().startswith(b"<!doctype html") or data.lstrip()[:6].lower().startswith(b"<html"):
        return ExtractedAttachment(name, "file", text=_html_to_text(data))
    if ext in TEXT_EXTENSIONS or _looks_like_text(data):
        return ExtractedAttachment(name, "file", text=_decode_text(data))

    size_kb = max(1, len(data) // 1024)
    return ExtractedAttachment(
        name,
        "file",
        warning=(
            f"Received {name} ({size_kb} KB). "
            "This file type has no readable text. "
            "Zip archives, PDF, Word, spreadsheets, plain text, images, audio, and video are read automatically."
        ),
    )


def _extract_media(name: str, data: bytes, kind: str) -> ExtractedAttachment:
    """Speech in an audio file, or the soundtrack of a video."""
    from speech_engines import transcribe_media_bytes
    from stt import spoken_booking_text

    try:
        text = transcribe_media_bytes(data, name)
    except Exception as exc:
        print(f"[Attachment] {kind} read failed for {name}: {exc}")
        text = ""
    if kind == "audio":
        text = spoken_booking_text(text)
    if text:
        return ExtractedAttachment(name, kind, text=text)
    return ExtractedAttachment(
        name,
        kind,
        warning=f"Received {kind} {name}, but no speech could be transcribed from it.",
    )


def _extract_image(name: str, data: bytes) -> ExtractedAttachment:
    mime = _sniff_image_mime(data)
    ext = os.path.splitext(name)[1].lower()
    if not mime and ext not in IMAGE_EXTENSIONS:
        return ExtractedAttachment(
            name,
            "image",
            warning="That file is not an image. Use the file button for documents and zip archives.",
        )
    if not mime:
        return ExtractedAttachment(
            name,
            "image",
            warning=f"Received {name}, but this image format could not be read. Try PNG, JPG, GIF, or WebP.",
        )
    text = _read_image_text(name, data, mime or "image/jpeg") or _ocr_image(data)
    if text:
        return ExtractedAttachment(name, "image", text=text)
    return ExtractedAttachment(
        name,
        "image",
        warning=f"Received image {name}, but no text could be read from it. Type the details you want used.",
    )


def _ocr_image(data: bytes) -> str:
    """RapidOCR, the same local reader Belfry uses for images."""
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception:
        return ""
    try:
        from image_ocr import read_image_text
        return read_image_text(image)
    except Exception as exc:
        print(f"[Attachment] Local image read failed: {exc}")
        return ""


def _looks_like_audio(data: bytes) -> bool:
    if data.startswith((b"ID3", b"OggS", b"fLaC")):
        return True
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return True
    if len(data) > 11 and data[4:8] == b"ftyp":
        return True
    return False


def _shrink_jpeg(data: bytes, max_side: int = 640) -> bytes:
    """Small JPEG so a vision call is one short request, not a full-size upload."""
    from PIL import Image, ImageOps

    image = Image.open(io.BytesIO(data))
    image.load()
    image = ImageOps.exif_transpose(image) or image
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height, 1))
    if scale < 1:
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BILINEAR,
        )
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=60, optimize=True)
    return buf.getvalue()


def _read_image_text(filename: str, data: bytes, mime: str) -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    model = os.getenv("IMAGE_VISION_MODEL", "").strip()
    if not key or not model:
        return ""
    try:
        jpeg = _shrink_jpeg(data)
    except Exception:
        jpeg = data
        mime = mime or "image/jpeg"
    else:
        mime = "image/jpeg"
    encoded = base64.b64encode(jpeg).decode("ascii")
    try:
        from openai import OpenAI

        client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key, timeout=8.0)
        result = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Read this image for a clinic scheduling assistant. "
                            "Transcribe every visible word, name, phone number, date, time, "
                            "and appointment type. If there is no text, describe the image in one sentence. "
                            "Do not invent details."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }],
            max_tokens=120,
            temperature=0,
        )
        return (result.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[Attachment] Image read failed for {filename}: {exc}")
        return ""


def _extract_pdf(name: str, data: bytes) -> ExtractedAttachment:
    try:
        from pypdf import PdfReader
    except Exception:
        return ExtractedAttachment(
            name,
            "file",
            warning=f"Received PDF {name}. Install pypdf to read the pages.",
        )
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            return ExtractedAttachment(name, "file", warning=f"{name} is a password-protected PDF.")
        pages = []
        for page in reader.pages[:25]:
            page_text = ""
            try:
                page_text = page.extract_text() or ""
            except Exception:
                page_text = ""
            if len(page_text.strip()) < 40:
                page_text = f"{page_text}\n{_ocr_pdf_page(page)}".strip()
            if page_text.strip():
                pages.append(page_text.strip())
            if sum(len(part) for part in pages) > MAX_EXTRACT_CHARS:
                break
        text = "\n".join(pages)
        if not text:
            return ExtractedAttachment(name, "file", warning=f"Received PDF {name}, but no text was found in it.")
        return ExtractedAttachment(name, "file", text=text)
    except Exception as exc:
        print(f"[Attachment] PDF read failed for {name}: {exc}")
        return ExtractedAttachment(name, "file", warning=f"Received PDF {name}, but it could not be read.")


def _ocr_pdf_page(page) -> str:
    """Read words painted on images embedded in a PDF page."""
    try:
        from PIL import Image
        images = list(getattr(page, "images", []) or [])
    except Exception:
        return ""
    chunks = []
    for image in images[:3]:
        raw = getattr(image, "data", None)
        if not raw:
            continue
        try:
            opened = Image.open(io.BytesIO(raw))
            opened.load()
            from image_ocr import read_image_text
            text = read_image_text(opened)
        except Exception:
            continue
        if text.strip():
            chunks.append(text.strip())
    return "\n".join(chunks)


def _extract_office(name: str, data: bytes, ext: str) -> ExtractedAttachment:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            blobs = []
            if "word/document.xml" in names:
                blobs.append(archive.read("word/document.xml"))
            elif "content.xml" in names:
                blobs.append(archive.read("content.xml"))
            else:
                interesting = [
                    item for item in names
                    if item.startswith("xl/") or item.startswith("ppt/slides/")
                ]
                for item in interesting[:30]:
                    if item.endswith(".xml"):
                        blobs.append(archive.read(item))
            text = "\n".join(_xml_text(blob) for blob in blobs if blob)
    except zipfile.BadZipFile:
        return ExtractedAttachment(name, "file", warning=f"{name} looks damaged and could not be opened.")
    text = text.strip()
    if not text:
        return ExtractedAttachment(name, "file", warning=f"Received {name}, but no text was found in it.")
    return ExtractedAttachment(name, "file", text=text)


def _extract_zip(name: str, data: bytes, depth: int) -> ExtractedAttachment:
    if depth > MAX_ZIP_DEPTH:
        return ExtractedAttachment(name, "file", warning=f"Skipped nested archive {name}.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return ExtractedAttachment(name, "file", warning=f"{name} is not a valid zip archive.")

    texts: list[str] = []
    skipped: list[str] = []
    total = 0
    members = 0
    with archive:
        for info in archive.infolist():
            member = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            if member.startswith("/") or ".." in member.split("/"):
                skipped.append(member)
                continue
            members += 1
            if members > MAX_ZIP_MEMBERS:
                skipped.append("further files")
                break
            if info.flag_bits & 0x1:
                skipped.append(f"{member} (encrypted)")
                continue
            if info.file_size > MAX_MEMBER_BYTES or total + info.file_size > MAX_ZIP_UNCOMPRESSED:
                skipped.append(member)
                continue
            total += info.file_size
            try:
                payload = archive.read(info)
            except RuntimeError:
                skipped.append(f"{member} (encrypted)")
                continue
            except Exception:
                skipped.append(member)
                continue
            inner = extract_upload(safe_filename(member), payload, depth=depth + 1)
            if inner.text.strip():
                texts.append(f"--- {member} ---\n{inner.text.strip()}")
            elif inner.kind == "image" and inner.warning:
                texts.append(f"--- {member} ---\n{inner.warning}")
            elif inner.warning:
                skipped.append(member)

    warning = ""
    if skipped:
        shown = ", ".join(skipped[:8])
        warning = f"Some files in {name} were skipped: {shown}."
    if not texts and not warning:
        warning = f"Opened {name}, but none of the files inside had readable text."
    return ExtractedAttachment(name, "file", text="\n\n".join(texts), warning=warning)


def _sniff_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):
        return "image/tiff"
    return None


def _is_ooxml(data: bytes, ext: str) -> bool:
    return ext in OFFICE_EXTENSIONS and data.startswith(b"PK")


def _looks_like_text(data: bytes) -> bool:
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    printable = sum(1 for byte in sample if byte in (9, 10, 13) or 32 <= byte < 127)
    return printable / max(1, len(sample)) > 0.85


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return data.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    return ""


def _xml_text(xml_bytes: bytes) -> str:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return ""
    chunks: list[str] = []
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag in {"t", "a"} and el.text:
            chunks.append(el.text)
        elif tag in {"tab"}:
            chunks.append(" ")
        elif tag in {"br", "p", "tr"}:
            chunks.append("\n")
    return re.sub(r"[ \t]+\n", "\n", "".join(chunks)).strip()


def _html_to_text(data: bytes) -> str:
    raw = _decode_text(data)
    raw = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    raw = (
        raw.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
    )
    return re.sub(r"\s+", " ", raw).strip()


def _rtf_to_text(data: bytes) -> str:
    raw = data.decode("latin-1", errors="ignore")
    raw = re.sub(r"\\'[0-9a-fA-F]{2}", " ", raw)
    raw = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", raw)
    raw = raw.replace("{", " ").replace("}", " ")
    return re.sub(r"\s+", " ", raw).strip()
