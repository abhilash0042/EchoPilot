"""Offline tests for file and image intake."""
import io
import os
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attachments import (
    AttachmentTooLarge,
    MAX_UPLOAD_BYTES,
    build_user_message,
    extract_upload,
)


def _docx(text: str) -> bytes:
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("word/document.xml", document)
    return buf.getvalue()


class TestAttachments(unittest.TestCase):
    def test_plain_text_file(self):
        extracted = extract_upload("referral.txt", b"Cardiology tomorrow at 10 AM for Ana")
        self.assertEqual(extracted.kind, "file")
        self.assertIn("Cardiology", extracted.text)
        message = build_user_message("please book this", extracted)
        self.assertIn("[Attached file: referral.txt]", message)
        self.assertIn("please book this", message)

    def test_zip_reads_members_and_ignores_path_traversal(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as archive:
            archive.writestr("notes/referral.txt", b"Dermatology on Friday for Priya 5551112222")
            archive.writestr("../secret.txt", b"should not be read")
        extracted = extract_upload("records.zip", buf.getvalue())
        self.assertIn("Dermatology", extracted.text)
        self.assertNotIn("should not be read", extracted.text)
        self.assertIn("skipped", extracted.warning.lower())

    def test_pdf(self):
        stream = b"BT /F1 12 Tf 20 100 Td (Cardiology tomorrow at 10 AM) Tj ET"
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        ]
        pdf = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(pdf))
            pdf += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(pdf)
        pdf += f"xref\n0 {len(objects) + 1}\n".encode()
        pdf += b"0000000000 65535 f \n"
        for offset in offsets[1:]:
            pdf += f"{offset:010d} 00000 n \n".encode()
        pdf += (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
        ).encode()
        extracted = extract_upload("referral.pdf", bytes(pdf))
        self.assertIn("Cardiology", extracted.text)

    def test_docx(self):
        extracted = extract_upload("letter.docx", _docx("Orthopedic evaluation next Monday"))
        self.assertIn("Orthopedic", extracted.text)

    @patch.dict(os.environ, {"GROQ_API_KEY": ""})
    def test_image_is_accepted_without_vision_key(self):
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        extracted = extract_upload("card.png", png, force_image=True)
        self.assertEqual(extracted.kind, "image")
        self.assertTrue(extracted.warning)
        message = build_user_message("", extracted)
        self.assertIn("[Attached image: card.png]", message)

    @patch("speech_engines.transcribe_media_bytes", return_value="Book a cardiology visit tomorrow at 10 AM")
    def test_audio_and_video_use_whisper(self, _transcribe):
        audio = extract_upload("note.mp3", b"ID3" + b"\x00" * 16)
        video = extract_upload("visit.mp4", b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 8)
        self.assertEqual(audio.kind, "audio")
        self.assertEqual(video.kind, "video")
        self.assertIn("cardiology", audio.text)
        self.assertIn("cardiology", video.text)
        self.assertIn("[Attached audio: note.mp3]", build_user_message("", audio))
        self.assertIn("[Attached video: visit.mp4]", build_user_message("", video))

    def test_unknown_binary_is_acknowledged(self):
        extracted = extract_upload("scan.bin", b"\x00\x01\x02\x03" * 20)
        self.assertIn("scan.bin", extracted.warning)
        self.assertEqual(extracted.text, "")

    def test_oversize_rejected(self):
        with self.assertRaises(AttachmentTooLarge):
            extract_upload("big.txt", b"a" * (MAX_UPLOAD_BYTES + 1))


if __name__ == "__main__":
    unittest.main()
