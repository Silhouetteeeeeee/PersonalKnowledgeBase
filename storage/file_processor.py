"""文件处理模块：从各类文件中提取文字内容。支持 PDF、DOCX、TXT/MD、图片（PaddleOCR）。"""

import hashlib
import logging
import os

logger = logging.getLogger(__name__)

# ── OCR 引擎（PaddleOCR 单例，懒加载）──

_ocr = None


def _get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        logger.info("Loading PaddleOCR (Chinese model)...")
        _ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False, show_log=False)
        logger.info("PaddleOCR loaded")
    return _ocr


def extract_text_from_file(file_path: str) -> str:
    """根据文件扩展名提取文字内容。"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return _extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return _extract_text_from_docx(file_path)
    elif ext in (".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".xml", ".html"):
        return _extract_text_from_txt(file_path)
    elif ext in (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"):
        return _extract_text_from_image(file_path)
    else:
        logger.warning("Unsupported file type: %s", ext)
        return ""


def _extract_text_from_pdf(file_path: str) -> str:
    """使用 PyMuPDF 提取 PDF 文字。"""
    import fitz
    text_parts = []
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text = page.get_text()
            if text.strip():
                text_parts.append(text)
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        logger.error("PDF extraction failed for %s: %s", file_path, e)
        raise


def _extract_text_from_docx(file_path: str) -> str:
    """使用 python-docx 提取 DOCX 文字。"""
    from docx import Document
    try:
        doc = Document(file_path)
        text_parts = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(text_parts)
    except Exception as e:
        logger.error("DOCX extraction failed for %s: %s", file_path, e)
        raise


def _extract_text_from_txt(file_path: str) -> str:
    """直接读取文本文件，自动检测编码。"""
    for encoding in ("utf-8", "gbk", "gb2312"):
        try:
            with open(file_path, "r", encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Unable to decode {file_path} with any supported encoding")


def _extract_text_from_image(file_path: str) -> str:
    """使用 PaddleOCR 提取图片文字。"""
    ocr = _get_ocr()
    try:
        result = ocr.ocr(file_path, cls=True)
        text_parts = []
        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                text_parts.append(text)
        text = "\n".join(text_parts)
        logger.info("OCR extracted %d chars from %s", len(text), os.path.basename(file_path))
        return text
    except Exception as e:
        logger.error("OCR failed for %s: %s", file_path, e)
        raise


def compute_file_hash(file_bytes: bytes) -> str:
    """计算文件 MD5 哈希。"""
    return hashlib.md5(file_bytes).hexdigest()
