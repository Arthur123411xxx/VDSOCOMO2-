"""
ocr.py - PDF vers images, prétraitement et OCR
Version V2: auto-retry + sélection PSM + passe dédiée chiffres (virgules)
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# Vérification de Tesseract
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
except Exception as e:
    TESSERACT_AVAILABLE = False
    TESSERACT_ERROR = str(e)

# Vérification de pdf2image/poppler
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except Exception as e:
    PDF2IMAGE_AVAILABLE = False
    PDF2IMAGE_ERROR = str(e)

# OpenCV pour prétraitement
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


def check_dependencies() -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not TESSERACT_AVAILABLE:
        errors.append(
            f"""
❌ Tesseract OCR non trouvé!

Installation:
- Windows: Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki
  Ajouter au PATH: C:\\Program Files\\Tesseract-OCR

- macOS: brew install tesseract tesseract-lang

- Linux: sudo apt-get install tesseract-ocr tesseract-ocr-fra

Erreur: {TESSERACT_ERROR}
"""
        )

    if not PDF2IMAGE_AVAILABLE:
        errors.append(
            f"""
❌ Poppler non trouvé (requis pour pdf2image)!

Installation:
- Windows: Télécharger depuis https://github.com/osborn/poppler-windows/releases
  Ajouter au PATH le dossier bin/

- macOS: brew install poppler

- Linux: sudo apt-get install poppler-utils

Erreur: {PDF2IMAGE_ERROR}
"""
        )

    return len(errors) == 0, errors


def _pil_to_gray_cv(image: Image.Image) -> np.ndarray:
    """PIL -> OpenCV gray."""
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    return gray


def preprocess_image(image: Image.Image, enhanced: bool = False) -> Image.Image:
    """
    Prétraitement "texte" (binarisation).
    """
    if not CV2_AVAILABLE:
        return image.convert("L")

    gray = _pil_to_gray_cv(image)

    if enhanced:
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)

        binary = cv2.adaptiveThreshold(
            contrast,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        coords = np.column_stack(np.where(binary < 255))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if 0.5 < abs(angle) < 10:
                (h, w) = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(
                    binary,
                    M,
                    (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )

        result = binary
    else:
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return Image.fromarray(result)


def preprocess_image_numbers(image: Image.Image, scale: float = 2.0) -> Image.Image:
    """
    Prétraitement "chiffres": conserve le gris + upscale + sharpen.
    Objectif: mieux garder les ponctuations petites (virgule/point).
    """
    if not CV2_AVAILABLE:
        w, h = image.size
        return image.convert("L").resize((int(w * scale), int(h * scale)))

    gray = _pil_to_gray_cv(image)

    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)

    sharp = cv2.bilateralFilter(sharp, d=5, sigmaColor=50, sigmaSpace=50)

    return Image.fromarray(sharp)


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError("pdf2image/poppler non disponible")
    return convert_from_path(pdf_path, dpi=dpi)


def _safe_tesseract(image: Image.Image, lang: str, config: str) -> str:
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("Tesseract non disponible")

    try:
        return pytesseract.image_to_string(image, lang=lang, config=config)
    except Exception:
        pass

    try:
        return pytesseract.image_to_string(image, lang="fra+eng", config=config)
    except Exception:
        pass

    return pytesseract.image_to_string(image, lang="eng", config=config)


def ocr_best(
    image: Image.Image,
    lang: str = "fra+eng",
    configs: Optional[List[str]] = None,
) -> Tuple[str, dict]:
    """
    OCR "texte": essaie plusieurs PSM et garde le meilleur via calculate_ocr_quality()
    """
    from utils import calculate_ocr_quality

    if configs is None:
        configs = [
            r"--oem 3 --psm 6 -c preserve_interword_spaces=1",
            r"--oem 3 --psm 4 -c preserve_interword_spaces=1",
            r"--oem 3 --psm 11 -c preserve_interword_spaces=1",
            r"--oem 3 --psm 3 -c preserve_interword_spaces=1",
        ]

    best_text = ""
    best_q = {"quality_score": -1, "length": 0}
    best_cfg = configs[0]

    for cfg in configs:
        try:
            txt = _safe_tesseract(image, lang=lang, config=cfg)
        except Exception:
            txt = ""

        q = calculate_ocr_quality(txt)
        score = q.get("quality_score", 0) or 0
        length = q.get("length", 0) or 0

        if (score > (best_q.get("quality_score", -1) or -1)) or (
            score == (best_q.get("quality_score", -1) or -1)
            and length > (best_q.get("length", 0) or 0)
        ):
            best_text = txt
            best_q = q
            best_cfg = cfg

    best_q = dict(best_q)
    best_q["tess_config"] = best_cfg
    return best_text, best_q


def ocr_numbers_best(
    image: Image.Image,
    configs: Optional[List[str]] = None,
) -> Tuple[str, dict]:
    """
    OCR "chiffres": whitelist digits + ponctuation.
    Scoring basé sur le nombre de motifs "décimaux" détectés.
    """
    if configs is None:
        base = (
            r"-c tessedit_char_whitelist=0123456789,.- "
            r"-c classify_bln_numeric_mode=1 "
            r"-c preserve_interword_spaces=1 "
            r"-c load_system_dawg=0 -c load_freq_dawg=0 "
        )
        configs = [
            rf"--oem 3 --psm 6 {base}",
            rf"--oem 3 --psm 7 {base}",
            rf"--oem 3 --psm 11 {base}",
            rf"--oem 3 --psm 4 {base}",
        ]

    best_text = ""
    best = {"match_count": -1, "length": 0}
    best_cfg = configs[0]

    dec_pat = re.compile(r"\b\d{1,3}(?:[ .]\d{3})*[.,]\d{1,2}\b")

    for cfg in configs:
        try:
            txt = _safe_tesseract(image, lang="eng", config=cfg)
        except Exception:
            txt = ""

        matches = dec_pat.findall(txt)
        score = len(matches)
        length = len(txt.strip())

        if (score > (best.get("match_count", -1) or -1)) or (
            score == (best.get("match_count", -1) or -1) and length > (best.get("length", 0) or 0)
        ):
            best_text = txt
            best = {"match_count": score, "length": length, "matches_preview": matches[:10]}
            best_cfg = cfg

    best["tess_config"] = best_cfg
    return best_text, best


def process_pdf(
    pdf_path: str,
    enhanced_preprocessing: bool = False,
    dpi: int = 350,
    lang: str = "fra+eng",
    min_quality: float = 0.6,
    numbers_pass: bool = True,
) -> dict:
    """
    - OCR texte (auto standard->enhanced si qualité faible)
    - + OCR chiffres (virgules) en passe dédiée (optionnel)
    """
    result = {
        "filename": Path(pdf_path).name,
        "pages": [],
        "full_text": "",
        "page_count": 0,
        "errors": [],
    }

    try:
        images = pdf_to_images(pdf_path, dpi=dpi)
        result["page_count"] = len(images)
        full_texts: List[str] = []

        for idx, img in enumerate(images):
            page_result = {
                "page_index": idx,
                "image": img,
                "text": "",
                "preprocessed_image": None,
                "quality": {},
                "numbers_text": "",
                "numbers_quality": {},
            }

            try:
                if enhanced_preprocessing:
                    preprocessed = preprocess_image(img, enhanced=True)
                    text, quality = ocr_best(preprocessed, lang=lang)
                    quality["auto_retry_used"] = "forced_enhanced"
                else:
                    pre_std = preprocess_image(img, enhanced=False)
                    text_std, q_std = ocr_best(pre_std, lang=lang)

                    warning = q_std.get("warning")
                    score = q_std.get("quality_score", 0) or 0
                    need_retry = (warning in ("OCR_VIDE", "OCR_FAIBLE", "OCR_BRUIT")) or (score < min_quality)

                    if need_retry:
                        pre_enh = preprocess_image(img, enhanced=True)
                        text_enh, q_enh = ocr_best(pre_enh, lang=lang)

                        score_std = q_std.get("quality_score", 0) or 0
                        score_enh = q_enh.get("quality_score", 0) or 0
                        len_std = q_std.get("length", 0) or 0
                        len_enh = q_enh.get("length", 0) or 0

                        better = (score_enh > score_std) or (score_enh == score_std and len_enh > len_std)

                        if better:
                            preprocessed, text, quality = pre_enh, text_enh, q_enh
                            quality["auto_retry_used"] = "enhanced"
                            quality["retry_reason"] = warning or f"quality<{min_quality}"
                        else:
                            preprocessed, text, quality = pre_std, text_std, q_std
                            quality["auto_retry_used"] = "standard"
                            quality["retry_reason"] = warning or f"quality<{min_quality}"
                    else:
                        preprocessed, text, quality = pre_std, text_std, q_std
                        quality["auto_retry_used"] = "none"

                page_result["preprocessed_image"] = preprocessed
                page_result["text"] = text
                page_result["quality"] = quality
                full_texts.append(text)

                if numbers_pass:
                    num_img = preprocess_image_numbers(img, scale=2.0)
                    num_txt, num_q = ocr_numbers_best(num_img)
                    page_result["numbers_text"] = num_txt
                    page_result["numbers_quality"] = num_q

            except Exception as e:
                page_result["error"] = str(e)
                result["errors"].append(f"Page {idx + 1}: {str(e)}")

            result["pages"].append(page_result)

        result["full_text"] = "\n\n--- PAGE BREAK ---\n\n".join(full_texts)

    except Exception as e:
        result["errors"].append(f"Erreur PDF: {str(e)}")

    return result


def save_temp_image(image: Image.Image, prefix: str = "ocr_preview") -> str:
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{prefix}_{id(image)}.png")
    image.save(temp_path)
    return temp_path