"""
ocr.py - PDF vers images, prétraitement et OCR
"""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

# Vérification de Tesseract
try:
    import pytesseract

    # Test si Tesseract est accessible
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
    """
    Vérifie que toutes les dépendances système sont disponibles
    Retourne: (ok, list_of_errors)
    """
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


def preprocess_image(image: Image.Image, enhanced: bool = False) -> Image.Image:
    """
    Prétraitement de l'image pour améliorer l'OCR

    Args:
        image: Image PIL
        enhanced: Si True, applique un prétraitement renforcé

    Returns:
        Image prétraitée
    """
    if not CV2_AVAILABLE:
        # Sans OpenCV, conversion simple en niveaux de gris
        return image.convert("L")

    # Conversion PIL -> OpenCV
    img_array = np.array(image)

    # Conversion en niveaux de gris si nécessaire
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    if enhanced:
        # Prétraitement renforcé

        # 1. Débruitage
        denoised = cv2.fastNlMeansDenoising(gray, h=10)

        # 2. Amélioration du contraste (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(denoised)

        # 3. Binarisation adaptative
        binary = cv2.adaptiveThreshold(
            contrast,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2,
        )

        # 4. Correction de l'inclinaison (deskew)
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
        # Prétraitement standard: binarisation Otsu
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Conversion OpenCV -> PIL
    return Image.fromarray(result)


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convertit un PDF en liste d'images

    Args:
        pdf_path: Chemin vers le fichier PDF
        dpi: Résolution de conversion

    Returns:
        Liste d'images PIL
    """
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError("pdf2image/poppler non disponible")

    return convert_from_path(pdf_path, dpi=dpi)


def _safe_tesseract(image: Image.Image, lang: str, config: str) -> str:
    """
    OCR avec fallback automatique si une langue Tesseract n'est pas dispo.
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("Tesseract non disponible")

    # Tentative 1: langue demandée
    try:
        return pytesseract.image_to_string(image, lang=lang, config=config)
    except Exception:
        pass

    # Tentative 2: fallback fra+eng
    try:
        return pytesseract.image_to_string(image, lang="fra+eng", config=config)
    except Exception:
        pass

    # Tentative 3: fallback eng seul
    return pytesseract.image_to_string(image, lang="eng", config=config)


def ocr_image(image: Image.Image, lang: str = "fra+eng", config: Optional[str] = None) -> str:
    """
    Effectue l'OCR sur une image (une seule passe).

    Args:
        image: Image PIL
        lang: Langues Tesseract (fra+eng par défaut)
        config: config Tesseract (par défaut optimisée factures)

    Returns:
        Texte extrait
    """
    if config is None:
        # Config par défaut (factures)
        config = r"--oem 3 --psm 6"

    return _safe_tesseract(image, lang=lang, config=config)


def ocr_best(
    image: Image.Image,
    lang: str = "fra+eng",
    configs: Optional[List[str]] = None,
) -> Tuple[str, dict]:
    """
    Tente plusieurs configurations Tesseract (PSM) et garde le meilleur texte
    selon calculate_ocr_quality().

    Returns:
        (best_text, best_quality_dict) avec best_quality_dict['tess_config'].
    """
    from utils import calculate_ocr_quality

    if configs is None:
        configs = [
            r"--oem 3 --psm 6",   # bloc de texte (défaut)
            r"--oem 3 --psm 4",   # colonnes / blocs
            r"--oem 3 --psm 11",  # texte épars
            r"--oem 3 --psm 3",   # auto
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


def process_pdf(
    pdf_path: str,
    enhanced_preprocessing: bool = False,
    dpi: int = 300,
    lang: str = "fra+eng",
    min_quality: float = 0.6,
) -> dict:
    """
    Traite un PDF complet: conversion + prétraitement + OCR.

    Stratégie "Option 1":
    - Passe standard (prétraitement standard + ocr_best)
    - Si OCR faible -> retry (prétraitement renforcé + ocr_best)
    - Garde le meilleur résultat

    Args:
        pdf_path: Chemin vers le PDF
        enhanced_preprocessing: Si True, force le prétraitement renforcé (pas d'auto-retry)
        dpi: Résolution
        lang: Langues OCR
        min_quality: seuil de qualité pour déclencher un retry (si warning ou quality_score < min_quality)

    Returns:
        {
            'filename': str,
            'pages': [{
                'page_index': int,
                'image': Image,
                'text': str,
                'quality': dict
            }],
            'full_text': str,
            'page_count': int,
            'errors': list
        }
    """
    result = {
        "filename": Path(pdf_path).name,
        "pages": [],
        "full_text": "",
        "page_count": 0,
        "errors": [],
    }

    try:
        # Conversion PDF -> images
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
            }

            try:
                if enhanced_preprocessing:
                    # Mode forcé: prétraitement renforcé
                    preprocessed = preprocess_image(img, enhanced=True)
                    text, quality = ocr_best(preprocessed, lang=lang)
                    quality["auto_retry_used"] = "forced_enhanced"
                else:
                    # Mode auto: standard puis retry si faible
                    pre_std = preprocess_image(img, enhanced=False)
                    text_std, q_std = ocr_best(pre_std, lang=lang)

                    warning = q_std.get("warning")
                    score = q_std.get("quality_score", 0) or 0
                    need_retry = (warning in ("OCR_VIDE", "OCR_FAIBLE", "OCR_BRUIT")) or (score < min_quality)

                    if need_retry:
                        pre_enh = preprocess_image(img, enhanced=True)
                        text_enh, q_enh = ocr_best(pre_enh, lang=lang)

                        # Choix du meilleur résultat
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

            except Exception as e:
                page_result["error"] = str(e)
                result["errors"].append(f"Page {idx + 1}: {str(e)}")

            result["pages"].append(page_result)

        result["full_text"] = "\n\n--- PAGE BREAK ---\n\n".join(full_texts)

    except Exception as e:
        result["errors"].append(f"Erreur PDF: {str(e)}")

    return result


def save_temp_image(image: Image.Image, prefix: str = "ocr_preview") -> str:
    """
    Sauvegarde une image temporaire et retourne le chemin
    """
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{prefix}_{id(image)}.png")
    image.save(temp_path)
    return temp_path