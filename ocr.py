"""
ocr.py - PDF vers images, prétraitement et OCR
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import shutil

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
    errors = []
    
    if not TESSERACT_AVAILABLE:
        errors.append(f"""
❌ Tesseract OCR non trouvé!

Installation:
- Windows: Télécharger depuis https://github.com/UB-Mannheim/tesseract/wiki
  Ajouter au PATH: C:\\Program Files\\Tesseract-OCR
  
- macOS: brew install tesseract tesseract-lang

- Linux: sudo apt-get install tesseract-ocr tesseract-ocr-fra

Erreur: {TESSERACT_ERROR}
""")
    
    if not PDF2IMAGE_AVAILABLE:
        errors.append(f"""
❌ Poppler non trouvé (requis pour pdf2image)!

Installation:
- Windows: Télécharger depuis https://github.com/osborn/poppler-windows/releases
  Ajouter au PATH le dossier bin/
  
- macOS: brew install poppler

- Linux: sudo apt-get install poppler-utils

Erreur: {PDF2IMAGE_ERROR}
""")
    
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
        return image.convert('L')
    
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
            contrast, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # 4. Correction de l'inclinaison (deskew)
        coords = np.column_stack(np.where(binary < 255))
        if len(coords) > 100:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = 90 + angle
            if abs(angle) > 0.5 and abs(angle) < 10:
                (h, w) = binary.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                binary = cv2.warpAffine(binary, M, (w, h),
                                       flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
        
        result = binary
    else:
        # Prétraitement standard
        # Simple binarisation Otsu
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
    
    images = convert_from_path(pdf_path, dpi=dpi)
    return images


def ocr_image(image: Image.Image, lang: str = 'fra+eng') -> str:
    """
    Effectue l'OCR sur une image
    
    Args:
        image: Image PIL
        lang: Langues Tesseract (fra+eng par défaut)
    
    Returns:
        Texte extrait
    """
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("Tesseract non disponible")
    
    # Configuration Tesseract optimisée pour les factures
    custom_config = r'--oem 3 --psm 6'
    
    text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
    return text


def process_pdf(pdf_path: str, enhanced_preprocessing: bool = False,
               dpi: int = 300, lang: str = 'fra+eng') -> dict:
    """
    Traite un PDF complet: conversion + prétraitement + OCR
    
    Args:
        pdf_path: Chemin vers le PDF
        enhanced_preprocessing: Activer le prétraitement renforcé
        dpi: Résolution
        lang: Langues OCR
    
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
    from utils import calculate_ocr_quality
    
    result = {
        'filename': Path(pdf_path).name,
        'pages': [],
        'full_text': '',
        'page_count': 0,
        'errors': []
    }
    
    try:
        # Conversion PDF -> images
        images = pdf_to_images(pdf_path, dpi=dpi)
        result['page_count'] = len(images)
        
        full_texts = []
        
        for idx, img in enumerate(images):
            page_result = {
                'page_index': idx,
                'image': img,
                'text': '',
                'preprocessed_image': None,
                'quality': {}
            }
            
            try:
                # Prétraitement
                preprocessed = preprocess_image(img, enhanced=enhanced_preprocessing)
                page_result['preprocessed_image'] = preprocessed
                
                # OCR
                text = ocr_image(preprocessed, lang=lang)
                page_result['text'] = text
                
                # Qualité OCR
                page_result['quality'] = calculate_ocr_quality(text)
                
                full_texts.append(text)
                
            except Exception as e:
                page_result['error'] = str(e)
                result['errors'].append(f"Page {idx + 1}: {str(e)}")
            
            result['pages'].append(page_result)
        
        result['full_text'] = '\n\n--- PAGE BREAK ---\n\n'.join(full_texts)
        
    except Exception as e:
        result['errors'].append(f"Erreur PDF: {str(e)}")
    
    return result


def save_temp_image(image: Image.Image, prefix: str = "ocr_preview") -> str:
    """
    Sauvegarde une image temporaire et retourne le chemin
    """
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{prefix}_{id(image)}.png")
    image.save(temp_path)
    return temp_path
