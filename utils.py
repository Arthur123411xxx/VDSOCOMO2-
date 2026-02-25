"""
utils.py - Utilitaires pour normalisation, scoring et parsing
"""

import re
from typing import Optional, List, Tuple
from datetime import datetime
from dateutil import parser as date_parser


def normalize_pe(pe_str: str) -> str:
    """
    Normalise un code PE au format PE123456
    Gère les variantes: PE 123456, P E123456, PE-123456, pe123456
    """
    if not pe_str:
        return ""
    
    # Uppercase et suppression des espaces/tirets
    cleaned = pe_str.upper().replace(" ", "").replace("-", "").replace(".", "")
    
    # Extraction du pattern PE + 6 chiffres
    match = re.search(r'PE(\d{6})', cleaned)
    if match:
        return f"PE{match.group(1)}"
    
    return cleaned


def normalize_amount(amount_str: str) -> Optional[float]:
    """
    Normalise un montant en float
    Gère: 1 234,56 € | 1234.56 | 1,234.56 | 1.234,56
    """
    if not amount_str:
        return None
    
    # Suppression des symboles de devise et espaces
    cleaned = amount_str.replace("€", "").replace("EUR", "").replace("$", "").strip()
    cleaned = re.sub(r'\s+', '', cleaned)
    
    # Détection du format (virgule ou point comme décimal)
    # Si contient virgule et point, déterminer lequel est le décimal
    if ',' in cleaned and '.' in cleaned:
        # Format européen: 1.234,56 ou américain: 1,234.56
        if cleaned.rfind(',') > cleaned.rfind('.'):
            # Virgule est le séparateur décimal (européen)
            cleaned = cleaned.replace('.', '').replace(',', '.')
        else:
            # Point est le séparateur décimal (américain)
            cleaned = cleaned.replace(',', '')
    elif ',' in cleaned:
        # Vérifier si c'est un séparateur de milliers ou décimal
        parts = cleaned.split(',')
        if len(parts) == 2 and len(parts[1]) <= 2:
            # Probablement décimal
            cleaned = cleaned.replace(',', '.')
        else:
            # Probablement séparateur de milliers
            cleaned = cleaned.replace(',', '')
    
    try:
        value = float(cleaned)
        return round(value, 2)
    except ValueError:
        return None


def extract_pe_candidates(text: str) -> List[dict]:
    """
    Extrait tous les PE candidats avec scoring
    Retourne: [{pe, score, context_snippet, page_index}]
    """
    candidates = []
    seen_pes = set()
    
    # Pattern strict: PE suivi directement de 6 chiffres
    strict_pattern = r'\bPE(\d{6})\b'
    
    # Pattern tolérant: PE avec espaces/tirets possibles
    tolerant_pattern = r'\bP\s*E\s*[-.]?\s*(\d\s*\d\s*\d\s*\d\s*\d\s*\d)\b'
    
    # Recherche stricte (score élevé)
    for match in re.finditer(strict_pattern, text, re.IGNORECASE):
        pe = f"PE{match.group(1)}"
        if pe not in seen_pes:
            # Contexte autour du match
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].replace('\n', ' ')
            
            # Score bonus si près de mots-clés
            score = 0.9
            context_lower = context.lower()
            if any(kw in context_lower for kw in ['référence', 'reference', 'ref', 'pe:', 'n°pe', 'code pe']):
                score = 1.0
            
            candidates.append({
                'pe': pe,
                'score': score,
                'context_snippet': context,
                'match_type': 'strict'
            })
            seen_pes.add(pe)
    
    # Recherche tolérante (score moyen)
    for match in re.finditer(tolerant_pattern, text, re.IGNORECASE):
        digits = re.sub(r'\s+', '', match.group(1))
        pe = f"PE{digits}"
        if pe not in seen_pes:
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].replace('\n', ' ')
            
            score = 0.7
            context_lower = context.lower()
            if any(kw in context_lower for kw in ['référence', 'reference', 'ref', 'pe:', 'n°pe', 'code pe']):
                score = 0.85
            
            candidates.append({
                'pe': pe,
                'score': score,
                'context_snippet': context,
                'match_type': 'tolerant'
            })
            seen_pes.add(pe)
    
    # Tri par score décroissant
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates


def extract_amount_candidates(text: str) -> List[dict]:
    """
    Extrait les montants candidats avec scoring basé sur les mots-clés
    Priorité: NET A PAYER > TOTAL TTC > TOTAL
    """
    candidates = []
    
    # Patterns de montants avec contexte
    amount_patterns = [
        # Montants avec symbole euro
        r'([\d\s.,]+)\s*€',
        r'€\s*([\d\s.,]+)',
        # Montants avec EUR
        r'([\d\s.,]+)\s*EUR\b',
        r'EUR\s*([\d\s.,]+)',
        # Montants génériques (nombres avec décimales)
        r'([\d]{1,3}(?:[\s.,]?\d{3})*(?:[.,]\d{2}))',
    ]
    
    # Mots-clés par priorité (du plus fiable au moins fiable)
    priority_keywords = [
        (1.0, ['net a payer', 'net à payer', 'à payer', 'a payer', 'grand total', 'montant net']),
        (0.85, ['total ttc', 'total tvac', 'montant ttc', 'total t.t.c']),
        (0.7, ['total général', 'total general', 'montant total', 'total facture']),
        (0.5, ['total', 'montant']),
    ]
    
    text_lower = text.lower()
    lines = text.split('\n')
    
    for line_idx, line in enumerate(lines):
        line_lower = line.lower()
        
        # Chercher les montants dans cette ligne
        for pattern in amount_patterns:
            for match in re.finditer(pattern, line, re.IGNORECASE):
                amount_str = match.group(1) if match.group(1) else match.group(0)
                amount = normalize_amount(amount_str)
                
                if amount is not None and amount > 0:
                    # Déterminer le score basé sur le contexte
                    score = 0.3  # Score par défaut
                    keyword_found = None
                    
                    # Chercher dans les lignes précédentes aussi (jusqu'à 2 lignes)
                    context_lines = []
                    for i in range(max(0, line_idx - 2), line_idx + 1):
                        if i < len(lines):
                            context_lines.append(lines[i].lower())
                    context = ' '.join(context_lines)
                    
                    for priority_score, keywords in priority_keywords:
                        for kw in keywords:
                            if kw in context:
                                if priority_score > score:
                                    score = priority_score
                                    keyword_found = kw
                                break
                        if keyword_found:
                            break
                    
                    candidates.append({
                        'amount': amount,
                        'raw': amount_str.strip(),
                        'score': score,
                        'keyword': keyword_found,
                        'context': line.strip()[:100],
                        'line_index': line_idx
                    })
    
    # Dédupliquer les montants identiques, garder le meilleur score
    unique_candidates = {}
    for c in candidates:
        key = c['amount']
        if key not in unique_candidates or c['score'] > unique_candidates[key]['score']:
            unique_candidates[key] = c
    
    # Tri par score décroissant
    result = sorted(unique_candidates.values(), key=lambda x: x['score'], reverse=True)
    return result


def parse_date(date_str: str) -> Optional[datetime]:
    """
    Parse une date depuis une chaîne avec différents formats
    """
    if not date_str:
        return None
    
    # Formats courants
    formats = [
        '%d/%m/%Y',
        '%d-%m-%Y',
        '%Y-%m-%d',
        '%d.%m.%Y',
        '%d %B %Y',
        '%d %b %Y',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    # Fallback avec dateutil
    try:
        return date_parser.parse(date_str, dayfirst=True)
    except:
        return None


def extract_invoice_number(text: str) -> Optional[str]:
    """
    Extrait le numéro de facture
    """
    patterns = [
        r'facture\s*n[°o]?\s*[:.]?\s*([A-Z0-9-]+)',
        r'invoice\s*n[°o]?\s*[:.]?\s*([A-Z0-9-]+)',
        r'n[°o]\s*facture\s*[:.]?\s*([A-Z0-9-]+)',
        r'numéro\s*[:.]?\s*([A-Z0-9-]+)',
        r'ref\s*[:.]?\s*([A-Z0-9-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return None


def extract_date(text: str) -> Optional[str]:
    """
    Extrait la date de facture
    """
    patterns = [
        r'date\s*[:.]?\s*(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
        r'le\s+(\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})',
        r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def extract_supplier(text: str) -> Optional[str]:
    """
    Extrait le nom du fournisseur (heuristique: premières lignes)
    """
    lines = text.strip().split('\n')
    
    # Chercher dans les 10 premières lignes
    for line in lines[:10]:
        line = line.strip()
        # Ignorer les lignes trop courtes ou trop longues
        if len(line) < 3 or len(line) > 100:
            continue
        # Ignorer les lignes qui ressemblent à des adresses ou numéros
        if re.match(r'^[\d\s\-+.()]+$', line):
            continue
        if any(kw in line.lower() for kw in ['facture', 'invoice', 'date', 'client', '@', 'www.']):
            continue
        # Première ligne significative = probablement le fournisseur
        if re.match(r'^[A-Za-zÀ-ÿ]', line):
            return line[:80]
    
    return None


def detect_currency(text: str) -> str:
    """
    Détecte la devise (EUR par défaut)
    """
    if '€' in text or 'EUR' in text.upper():
        return 'EUR'
    if '$' in text or 'USD' in text.upper():
        return 'USD'
    if '£' in text or 'GBP' in text.upper():
        return 'GBP'
    return 'EUR'


def calculate_ocr_quality(text: str) -> dict:
    """
    Calcule des métriques de qualité OCR
    """
    if not text:
        return {
            'length': 0,
            'alphanum_ratio': 0,
            'token_count': 0,
            'quality_score': 0,
            'warning': 'OCR_VIDE'
        }
    
    length = len(text)
    alphanum = sum(1 for c in text if c.isalnum())
    alphanum_ratio = alphanum / length if length > 0 else 0
    tokens = len(text.split())
    
    # Score de qualité heuristique
    quality_score = 0
    if length > 100:
        quality_score += 0.3
    if length > 500:
        quality_score += 0.2
    if alphanum_ratio > 0.5:
        quality_score += 0.3
    if tokens > 50:
        quality_score += 0.2
    
    warning = None
    if length < 100:
        warning = 'OCR_FAIBLE'
    elif alphanum_ratio < 0.3:
        warning = 'OCR_BRUIT'
    
    return {
        'length': length,
        'alphanum_ratio': round(alphanum_ratio, 2),
        'token_count': tokens,
        'quality_score': round(quality_score, 2),
        'warning': warning
    }


def calculate_delta(invoice_amount: Optional[float], expected_amount: Optional[float]) -> dict:
    """
    Calcule l'écart entre montant facture et montant attendu
    """
    result = {
        'delta': None,
        'delta_abs': None,
        'delta_pct': None,
        'status': 'INCOMPLET'
    }
    
    if invoice_amount is None or expected_amount is None:
        return result
    
    delta = invoice_amount - expected_amount
    delta_abs = abs(delta)
    delta_pct = (delta / expected_amount * 100) if expected_amount != 0 else None
    
    result['delta'] = round(delta, 2)
    result['delta_abs'] = round(delta_abs, 2)
    result['delta_pct'] = round(delta_pct, 2) if delta_pct is not None else None
    
    return result


def check_tolerance(delta_abs: float, delta_pct: Optional[float], 
                   tol_euros: float = 0.05, tol_percent: float = 0.5) -> str:
    """
    Vérifie si l'écart est dans les tolérances
    Retourne: 'OK' ou 'ECART'
    """
    if delta_abs is None:
        return 'INCOMPLET'
    
    # OK si écart absolu <= tolérance € OU écart relatif <= tolérance %
    if delta_abs <= tol_euros:
        return 'OK'
    
    if delta_pct is not None and abs(delta_pct) <= tol_percent:
        return 'OK'
    
    return 'ECART'

import re
import pandas as pd

def parse_eu_number(x):
    """
    "1.234,56" -> 1234.56
    "1 234,56" -> 1234.56
    "1234,56"  -> 1234.56
    "1234.56"  -> 1234.56
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ["nan", "none"]:
        return None

    s = s.replace("€", "").replace("\u00a0", " ")  # espace insécable
    s = s.replace(" ", "")  # enlève séparateur milliers espace

    # EU: "." milliers + "," décimal
    if "." in s and "," in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")

    s = re.sub(r"[^0-9\.\-]", "", s)

    try:
        return float(s)
    except:
        return None

