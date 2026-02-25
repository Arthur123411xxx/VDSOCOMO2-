"""
extract.py - Extraction de données des factures (PE, montants, champs bonus)
"""

from typing import Optional, List
from utils import (
    extract_pe_candidates,
    extract_amount_candidates,
    extract_invoice_number,
    extract_date,
    extract_supplier,
    detect_currency,
    normalize_pe
)

def extract_ac_or_am(full_text: str) -> dict:
    """
    Détecte un code type ACxxxx ou AMxxxx dans le texte.
    Retourne:
      - ac      : valeur NORMALISÉE (toujours 'AC' + digits)
      - ac_raw  : valeur BRUTE trouvée (ex: 'AM084594' ou 'AC084594')
      - ac_type : 'AC' ou 'AM' (préfixe détecté)
    """
    t = (full_text or "").replace("\r", "\n")

    patterns = [
        r"\bALBAR[AÁ]N\b.*?\b(A[CM])\s*(\d{4,})\b",      # "ALBARAN ... AC084594" ou "AM 084594"
        r"\b(A[CM])\s*[:\-]?\s*(\d{4,})\b",             # "AC: 084594" / "AM-084594"
        r"\b(A[CM])(\d{4,})\b",                         # "AC084594" / "AM084594"
    ]

    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue

        prefix = (m.group(1) or "").upper().strip()
        digits = (m.group(2) or "").strip()
        if not digits.isdigit():
            continue

        ac_raw = f"{prefix}{digits}"
        ac_norm = f"AC{digits}"   # on normalise en AC pour le workflow
        return {"ac": ac_norm, "ac_raw": ac_raw, "ac_type": prefix}

    return {"ac": "", "ac_raw": "", "ac_type": ""}



def extract_invoice_data(ocr_result: dict, include_images: bool = True) -> dict:
    """
    Extrait toutes les données d'une facture à partir du résultat OCR
    
    Args:
        ocr_result: Résultat de process_pdf()
    
    Returns:
        {
            'filename': str,
            'pe_candidates': list,
            'pe_selected': str or None,
            'pe_status': 'OK' | 'MANQUANT_PE' | 'MULTI_PE',
            'amount_candidates': list,
            'total_facture': float or None,
            'total_status': 'OK' | 'MANQUANT_TOTAL' | 'TOTAL_AMBIGU',
            'supplier': str or None,
            'invoice_number': str or None,
            'invoice_date': str or None,
            'currency': str,
            'warnings': list,
            'pages_data': list
        }
    """
    result = {
        'filename': ocr_result.get('filename', ''),
        'pe_candidates': [],
        'pe_selected': None,
        'pe_status': 'OK',
        'amount_candidates': [],
        'total_facture': None,
        'total_status': 'OK',
        'supplier': None,
        'invoice_number': None,
        'invoice_date': None,
        'currency': 'EUR',
        'warnings': [],
        'pages_data': [],
        'full_text': ocr_result.get('full_text', ''),
        'page_count': ocr_result.get('page_count', 0),
        'flags': [],

    }
    
    full_text = ocr_result.get('full_text', '')
    acinfo = extract_ac_or_am(full_text)
    result["ac"] = acinfo["ac"]
    result["ac_raw"] = acinfo["ac_raw"]
    result["ac_type"] = acinfo["ac_type"]

    if acinfo["ac_type"] == "AM":
        result["flags"].append("AC_LU_COMME_AM")

    # Extraction des lignes articles (multi-lignes)
    result["items"] = extract_item_lines(full_text)
    try:
        cnt_items = (result["items"] or {}).get("count", 0)
        if cnt_items == 0:
            result["warnings"].append("Aucune ligne article détectée (quantités OCR = vides).")
    except Exception:
        pass





    # -----------------------------
    # Quantités OCR: somme par unité
    # -----------------------------
    items_list = (result.get("items") or {}).get("items", []) or []

    qty_by_unit = {}
    qty_found = 0

    for it in items_list:
        q = it.get("quantity")
        u = (it.get("unit") or "").strip().lower()

        if q is None:
            continue

        try:
            qf = float(q)
        except Exception:
            continue

        if not u:
            u = "unknown"

        qty_by_unit[u] = qty_by_unit.get(u, 0.0) + qf
        qty_found += 1

    # Choix d'une unité "prioritaire" (colis/boîtes d'abord)
    preferred_units = ["caja", "colis", "unidad", "kg", "pza", "pz", "unknown"]
    pick = next((uu for uu in preferred_units if uu in qty_by_unit), None)

    result["quantite_by_unit_ocr"] = qty_by_unit
    result["quantite_total_ocr"] = qty_by_unit.get(pick) if pick else None
    result["quantite_unite_ocr"] = pick

    if qty_found == 0 and items_list:
        result["warnings"].append("Lignes articles détectées mais aucune quantité exploitable (regex quantité/unités).")
    if not items_list:
        result["warnings"].append("Aucune ligne article détectée (quantité OCR vide).")

    # Optionnel: garder un "first_item" pour compatibilité UI
    if result["items"].get("items"):
        result["first_item"] = result["items"]["items"][0]
    else:
        result["first_item"] = {"ok": False}

    # Extraction PE
    pe_candidates = []
    for page_data in ocr_result.get('pages', []):
        page_text = page_data.get('text', '')
        page_idx = page_data.get('page_index', 0)
        
        candidates = extract_pe_candidates(page_text)
        for c in candidates:
            c['page_index'] = page_idx
        pe_candidates.extend(candidates)
    
    # Dédupliquer par PE, garder le meilleur score
    unique_pe = {}
    for c in pe_candidates:
        pe = c['pe']
        if pe not in unique_pe or c['score'] > unique_pe[pe]['score']:
            unique_pe[pe] = c
    
    result['pe_candidates'] = list(unique_pe.values())
    
    # Sélection automatique du PE
    if len(result['pe_candidates']) == 0:
        result['pe_status'] = 'MANQUANT_PE'
        result['warnings'].append('Aucun PE détecté')
    elif len(result['pe_candidates']) == 1:
        result['pe_selected'] = result['pe_candidates'][0]['pe']
        result['pe_status'] = 'OK'
    else:
        # Plusieurs PE: prendre le meilleur score
        result['pe_candidates'].sort(key=lambda x: x['score'], reverse=True)
        result['pe_selected'] = result['pe_candidates'][0]['pe']
        result['pe_status'] = 'MULTI_PE'
        result['warnings'].append(f"Plusieurs PE détectés: {[c['pe'] for c in result['pe_candidates']]}")
    
    # Extraction montants
    amount_candidates = extract_amount_candidates(full_text)
    result['amount_candidates'] = amount_candidates
    
    if len(amount_candidates) == 0:
        result['total_status'] = 'MANQUANT_TOTAL'
        result['warnings'].append('Aucun montant détecté')
    else:
        # Sélection du meilleur montant
        best = amount_candidates[0]
        result['total_facture'] = best['amount']
        
        # Vérifier si ambigu (plusieurs montants avec scores similaires)
        if len(amount_candidates) > 1:
            second_best = amount_candidates[1]
            if second_best['score'] >= 0.7 * best['score'] and second_best['amount'] != best['amount']:
                result['total_status'] = 'TOTAL_AMBIGU'
                result['warnings'].append(
                    f"Ambiguïté: {best['amount']}€ (score {best['score']}) vs {second_best['amount']}€ (score {second_best['score']})"
                )
            else:
                result['total_status'] = 'OK'
        else:
            result['total_status'] = 'OK'
    
    # Champs bonus
    result['supplier'] = extract_supplier(full_text)
    result['invoice_number'] = extract_invoice_number(full_text)
    result['invoice_date'] = extract_date(full_text)
    result['currency'] = detect_currency(full_text)
    
    # Données par page (pour l'UI)
    for page_data in ocr_result.get('pages', []):
        page_idx = page_data.get('page_index', 0)
        result['pages_data'].append({
            'page_index': page_idx,
            'text': page_data.get('text', ''),
            'quality': page_data.get('quality', {}),
            'image': page_data.get('image') if include_images else None,
            'preprocessed_image': page_data.get('preprocessed_image') if include_images else None,
            'pe_on_page': [c for c in result['pe_candidates'] if c.get('page_index') == page_idx]
        })
    
    return result


def apply_manual_correction(invoice_data: dict, 
                           pe_correction: Optional[str] = None,
                           amount_correction: Optional[float] = None) -> dict:
    """
    Applique des corrections manuelles sur les données extraites
    
    Args:
        invoice_data: Données extraites
        pe_correction: Nouveau PE (si correction)
        amount_correction: Nouveau montant (si correction)
    
    Returns:
        Données mises à jour
    """
    result = invoice_data.copy()
    
    if pe_correction is not None:
        normalized_pe = normalize_pe(pe_correction)
        if normalized_pe:
            result['pe_selected'] = normalized_pe
            result['pe_status'] = 'OK'
            result['pe_manual'] = True
            # Retirer le warning PE si présent
            result['warnings'] = [w for w in result['warnings'] if 'PE' not in w.upper()]
    
    if amount_correction is not None:
        result['total_facture'] = round(amount_correction, 2)
        result['total_status'] = 'OK'
        result['amount_manual'] = True
        # Retirer le warning montant si présent
        result['warnings'] = [w for w in result['warnings'] if 'montant' not in w.lower() and 'AMBIGU' not in w]
    
    return result


def summarize_extraction(invoice_data: dict) -> dict:
    """
    Génère un résumé compact des données extraites
    """
    return {
        'filename': invoice_data.get('filename', ''),
        'pe': invoice_data.get('pe_selected'),
        'pe_status': invoice_data.get('pe_status', 'MANQUANT_PE'),
        'pe_candidates_count': len(invoice_data.get('pe_candidates', [])),
        'total_facture': invoice_data.get('total_facture'),
        'total_status': invoice_data.get('total_status', 'MANQUANT_TOTAL'),
        'supplier': invoice_data.get('supplier'),
        'invoice_number': invoice_data.get('invoice_number'),
        'invoice_date': invoice_data.get('invoice_date'),
        'currency': invoice_data.get('currency', 'EUR'),
        'warnings_count': len(invoice_data.get('warnings', [])),
        'page_count': invoice_data.get('page_count', 0)
    }


import re

def extract_item_lines(full_text: str, max_items: int = 50) -> dict:
    """
    Extrait toutes les lignes articles.
    Gère le format OCR type: "17257| ... 48 Caja | 1.248,000 Unidades 0,798 995,904"
    """
    t = (full_text or "").replace("\r", "\n")
    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]

    def to_float_fr(x: str) -> float:
        return float(x.replace(".", "").replace(",", "."))

    items = []
    for ln in lines:
        # Heuristique: une ligne article commence par un code produit (tolérant espaces / pas de |)
        if not re.search(r"^\s*\d{3,}\b", ln):
            continue

        flat = re.sub(r"\s+", " ", ln.replace("|", " ")).strip()

        # code
        mcode = re.match(r"^(\d{3,})\b", flat)
        if not mcode:
            continue
        code = mcode.group(1)

        mqty = re.search(
            r"\b(\d{1,6}(?:[.,]\d{1,3})?)\s*(caja|cajas|colis|unidad|unidades|ud|uds|kg|kilos|kilo|pza|pzas|pz)\b",
            flat,
            flags=re.IGNORECASE
        )

        qty = None
        unit = None

        if mqty:
            raw_qty = mqty.group(1).replace(",", ".")
            try:
                qty = float(raw_qty)
            except Exception:
                qty = None

            unit = mqty.group(2).lower().strip()

            # Normalisation unités
            if unit == "cajas":
                unit = "caja"
            if unit in ("ud", "uds"):
                unit = "unidad"
            if unit == "pzas":
                unit = "pza"
            if unit in ("kilos", "kilo"):
                unit = "kg"
            if unit == "colis":
                unit = "caja"  # si tu veux aligner "colis" avec caja

        # unités détail (ex: 1.248,000 Unidades)
        munits = re.search(
            r"\b(\d[\d\.,]*)(?:\s+)(unidades|kilos|kilo|kg)\b",
            flat,
            flags=re.IGNORECASE
        )

        units_detail = None
        units_detail_uom = None

        if munits:
            raw_num = munits.group(1)
            units_detail_uom = munits.group(2).lower()

            # conversion robuste: garde le dernier séparateur comme décimal si cohérent,
            # sinon interprète comme milliers
            s = raw_num.replace(" ", "").replace("\u00a0", "")

            # Cas "2,664,000" (OCR) -> 2664.000 ou 2664000 selon contexte.
            # Ici avec "Kilos", c'est très probablement 2664.000 (car total 2.813.184 ~ 2813.184)
            if s.count(",") >= 2 and "." not in s:
                # interprète dernière virgule comme décimale, les autres comme milliers
                parts = s.split(",")
                s = "".join(parts[:-1]) + "." + parts[-1]
                units_detail = float(s)
            else:
                # fallback général: supprime milliers, garde décimales
                # ex: "1.248,000" -> 1248.000 ; "2.813.184" -> 2813184 (si pas de virgule)
                if "," in s and "." in s:
                    # suppose '.' milliers et ',' décimal
                    units_detail = float(s.replace(".", "").replace(",", "."))
                elif "," in s:
                    # suppose ',' décimal
                    units_detail = float(s.replace(".", "").replace(",", "."))
                else:
                    # que des points -> milliers
                    units_detail = float(s.replace(".", ""))

        # ✅ retire "1.248,000 Unidades" avant de chercher PU et Total
        flat_no_units = re.sub(r"\b\d[\d\.]*,\d+\s+unidades\b", " ", flat, flags=re.IGNORECASE)
        flat_no_units = re.sub(r"\s+", " ", flat_no_units).strip()

        # prix unit + total ligne = 2 derniers décimaux
        def parse_num(s: str) -> float:
            s = s.replace("€", "").replace(" ", "").replace("\u00a0", "")
            # cas 1: "1.234,56" -> 1234.56
            if "," in s and "." in s:
                return float(s.replace(".", "").replace(",", "."))
            # cas 2: "1,234" -> 1.234
            if "," in s:
                return float(s.replace(".", "").replace(",", "."))
            # cas 3: "7.824.960" -> 7824960
            if s.count(".") >= 2:
                return float(s.replace(".", ""))
            # cas 4: "37.225" -> 37225 (souvent milliers)
            if s.count(".") == 1 and len(s.split(".")[-1]) == 3:
                return float(s.replace(".", ""))
            return float(s)

        nums = re.findall(r"\d[\d\.,]*", flat_no_units)
        unit_price = line_total = None
        if len(nums) >= 2:
            try:
                unit_price = parse_num(nums[-2])
                line_total = parse_num(nums[-1])
            except Exception:
                unit_price = line_total = None

        # description: texte entre code et quantité (si quantité trouvée)
        description = None
        if qty is not None and unit is not None:
            tmp = re.sub(r"^\d{3,}\s*", "", flat).strip()
            parts = re.split(rf"\b{qty}\s+{re.escape(unit)}\b", tmp, maxsplit=1, flags=re.IGNORECASE)
            description = parts[0].strip() if parts else tmp
        else:
            description = re.sub(r"^\d{3,}\s*", "", flat).strip()

        ok = bool(code and qty is not None and unit_price is not None and line_total is not None)
        # On garde même si ok=False, utile pour debug
        items.append({
            "ok": ok,
            "item_code": code,
            "description": description,
            "quantity": qty,
            "unit": unit,
            "units_detail": units_detail,
            "units_detail_uom": units_detail_uom,
            "unit_price": unit_price,
            "line_total": line_total,
            "raw_line": ln,
        })

        if len(items) >= max_items:
            break

    # somme des lignes (uniquement celles ok)
    sum_lines = sum(it["line_total"] for it in items if it.get("ok") and isinstance(it.get("line_total"), float))

    return {
        "ok": len(items) > 0,
        "count": len(items),
        "items": items,
        "sum_line_totals": sum_lines,
    }
