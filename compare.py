"""
compare.py - Jointure au référentiel, calcul des écarts et détection des doublons
"""

from __future__ import annotations

import pandas as pd
from typing import List, Optional, Tuple

from utils import normalize_pe, calculate_delta, check_tolerance

# Tolérances "anti faux écarts"
PU_TOL = 0.005      # PU: 0,5 centime (OCR 3dp vs ref 2dp)
TOTAL_TOL = 0.02    # Total ligne: 2 centimes
QTY_TOL = 1e-6      # Quantité: quasi strict (mets 0.001 si quantités décimales)

# -----------------------------
# Helpers OCR / normalisation
# -----------------------------

def norm_pu_ref_to_3dp(pu_ref) -> Optional[float]:
    """Référentiel à 2 décimales -> normalisé à 3 décimales pour comparer à l'OCR (3dp)."""
    if pu_ref is None:
        return None
    try:
        s = str(pu_ref).strip().replace(",", ".")
        return round(float(s), 3)
    except Exception:
        return None


def fix_ocr_pu_3dp(raw) -> Optional[float]:
    """PU OCR (SOCOBO) toujours à 3 décimales.
    Ex: 1336 -> 1.336 ; '1,336' -> 1.336 ; 1.336 -> 1.336
    """
    if raw is None:
        return None

    # Si OCR a déjà converti en numérique
    if isinstance(raw, (int, float)):
        x = float(raw)
        return x / 1000.0 if x.is_integer() else x

    s = str(raw).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        digits = "".join(ch for ch in s if ch.isdigit())
        if not digits:
            return None
        if len(digits) <= 3:
            digits = digits.zfill(4)
        return float(digits[:-3] + "." + digits[-3:])


def fix_ocr_total_base(raw) -> Optional[float]:
    """Parse un total OCR en float (2dp quand possible), sans corriger l'échelle ×10/×100."""
    if raw is None:
        return None

    if isinstance(raw, (int, float)):
        return float(raw)

    s = str(raw).strip().replace(",", ".")
    try:
        return float(s)
    except Exception:
        digits = "".join(ch for ch in s if ch.isdigit())
        if not digits:
            return None
        # on suppose 2 décimales si digits-only
        if len(digits) <= 2:
            digits = digits.zfill(3)
        return float(digits[:-2] + "." + digits[-2:])


def best_scale_to_match(ref_value, ocr_value, candidates=(1, 10, 100, 1000, 10000)) -> Tuple[Optional[float], int]:
    """Corrige les totaux OCR dont la virgule est décalée (×10/×100) en choisissant l'échelle la plus proche du ref."""
    if ref_value is None or ocr_value is None:
        return ocr_value, 1

    try:
        ref = float(ref_value)
        ocr = float(ocr_value)
    except Exception:
        return ocr_value, 1

    best_v = ocr
    best_div = 1
    best_err = float("inf")

    for d in candidates:
        v = ocr / d
        err = abs(v - ref)
        if err < best_err:
            best_err = err
            best_v = v
            best_div = d

    return best_v, best_div


def _ok_delta(a, b, tol: float) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def to_money_2dp(series: pd.Series) -> pd.Series:
    """Conversion robuste des montants référentiel: supporte EU/US + valeurs en centimes (digits-only)."""
    s = series.astype(str).str.strip()
    s = (s.str.replace("\u00A0", "", regex=False)
           .str.replace("\u202F", "", regex=False)
           .str.replace(" ", "", regex=False)
           .str.replace("€", "", regex=False))

    # EU 1.234,56 -> 1234.56
    mask_eu = s.str.contains(",", na=False) & s.str.contains(r"\.", na=False)
    s.loc[mask_eu] = s.loc[mask_eu].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

    # US 1,234.56 -> 1234.56
    mask_us = s.str.contains(r"\.", na=False) & s.str.contains(",", na=False) & ~mask_eu
    s.loc[mask_us] = s.loc[mask_us].str.replace(",", "", regex=False)

    # 1234,56 -> 1234.56
    mask_comma = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
    s.loc[mask_comma] = s.loc[mask_comma].str.replace(",", ".", regex=False)

    out = pd.to_numeric(s, errors="coerce")

    # digits-only => centimes => /100
    digits_only = s.str.fullmatch(r"\d+", na=False)
    out.loc[digits_only & out.notna()] = out.loc[digits_only & out.notna()] / 100.0
    return out


# -----------------------------
# Référentiel + jointure
# -----------------------------

def load_referential(file_path: str) -> pd.DataFrame:
    """Charge le référentiel interne depuis un fichier CSV ou Excel, et normalise les colonnes."""
    # lecture
    if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
        df = pd.read_excel(file_path, dtype=str, engine="openpyxl")
    else:
        df = None
        for encoding in ["utf-8", "latin-1", "cp1252"]:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=";", engine="python", dtype=str)
                break
            except UnicodeDecodeError:
                continue
        if df is None:
            raise ValueError("Impossible de lire le fichier CSV (encodage non supporté)")

    # normalisation colonnes
    df.columns = df.columns.astype(str).str.strip().str.lower()

    # mapping fixe pour ton Excel
    column_mapping = {
        "reference client 1": "pe",
        "code produit": "code_article",
        "nb colis total": "qte_ligne_attendue",
        "prix vente": "pu_ligne_attendu",
        "mnt facturé": "total_ligne_attendu",
    }
    df = df.rename(columns=column_mapping)

    required_cols = ["pe", "code_article", "qte_ligne_attendue", "pu_ligne_attendu", "total_ligne_attendu"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Colonnes manquantes après mapping: {missing}. Colonnes dispo: {list(df.columns)}")

    # PE propre + normalisé
    df["pe"] = df["pe"].astype(str).str.strip().apply(normalize_pe)

    # code article = chiffres uniquement
    df["code_article"] = (
        df["code_article"].astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    )

    # numériques (référentiel = 2 décimales)
    def to_num(series: pd.Series) -> pd.Series:
        s = series.astype(str).str.strip()
        s = (s.str.replace("\u00A0", "", regex=False)
               .str.replace("\u202F", "", regex=False)
               .str.replace(" ", "", regex=False)
               .str.replace("€", "", regex=False))

        # EU 1.234,56 -> 1234.56
        mask_eu = s.str.contains(",", na=False) & s.str.contains(r"\.", na=False)
        s.loc[mask_eu] = s.loc[mask_eu].str.replace(".", "", regex=False).str.replace(",", ".", regex=False)

        # US 1,234.56 -> 1234.56
        mask_us = s.str.contains(r"\.", na=False) & s.str.contains(",", na=False) & ~mask_eu
        s.loc[mask_us] = s.loc[mask_us].str.replace(",", "", regex=False)

        # 1234,56 -> 1234.56
        mask_comma = s.str.contains(",", na=False) & ~s.str.contains(r"\.", na=False)
        s.loc[mask_comma] = s.loc[mask_comma].str.replace(",", ".", regex=False)

        out = pd.to_numeric(s, errors="coerce")
        if out.isna().any():
            out2 = pd.to_numeric(s.str.replace(".", "", regex=False), errors="coerce")
            out = out.fillna(out2)
        return out

    df["qte_ligne_attendue"] = to_num(df["qte_ligne_attendue"])
    df["pu_ligne_attendu"] = to_num(df["pu_ligne_attendu"])
    df["total_ligne_attendu"] = to_money_2dp(df["total_ligne_attendu"])

    # Optionnel: quantité totale header
    if "quantite_totale_attendue" in df.columns:
        q = df["quantite_totale_attendue"].astype(str).str.strip()
        q = (q.str.replace("\u00A0", "", regex=False)
               .str.replace(" ", "", regex=False)
               .str.replace(".", "", regex=False)
               .str.replace(",", ".", regex=False))
        df["quantite_totale_attendue"] = pd.to_numeric(q, errors="coerce")

    return df


def join_with_referential(invoices_data, referential_df):
    """Injecte sur chaque facture: montant_total_attendu, quantite_totale_attendue et ref_lines (lignes par code)."""
    required = ["pe", "code_article", "total_ligne_attendu"]
    missing = [c for c in required if c not in referential_df.columns]
    if missing:
        raise KeyError(
            f"Référentiel: colonnes manquantes: {missing}. Colonnes dispo: {list(referential_df.columns)}"
        )

    # 1) Header agrégé par PE
    agg = {"total_ligne_attendu": "sum"}
    if "qte_ligne_attendue" in referential_df.columns:
        agg["qte_ligne_attendue"] = "sum"

    header = referential_df.groupby("pe", as_index=False).agg(agg)
    header_index = header.set_index("pe").to_dict("index")

    # 2) Toutes les lignes (articles) par PE
    cols = ["code_article", "total_ligne_attendu"]
    if "qte_ligne_attendue" in referential_df.columns:
        cols.append("qte_ligne_attendue")
    if "pu_ligne_attendu" in referential_df.columns:
        cols.append("pu_ligne_attendu")

    ref_lines_index = (
        referential_df.groupby("pe")
        .apply(lambda g: g[cols].to_dict("records"))
        .to_dict()
    )

    results = []
    for invoice in invoices_data:
        result = invoice.copy()
        pe = result.get("pe_selected")

        result["ref_found"] = False
        result["montant_total_attendu"] = None
        result["quantite_totale_attendue"] = None
        result["ref_lines"] = []

        if pe and pe in header_index:
            result["ref_found"] = True
            result["montant_total_attendu"] = header_index[pe].get("total_ligne_attendu")
            result["quantite_totale_attendue"] = header_index[pe].get("qte_ligne_attendue")
            result["ref_lines"] = ref_lines_index.get(pe, [])
        elif pe:
            result["warnings"] = result.get("warnings", []) + [f"PE {pe} non trouvé dans le référentiel"]

        results.append(result)

    return results


# -----------------------------
# Comparaison lignes OCR vs REF
# -----------------------------

def compare_lines(invoice: dict, tol_qty: float = QTY_TOL, tol_pu: float = PU_TOL, tol_total: float = TOTAL_TOL) -> dict:
    """Compare les lignes OCR vs référentiel sur code_article."""
    ocr_items = ((invoice.get("items") or {}).get("items") or [])
    ref_lines = invoice.get("ref_lines") or []

    # Index OCR par code_article (agrégé)
    ocr_by_code = {}
    for it in ocr_items:
        code = str(it.get("item_code") or "").strip()
        if not code:
            continue
        ocr_by_code.setdefault(code, {"qty": 0.0, "total": 0.0, "pu": None, "desc": None})

        if it.get("quantity") is not None:
            try:
                ocr_by_code[code]["qty"] += float(it["quantity"])
            except Exception:
                pass
        if it.get("line_total") is not None:
            try:
                ocr_by_code[code]["total"] += float(it["line_total"])
            except Exception:
                pass
        if it.get("unit_price") is not None:
            # ⚠️ on garde brut (float/int), la correction 3dp sera faite plus bas
            try:
                ocr_by_code[code]["pu"] = it["unit_price"]
            except Exception:
                pass
        if it.get("description"):
            ocr_by_code[code]["desc"] = it["description"]

    # Index REF par code_article (agrégé)
    ref_by_code = {}
    for r in ref_lines:
        code = str(r.get("code_article") or "").strip()
        if not code:
            continue
        ref_by_code.setdefault(code, {"qty": 0.0, "total": 0.0, "pu": None})

        if r.get("qte_ligne_attendue") is not None:
            try:
                ref_by_code[code]["qty"] += float(r["qte_ligne_attendue"])
            except Exception:
                pass
        if r.get("total_ligne_attendu") is not None:
            try:
                ref_by_code[code]["total"] += float(r["total_ligne_attendu"])
            except Exception:
                pass
        if r.get("pu_ligne_attendu") is not None:
            try:
                ref_by_code[code]["pu"] = r["pu_ligne_attendu"]
            except Exception:
                pass

    all_codes = sorted(set(ocr_by_code.keys()) | set(ref_by_code.keys()))

    rows = []
    any_issue = False

    for code in all_codes:
        o = ocr_by_code.get(code)
        r = ref_by_code.get(code)

        qty_ocr = o["qty"] if o else None
        qty_ref = r["qty"] if r else None

        # --- REF D'ABORD (pour corriger l'échelle du total OCR)
        pu_ref = norm_pu_ref_to_3dp(r["pu"] if r else None)
        tot_ref = r["total"] if r else None
        try:
            tot_ref = round(float(tot_ref), 2) if tot_ref is not None else None
        except Exception:
            tot_ref = None

        # --- OCR ensuite
        pu_ocr_raw = o["pu"] if o else None
        tot_ocr_raw = o["total"] if o else None

        pu_ocr = fix_ocr_pu_3dp(pu_ocr_raw)
        pu_ocr = round(pu_ocr, 3) if pu_ocr is not None else None

        tot_ocr_base = fix_ocr_total_base(tot_ocr_raw)
        tot_ocr, _tot_div = best_scale_to_match(tot_ref, tot_ocr_base)
        tot_ocr = round(tot_ocr, 2) if tot_ocr is not None else None

        status_qty = "OK" if _ok_delta(qty_ocr, qty_ref, tol_qty) else "ECART"
        status_pu = "OK" if _ok_delta(pu_ocr, pu_ref, tol_pu) else "ECART"
        status_tot = "OK" if _ok_delta(tot_ocr, tot_ref, tol_total) else "ECART"

        # ✅ anti-bruit: si QTY ou PU est en ECART, le total devient INFO (pas un "problème" principal)
        if status_tot == "ECART" and (status_qty == "ECART" or status_pu == "ECART"):
            status_tot = "INFO"

        # Statut global ligne
        if o is None:
            status = "MANQUANT_OCR"
            any_issue = True
        elif r is None:
            status = "CODE_INCONNU_REF"
            any_issue = True
        else:
            status = "OK" if (status_qty == "OK" and status_pu == "OK" and status_tot in ("OK", "INFO")) else "ECART"
            if status != "OK":
                any_issue = True

        rows.append({
            "code_article": code,
            "desc_ocr": (o.get("desc") if o else ""),
            "qty_ocr": qty_ocr,
            "qty_ref": qty_ref,
            "pu_ocr": pu_ocr,
            "pu_ref": pu_ref,
            "total_ocr": tot_ocr,
            "total_ref": tot_ref,
            "status": status,
            "status_qty": status_qty,
            "status_pu": status_pu,
            "status_total": status_tot,
        })

    invoice["line_comparison"] = rows
    invoice["line_status"] = "ECART_LIGNES" if any_issue else "OK"

    flags = set(invoice.get("flags") or [])
    if any_issue:
        flags.add("ECART_LIGNES")
    invoice["flags"] = sorted(flags)

    return invoice


def calculate_comparison(invoices_data: List[dict],
                        tol_euros: float = 0.05,
                        tol_percent: float = 0.5) -> List[dict]:
    """
    Calcule les écarts et statuts pour chaque facture
    """
    results = []

    for invoice in invoices_data:
        result = invoice.copy()

        total_facture = result.get('total_facture')
        montant_attendu = result.get('montant_total_attendu')

        # Calcul des deltas
        delta_info = calculate_delta(total_facture, montant_attendu)
        result.update(delta_info)

        # Détermination du statut final
        pe_status = result.get('pe_status', 'MANQUANT_PE')
        total_status = result.get('total_status', 'MANQUANT_TOTAL')

        if pe_status == 'MANQUANT_PE':
            result['statut_final'] = 'MANQUANT_PE'
        elif not result.get('ref_found'):
            result['statut_final'] = 'PE_INCONNU'
        elif total_status == 'MANQUANT_TOTAL' or montant_attendu is None:
            result['statut_final'] = 'INCOMPLET'
        else:
            tolerance_status = check_tolerance(
                result.get('delta_abs'),
                result.get('delta_pct'),
                tol_euros,
                tol_percent
            )
            result['statut_final'] = tolerance_status

        # -----------------------
        # Comparaison Quantités (OCR vs Référentiel)
        # -----------------------
        qty_ocr = result.get('quantite_total_ocr')
        qty_ref = result.get('quantite_totale_attendue')

        result['delta_qte'] = None
        result['quantite_status'] = 'NA'

        if qty_ref is None and qty_ocr is None:
            result['quantite_status'] = 'NA'
        elif qty_ref is None:
            result['quantite_status'] = 'MANQUANT_QTE_REF'
        elif qty_ocr is None:
            result['quantite_status'] = 'MANQUANT_QTE_OCR'
        else:
            try:
                dq = float(qty_ocr) - float(qty_ref)
                result['delta_qte'] = dq
                if abs(dq) <= 0.01:
                    result['quantite_status'] = 'OK'
                else:
                    result['quantite_status'] = 'ECART_QTE'
            except Exception:
                result['quantite_status'] = 'ERREUR_QTE'

        # Flags additionnels (NE PAS écraser les flags déjà posés à l'extraction)
        result["flags"] = list(result.get("flags") or [])

        def _add_flag(f: str):
            if f not in result["flags"]:
                result["flags"].append(f)

        if pe_status == "MULTI_PE":
            _add_flag("MULTI_PE")
        if total_status == "TOTAL_AMBIGU":
            _add_flag("TOTAL_AMBIGU")

        if result.get("quantite_status") == "ECART_QTE":
            _add_flag("ECART_QTE")

        for page in result.get("pages_data", []):
            if page.get("quality", {}).get("warning") == "OCR_FAIBLE":
                _add_flag("OCR_FAIBLE")

        # (optionnel) nettoyage final
        result["flags"] = sorted(set(result["flags"]))

        # --- Comparaison lignes (OCR vs Référentiel par code_article) ---
        result = compare_lines(result)

        # Si tout était OK au niveau total mais lignes KO -> statut final devient ECART_LIGNES
        if result.get("statut_final") == "OK" and result.get("line_status") == "ECART_LIGNES":
            result["statut_final"] = "ECART_LIGNES"

        results.append(result)

    return results


def detect_duplicates(invoices_data: List[dict], tol_amount: float = 1.0) -> List[dict]:
    """
    Détecte les doublons dans le lot de factures
    """
    results = []

    # Index par PE
    pe_groups = {}
    for idx, inv in enumerate(invoices_data):
        pe = inv.get("pe_selected")
        if pe:
            pe_groups.setdefault(pe, []).append(idx)

    # Index par fournisseur + n° facture
    invoice_groups = {}
    for idx, inv in enumerate(invoices_data):
        supplier = (inv.get("supplier") or "").strip().lower()
        inv_num = (inv.get("invoice_number") or "").strip().lower()
        if supplier and inv_num:
            key = f"{supplier}|{inv_num}"
            invoice_groups.setdefault(key, []).append(idx)

    # Helper: récupérer un montant comparable
    def _get_amount(inv: dict):
        a = inv.get("total_facture")
        try:
            return float(a) if a is not None else None
        except Exception:
            return None

    for inv in invoices_data:
        result = inv.copy()
        result.setdefault("flags", [])
        result["duplicates"] = []

        filename = result.get("filename", "")
        pe = result.get("pe_selected")

        # 1) Doublon PE
        if pe and pe in pe_groups and len(pe_groups[pe]) > 1:
            if "DOUBLON_PE_LOT" not in result["flags"]:
                result["flags"].append("DOUBLON_PE_LOT")

            other_files = [
                invoices_data[i].get("filename", "")
                for i in pe_groups[pe]
                if invoices_data[i].get("filename", "") != filename
            ]
            if other_files:
                result["duplicates"].append({"type": "PE", "other_files": other_files})

        # 2) Doublon facture (fournisseur + n° facture)
        supplier = (result.get("supplier") or "").strip().lower()
        inv_num = (result.get("invoice_number") or "").strip().lower()
        if supplier and inv_num:
            key = f"{supplier}|{inv_num}"
            if key in invoice_groups and len(invoice_groups[key]) > 1:
                if "DOUBLON_FACTURE" not in result["flags"]:
                    result["flags"].append("DOUBLON_FACTURE")

                other_files = [
                    invoices_data[i].get("filename", "")
                    for i in invoice_groups[key]
                    if invoices_data[i].get("filename", "") != filename
                ]
                if other_files:
                    result["duplicates"].append({"type": "FACTURE", "other_files": other_files})

        # 3) Doublon faible (même PE + montant proche)
        amount = _get_amount(result)
        if pe and amount is not None and pe in pe_groups and len(pe_groups[pe]) > 1:
            close_files = []
            for i in pe_groups[pe]:
                other = invoices_data[i]
                if other.get("filename", "") == filename:
                    continue
                other_amount = _get_amount(other)
                if other_amount is None:
                    continue
                if abs(other_amount - amount) <= tol_amount:
                    close_files.append(other.get("filename", ""))

            if close_files:
                if "DOUBLON_FAIBLE" not in result["flags"]:
                    result["flags"].append("DOUBLON_FAIBLE")
                result["duplicates"].append({"type": "FAIBLE", "other_files": close_files, "tol": tol_amount})

        results.append(result)

    return results


def generate_summary_table(invoices_data: List[dict]) -> pd.DataFrame:
    """
    Génère un tableau récapitulatif des factures
    """
    rows = []

    for inv in invoices_data:
        row = {
            'Fichier': inv.get('filename', ''),
            'PE': inv.get('pe_selected', ''),
            'Statut PE': inv.get('pe_status', ''),
            'Fournisseur': inv.get('supplier', ''),
            'N° Facture': inv.get('invoice_number', ''),
            'Date Facture': inv.get('invoice_date', ''),
            'Devise': inv.get('currency', 'EUR'),
            'Total Facture': inv.get('total_facture'),
            'Montant Attendu': inv.get('montant_total_attendu'),
            'Delta (€)': inv.get('delta'),
            'Delta (%)': inv.get('delta_pct'),
            'Statut': inv.get('statut_final', ''),
            'Flags': ', '.join(inv.get('flags', [])),
            'Warnings': len(inv.get('warnings', []))
        }
        rows.append(row)

    return pd.DataFrame(rows)


def filter_invoices(invoices_data: List[dict],
                   status_filter: Optional[List[str]] = None,
                   flag_filter: Optional[List[str]] = None,
                   pe_search: Optional[str] = None) -> List[dict]:
    """
    Filtre les factures selon différents critères
    """
    results = invoices_data

    if status_filter:
        results = [inv for inv in results if inv.get('statut_final') in status_filter]

    if flag_filter:
        results = [inv for inv in results
                  if any(f in inv.get('flags', []) for f in flag_filter)]

    if pe_search:
        pe_search_norm = normalize_pe(pe_search)
        results = [inv for inv in results
                  if pe_search_norm in (inv.get('pe_selected') or '')]

    return results

