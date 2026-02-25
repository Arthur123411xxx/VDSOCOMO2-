"""
app.py - Interface Streamlit principale pour le contrôle de factures
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path

# Configuration de la page

st.set_page_config(
    page_title="VANDAMME — Contrôle factures",
    page_icon="🍌",
    layout="wide",
    initial_sidebar_state="expanded"
)

LOGO = Path(__file__).parent / "assets" / "VD.png"
if LOGO.exists():
    st.sidebar.image(str(LOGO), use_container_width=True)

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; }
    [data-testid="stSidebar"] { padding-top: 1rem; }
    .stMetric { background: #ffffff0d; border-radius: 12px; padding: 12px; }
</style>
""", unsafe_allow_html=True)


# Imports locaux
from ocr import check_dependencies, process_pdf
from extract import extract_invoice_data, apply_manual_correction
from compare import (
    load_referential,
    join_with_referential,
    calculate_comparison,
    detect_duplicates,
    generate_summary_table,
    filter_invoices
)

from export import generate_report_filename, export_bundle_zip


# Vérification des dépendances au démarrage
deps_ok, deps_errors = check_dependencies()

if not deps_ok:
    st.error("⚠️ Dépendances système manquantes")
    for error in deps_errors:
        st.error(error)
    st.stop()

# Initialisation du state
if 'invoices_data' not in st.session_state:
    st.session_state.invoices_data = []
if 'referential_df' not in st.session_state:
    st.session_state.referential_df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'import'


# ===============================
# i18n (FR / NL)
# ===============================
LANG_OPTIONS = {
    "fr": "🇫🇷 Français",
    "nl": "🇳🇱 Nederlands",
}

if "lang" not in st.session_state:
    st.session_state.lang = "fr"

lang = st.sidebar.selectbox(
    "Langue / Taal",
    options=list(LANG_OPTIONS.keys()),
    format_func=lambda k: LANG_OPTIONS[k],
    key="lang",
)

TRANSLATIONS = {
    "fr": {
        "sidebar_title": "System **V1**",
        "sidebar_caption": "Analyse & export des factures",
        "nav": "Navigation",
        "others": "Autres",
        "others_placeholder": "— Autres pages —",
        "params": "⚙️ Paramètres",
        "tol_euros": "Tolérance (€)",
        "tol_euros_help": "Écart **ABSOLU** acceptable",
        "tol_percent": "Tolérance (%)",
        "tol_percent_help": "Écart **relatif** acceptable",
        "enhanced": "Prétraitement renforcé",
        "enhanced_help": "Améliore l'OCR pour les documents de mauvaise qualité (plus lent)",
        "light_mode": "Mode léger (ne garde pas les images en mémoire)",
        "light_mode_help": "Réduit fortement la RAM en évitant de stocker les images des pages dans session_state",
        "stats": "📈 Statistiques",
        "total": "Total",
        "ok": "✅ OK",
        "gaps": "⚠️ Écarts",
        "incomplete": "❓ Incomplet",
    },
    "nl": {
        "sidebar_title": "Systeem **V1**",
        "sidebar_caption": "Analyse & export van facturen",
        "nav": "Navigatie",
        "others": "Overige",
        "others_placeholder": "— Andere pagina’s —",
        "params": "⚙️ Instellingen",
        "tol_euros": "Tolerantie (€)",
        "tol_euros_help": "**Absolute** toegestane afwijking",
        "tol_percent": "Tolerantie (%)",
        "tol_percent_help": "**Relatieve** toegestane afwijking",
        "enhanced": "Versterkte voorbewerking",
        "enhanced_help": "Verbetert OCR bij slechte kwaliteit (trager)",
        "light_mode": "Lichte modus (bewaar geen afbeeldingen)",
        "light_mode_help": "Bespaart RAM door geen paginabeelden in session_state te bewaren",
        "stats": "📈 Statistieken",
        "total": "Totaal",
        "ok": "✅ OK",
        "gaps": "⚠️ Afwijkingen",
        "incomplete": "❓ Onvolledig",
    },
}

def t(key: str) -> str:
    return TRANSLATIONS.get(lang, TRANSLATIONS["fr"]).get(key, key)

def tr(fr: str, nl: str) -> str:
    return fr if lang == "fr" else nl

PAGE_LABELS = {
    "import": {"fr": "📥 Import", "nl": "📥 Importeren"},
    "export": {"fr": "📤 Export", "nl": "📤 Exporteren"},
    "traitement": {"fr": "⚙️ Traitement", "nl": "⚙️ Verwerking"},
    "preview": {"fr": "👁️ Preview OCR", "nl": "👁️ OCR-voorbeeld"},
    "resultats": {"fr": "📊 Résultats", "nl": "📊 Resultaten"},
}

def page_label(pid: str) -> str:
    return PAGE_LABELS.get(pid, {}).get(lang, pid)



# Sidebar - Navigation (1 seul menu + toggle avancé)
st.sidebar.markdown(t("sidebar_title"))
st.sidebar.caption(t("sidebar_caption"))
st.sidebar.markdown("---")

if "advanced_mode" not in st.session_state:
    st.session_state.advanced_mode = False

st.session_state.advanced_mode = st.sidebar.toggle(
    "Mode avancé" if lang == "fr" else "Geavanceerde modus",
    value=st.session_state.advanced_mode
)

# ✅ Résultats en NORMAL (pas avancé)
BASE_PAGES = ["import","traitement","export"]
# ✅ Avancé = uniquement outils
ADV_PAGES = ["preview","resultats"]

PAGE_IDS = BASE_PAGES + (ADV_PAGES if st.session_state.advanced_mode else [])

# Si l’utilisateur était sur une page avancée et désactive le mode avancé
current_id = st.session_state.get("current_page", "import")
if current_id not in PAGE_IDS:
    current_id = "import"

page_id = st.sidebar.radio(
    t("nav"),
    PAGE_IDS,
    index=PAGE_IDS.index(current_id),
    format_func=page_label
)

st.session_state.current_page = page_id



# Paramètres dans la sidebar
st.sidebar.markdown("---")
st.sidebar.subheader(t("params"))

tol_euros = st.sidebar.number_input(
    t("tol_euros"),
    min_value=0.0,
    max_value=100000.0,
    value=0.05,
    step=0.01,
    help=t("tol_euros_help")
)

tol_percent = st.sidebar.number_input(
    t("tol_percent"),
    min_value=0.0,
    max_value=100.0,
    value=0.5,
    step=0.01,
    help=t("tol_percent_help")
)

enhanced_preprocessing = st.sidebar.checkbox(
    t("enhanced"),
    value=False,
    help=t("enhanced_help")
)
light_mode = st.sidebar.checkbox(
    t("light_mode"),
    value=False,
    help=t("light_mode_help")
)

# Stats dans la sidebar
if st.session_state.invoices_data:
    st.sidebar.markdown("---")
    st.sidebar.subheader(t("stats"))
    total = len(st.session_state.invoices_data)
    ok = sum(1 for i in st.session_state.invoices_data if i.get('statut_final') == 'OK')
    ecart = sum(1 for i in st.session_state.invoices_data if i.get('statut_final') == 'ECART')
    incomplet = sum(1 for i in st.session_state.invoices_data if i.get('statut_final') in ['INCOMPLET', 'MANQUANT_PE', 'PE_INCONNU'])

    col1, col2 = st.sidebar.columns(2)
    col1.metric(t("total"), total)
    col2.metric(t("ok"), ok)
    col1.metric(t("gaps"), ecart)
    col2.metric(t("incomplete"), incomplet)

# ===============================
# PAGE: IMPORT
# ===============================
if page_id == "import":
    st.title(tr("📥 Import des fichiers", "📥 Bestanden importeren"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(tr("📄 Factures (PDF)", "📄 Facturen (PDF)"))
        uploaded_pdfs = st.file_uploader(
            tr("Glissez vos PDFs scannés ici", "Sleep uw gescande PDF's hierheen"),
            type=['pdf'],
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if uploaded_pdfs:
            st.success(f"✅ {len(uploaded_pdfs)} fichier(s) chargé(s)")
            for f in uploaded_pdfs:
                st.text(f"  • {f.name}")

    with col2:
        st.subheader(tr("📋 Référentiel interne", "📋 Interne referentiel"))
        uploaded_ref = st.file_uploader(
            tr("Fichier CSV ou Excel avec colonnes PE et montant_total_attendu", "CSV- of Excelbestand met kolommen PE en verwacht_totaal"),
            type=['csv', 'xlsx', 'xls'],
            key="ref_uploader"
        )

        if uploaded_ref:
            try:
                # Sauvegarder temporairement
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_ref.name).suffix) as tmp:
                    tmp.write(uploaded_ref.read())
                    tmp_path = tmp.name

                ref_df = load_referential(tmp_path)
                st.session_state.referential_df = ref_df
                os.unlink(tmp_path)

                st.success(f"✅ Référentiel chargé: {len(ref_df)} entrées")

                with st.expander("Aperçu du référentiel"):
                    st.dataframe(ref_df.head(10))

            except Exception as e:
                st.error(f"❌ Erreur: {str(e)}")

    # Bouton de lancement
    st.markdown("---")
    if uploaded_pdfs and st.session_state.referential_df is not None:
        if st.button(tr("🍌 Lancer le traitement", "🍌 Start verwerking"), type="primary", use_container_width=True):
            st.session_state.uploaded_pdfs = uploaded_pdfs
            st.session_state.current_page = 'traitement'
            st.rerun()
    else:
        st.info(tr("📌 Chargez les factures PDF et le référentiel pour continuer", "📌 Laad de PDF-facturen en het referentiel om verder te gaan"))

# ===============================
# PAGE: TRAITEMENT
# ===============================
elif page_id == "traitement":
    st.title(tr("⚙️ Traitement OCR & Extraction", "⚙️ OCR-verwerking & extractie"))

    uploaded_pdfs = st.session_state.get('uploaded_pdfs', [])

    if not uploaded_pdfs:
        st.warning(tr("⚠️ Aucun fichier à traiter. Retournez à l'import.", "⚠️ Geen bestand om te verwerken. Ga terug naar import."))
        st.stop()

    if st.session_state.processed:
        st.success(tr("✅ Traitement déjà effectué", "✅ Verwerking al uitgevoerd"))
        if st.button(tr("🔄 Relancer le traitement", "🔄 Verwerking opnieuw starten")):
            st.session_state.processed = False
            st.rerun()
        st.stop()

    progress_bar = st.progress(0)
    status_text = st.empty()

    invoices_data = []
    total_files = len(uploaded_pdfs)

    for idx, pdf_file in enumerate(uploaded_pdfs):
        status_text.text(f"📄 Traitement de {pdf_file.name} ({idx + 1}/{total_files})")

        try:
            tmp_path = None
            try:
                # ✅ garde le PDF d'origine pour export ZIP
                pdf_bytes = pdf_file.getvalue()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_bytes)
                    tmp_path = tmp.name

                ocr_result = process_pdf(tmp_path, enhanced_preprocessing=enhanced_preprocessing)
                invoice_data = extract_invoice_data(ocr_result, include_images=not light_mode)

                invoice_data["pdf_bytes"] = pdf_bytes
                invoice_data["source_filename"] = pdf_file.name

                invoices_data.append(invoice_data)

            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass


        except Exception as e:
            st.error(f"❌ Erreur sur {pdf_file.name}: {str(e)}")
            invoices_data.append({
                'filename': pdf_file.name,
                'error': str(e),
                'pe_status': 'ERREUR',
                'statut_final': 'ERREUR'
            })

        progress_bar.progress((idx + 1) / total_files)

    status_text.text("🔗 Jointure avec le référentiel...")

    # Jointure référentiel
    if st.session_state.referential_df is not None:
        invoices_data = join_with_referential(
            invoices_data,
            st.session_state.referential_df
        )

    # Calcul des comparaisons
    invoices_data = calculate_comparison(
        invoices_data,
        tol_euros=tol_euros,
        tol_percent=tol_percent
    )

    # Détection doublons
    invoices_data = detect_duplicates(invoices_data)

    st.session_state.invoices_data = invoices_data
    st.session_state.processed = True

    status_text.text("✅ Traitement terminé!")

    # Résumé
    st.markdown("---")
    st.subheader(tr("📊 Résumé", "📊 Samenvatting"))

    df_summary = generate_summary_table(invoices_data)
    st.dataframe(df_summary, use_container_width=True)

    st.success(tr("✅ Traitement terminé! Va dans 'Résultats' pour voir les détails.", "✅ Klaar! Ga naar 'Resultaten' voor details."))

# ===============================
# PAGE: PREVIEW OCR
# ===============================
elif page_id == "preview":
    st.title(tr("👁️ Preview OCR & Corrections", "👁️ OCR-voorbeeld & correcties"))

    if not st.session_state.invoices_data:
        st.warning("⚠️ Aucune donnée. Lancez d'abord le traitement.")
        st.stop()

    invoices = st.session_state.invoices_data

    # Sélection de la facture
    filenames = [inv.get('filename', f'Facture {i+1}') for i, inv in enumerate(invoices)]
    selected_idx = st.selectbox(
        "Sélectionner une facture",
        range(len(filenames)),
        format_func=lambda x: filenames[x]
    )

    invoice = invoices[selected_idx]

    # Layout en colonnes
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📸 Image & OCR")

        pages_data = invoice.get('pages_data', [])
        if pages_data:
            page_idx = st.selectbox(
                "Page",
                range(len(pages_data)),
                format_func=lambda x: f"Page {x + 1}"
            )

            page = pages_data[page_idx]

            # Afficher l'image
            if page.get('preprocessed_image'):
                st.image(page['preprocessed_image'], caption="Image prétraitée", use_container_width=True)
            elif page.get('image'):
                st.image(page['image'], caption="Image originale", use_container_width=True)

            # Texte OCR
            with st.expander("📝 Texte OCR", expanded=False):
                st.text_area(
                    "Texte extrait",
                    value=page.get('text', ''),
                    height=300,
                    disabled=True
                )

            # Qualité OCR
            quality = page.get('quality', {})
            if quality:
                st.markdown("**Qualité OCR:**")
                cols = st.columns(4)
                cols[0].metric("Longueur", quality.get('length', 0))
                cols[1].metric("Ratio alphanum", f"{quality.get('alphanum_ratio', 0):.0%}")
                cols[2].metric("Tokens", quality.get('token_count', 0))
                cols[3].metric("Score", f"{quality.get('quality_score', 0):.0%}")

                if quality.get('warning'):
                    st.warning(f"⚠️ {quality.get('warning')}")

    with col2:
        st.subheader("🔍 Données extraites")

        # PE
        st.markdown("**PE détectés:**")
        pe_candidates = invoice.get('pe_candidates', [])
        if pe_candidates:
            for c in pe_candidates:
                score_pct = f"{c['score']:.0%}"
                st.markdown(f"- `{c['pe']}` (score: {score_pct}) - _{c.get('context_snippet', '')[:50]}..._")
        else:
            st.warning("Aucun PE détecté")

        st.markdown(f"**PE sélectionné:** `{invoice.get('pe_selected', 'N/A')}` ({invoice.get('pe_status', '')})")

        st.markdown("---")

        # Montants
        st.markdown("**Montants détectés:**")
        amount_candidates = invoice.get('amount_candidates', [])[:5]
        if amount_candidates:
            for c in amount_candidates:
                score_pct = f"{c['score']:.0%}"
                keyword = f" [{c.get('keyword', '')}]" if c.get('keyword') else ""
                st.markdown(f"- **{c['amount']:.2f}€** (score: {score_pct}){keyword}")
        else:
            st.warning("Aucun montant détecté")

        st.markdown(f"**Total sélectionné:** `{invoice.get('total_facture', 'N/A')}€` ({invoice.get('total_status', '')})")
        st.markdown("---")

        st.markdown("**🧾 Lignes articles détectées :**")

        items_block = invoice.get("items", {}) or {}
        items = items_block.get("items", []) if items_block else []

        if items:
            st.markdown(f"- Nombre de lignes: `{len(items)}`")
            st.markdown(f"- Somme des totaux lignes: `{items_block.get('sum_line_totals')}`")


            # Première ligne (résumé rapide)
            fi = items[0]
            st.markdown("**➡️ Première ligne (résumé):**")
            st.markdown(f"- Code: `{fi.get('item_code')}`")
            st.markdown(f"- Quantité: `{fi.get('quantity')}`")
            st.markdown(f"- Unité: `{fi.get('unit')}`")
            if fi.get("units_detail") is not None:
                st.markdown(f"- Détail unités: `{fi.get('units_detail')}`")
            st.markdown(f"- Prix unitaire: `{fi.get('unit_price')}`")
            st.markdown(f"- Total ligne: `{fi.get('line_total')}`")

            # Tableau complet
            df_items = pd.DataFrame([{
                "Code": it.get("item_code"),
                "Description": it.get("description"),
                "Qté": it.get("quantity"),
                "Unité": it.get("unit"),
                "Unités détail": it.get("units_detail"),
                "PU (SOCOMO)": it.get("unit_price"),
                "PU attendu (VANDAMME)": it.get("expected_unit_price"),
                "Δ PU": it.get("delta_unit"),
                "Δ PU %": it.get("delta_unit_pct"),
                "Statut PU": it.get("item_price_status"),
                "Preset label": it.get("preset_label"),
                "Valid from": it.get("preset_valid_from"),
                "Valid to": it.get("preset_valid_to"),
                "Total ligne": it.get("line_total"),
                "OK OCR": it.get("ok"),
            } for it in items])

            st.dataframe(df_items, use_container_width=True)

            with st.expander("Voir les lignes brutes"):
                for it in items[:30]:
                    st.code(it.get("raw_line") or it.get("raw_item_line") or "")
        else:
            st.info("Aucune ligne article détectée.")

        # Champs bonus
        st.markdown("**Autres informations:**")
        st.markdown(f"- Fournisseur: _{invoice.get('supplier', 'N/A')}_")
        st.markdown(f"- N° Facture: _{invoice.get('invoice_number', 'N/A')}_")
        st.markdown(f"- Date: _{invoice.get('invoice_date', 'N/A')}_")
        st.markdown(f"- Devise: _{invoice.get('currency', 'EUR')}_")

        # Warnings
        warnings = invoice.get('warnings', [])
        if warnings:
            st.markdown("---")
            st.markdown("**⚠️ Warnings:**")
            for w in warnings:
                st.warning(w)

        # Corrections manuelles
        st.markdown("---")
        st.subheader("✏️ Corrections manuelles")

        with st.form(f"correction_form_{selected_idx}"):
            new_pe = st.text_input(
                "Corriger le PE",
                value=invoice.get('pe_selected', ''),
                placeholder="PE123456"
            )

            new_amount = st.number_input(
                "Corriger le montant total (€)",
                value=invoice.get('total_facture') or 0.0,
                min_value=0.0,
                step=0.01
            )

            if st.form_submit_button("💾 Appliquer les corrections"):
                # Appliquer les corrections
                pe_corr = new_pe if new_pe != invoice.get('pe_selected', '') else None
                amt_corr = new_amount if new_amount != invoice.get('total_facture') else None

                if pe_corr or amt_corr:
                    updated = apply_manual_correction(invoice, pe_corr, amt_corr)

                    # Recalculer comparaison
                    if st.session_state.referential_df is not None:
                        updated_list = join_with_referential(
                            [updated],
                            st.session_state.referential_df
                        )
                        updated_list = calculate_comparison(
                            updated_list,
                            tol_euros=tol_euros,
                            tol_percent=tol_percent
                        )
                        updated = updated_list[0]

                    st.session_state.invoices_data[selected_idx] = updated
                    st.success("✅ Corrections appliquées!")
                    st.rerun()

# ===============================
# PAGE: RESULTATS
# ===============================
elif page_id == "resultats":
    st.title(tr("📊 Résultats", "📊 Resultaten"))

    invoices = st.session_state.get("invoices_data", []) or []
    if not invoices:
        st.info(tr("Aucune facture traitée pour l’instant. Va dans ⚙️ Traitement.",
                   "Nog geen verwerkte facturen. Ga naar ⚙️ Verwerking."))
    else:
        # KPI
        n_total = len(invoices)
        n_price_err = sum(1 for inv_ in invoices if "PRIX_ARTICLE_ECART" in (inv_.get("flags") or []))
        n_unknown = sum(1 for inv_ in invoices if "CODE_ARTICLE_INCONNU" in (inv_.get("flags") or []))
        n_duplicates = sum(1 for inv_ in invoices if "DOUBLON" in (inv_.get("flags") or []))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(tr("Factures", "Facturen"), n_total)
        k2.metric(tr("❌ Erreurs prix", "❌ Prijsfouten"), n_price_err)
        k3.metric(tr("🟧 Codes inconnus", "🟧 Onbekende codes"), n_unknown)
        k4.metric(tr("🔁 Doublons", "🔁 Dubbels"), n_duplicates)

        tab1, tab2, tab3, tab4 = st.tabs([
            tr("📌 Vue générale", "📌 Overzicht"),
            tr("💥 Erreurs prix", "💥 Prijsfouten"),
            tr("🟧 Codes inconnus", "🟧 Onbekende codes"),
            tr("🔁 Doublons", "🔁 Dubbels"),
        ])

        # Helpers
        def inv_status_emoji(inv: dict) -> str:
            flags = inv.get("flags") or []
            if "PRIX_ARTICLE_ECART" in flags:
                return tr("❌ ECART PRIX", "❌ PRIJS-AFWIJKING")
            if "CODE_ARTICLE_INCONNU" in flags:
                return tr("🟧 CODE INCONNU", "🟧 ONBEKENDE CODE")
            if "INCOMPLET" in flags:
                return tr("⚠️ INCOMPLET", "⚠️ ONVOLLEDIG")
            return tr("✅ OK", "✅ OK")

        def _format_duplicates(inv: dict) -> str:
            dups = inv.get("duplicates") or []
            parts = []
            for d in dups:
                t_ = d.get("type", "?")
                files = d.get("other_files", []) or []
                if files:
                    parts.append(f"{t_}: " + ", ".join(files))
            return " | ".join(parts)

        def invoice_row(inv: dict) -> dict:
            return {
                "Statut": inv.get("statut_final") or inv_status_emoji(inv),
                "PE": inv.get("pe_selected") or inv.get("pe") or "",
                "Date": inv.get("invoice_date") or "",
                "Fichier": inv.get("filename") or "",
                "Total facture": inv.get("total_facture") or inv.get("invoice_total") or inv.get("total_amount") or "",
                "Attendu (CSV)": inv.get("montant_total_attendu") or inv.get("expected_total") or "",
                "Qté OCR": inv.get("quantite_total_ocr") if inv.get("quantite_total_ocr") is not None else "",
                "Unité Qté": inv.get("quantite_unite_ocr") or "",
                "Qté (CSV)": inv.get("quantite_totale_attendue") if inv.get("quantite_totale_attendue") is not None else "",
                "Delta Qté": inv.get("delta_qte") if inv.get("delta_qte") is not None else "",
                "Statut Qté": inv.get("quantite_status") or "",
                "Delta (€)": inv.get("delta"),
                "Delta (%)": inv.get("delta_pct"),
                "Flags": ", ".join(inv.get("flags") or []),
                "Liens (doublons)": _format_duplicates(inv),
                "Items écarts prix": inv.get("items_ecarts", 0),
                "Items codes inconnus": inv.get("items_unknown_codes", 0),
            }

        # Tab 1: Vue générale + détail
        with tab1:
            df = pd.DataFrame([invoice_row(inv_) for inv_ in invoices])
            st.dataframe(df, use_container_width=True)

            st.markdown("### " + tr("🔎 Ouvrir une facture (détails)", "🔎 Factuur openen (details)"))
            inv_keys = [inv_.get("pe_selected") or inv_.get("pe") or inv_.get("filename") for inv_ in invoices]
            selected_key = st.selectbox(tr("Choisis une facture", "Kies een factuur"), inv_keys)

            inv_map = {(inv_.get("pe_selected") or inv_.get("pe") or inv_.get("filename")): inv_ for inv_ in invoices}
            inv_selected = inv_map.get(selected_key)

            if inv_selected:
                st.write("**PE :**", inv_selected.get("pe_selected") or inv_selected.get("pe") or "")
                st.write("**Statut :**", inv_status_emoji(inv_selected))
                st.write("**Flags :**", ", ".join(inv_selected.get("flags") or []))

                items = (inv_selected.get("items") or {}).get("items") or []
                if items:
                    df_items = pd.DataFrame([{
                        "Code": it.get("item_code"),
                        "Description": it.get("description"),
                        "Qté": it.get("quantity"),
                        "PU (SOCOMO)": it.get("unit_price"),
                        "PU attendu": it.get("expected_unit_price"),
                        "Δ PU": it.get("delta_unit"),
                        "Δ PU %": it.get("delta_unit_pct"),
                        "Statut PU": it.get("item_price_status"),
                        "Total ligne": it.get("line_total"),
                    } for it in items])

                    show_only_ecarts = st.checkbox(
                        tr("Afficher uniquement les écarts de prix", "Alleen prijsafwijkingen tonen"),
                        value=True
                    )
                    if show_only_ecarts:
                        df_items = df_items[df_items["Statut PU"] == "ECART"]

                    st.dataframe(df_items, use_container_width=True)
                else:
                    st.info(tr("Aucune ligne produit trouvée.", "Geen productregels gevonden."))

                st.markdown("---")
                show_all_lines = st.checkbox(
                    tr("Afficher le détail lignes pour TOUTES les factures", "Detailregels voor ALLE facturen tonen"),
                    value=False
                )

                def render_lines_block(one_inv: dict):
                    pe = one_inv.get("pe_selected") or one_inv.get("pe") or "PE?"
                    with st.expander(f"🧾 {tr('Détail lignes', 'Detailregels')} (PE {pe})", expanded=False):
                        if one_inv.get("line_comparison") is not None and len(one_inv.get("line_comparison")) > 0:
                            st.dataframe(one_inv["line_comparison"], use_container_width=True)
                        else:
                            st.info(tr(
                                "Pas de comparaison lignes : PE manquante / PE pas trouvée / lignes OCR non détectées.",
                                "Geen regelvergelijking: PE ontbreekt / PE niet gevonden / OCR-regels niet gedetecteerd."
                            ))

                if show_all_lines:
                    for inv_ in invoices:
                        render_lines_block(inv_)
                else:
                    render_lines_block(inv_selected)

        # Tab 2: Erreurs prix
        with tab2:
            rows = []
            for inv_ in invoices:
                if "PRIX_ARTICLE_ECART" not in (inv_.get("flags") or []):
                    continue
                pe = inv_.get("pe_selected") or inv_.get("pe") or ""
                items = (inv_.get("items") or {}).get("items") or []
                for it in items:
                    if (it.get("item_price_status") or "").upper() != "ECART":
                        continue
                    rows.append({
                        "PE": pe,
                        "Code": it.get("item_code"),
                        "Description": it.get("description"),
                        "PU (SOCOMO)": it.get("unit_price"),
                        "PU attendu": it.get("expected_unit_price"),
                        "Δ PU": it.get("delta_unit"),
                        "Δ PU %": it.get("delta_unit_pct"),
                    })

            if not rows:
                st.success(tr("Aucune erreur de prix détectée ✅", "Geen prijsfouten gedetecteerd ✅"))
            else:
                df_ecarts = pd.DataFrame(rows)
                st.dataframe(df_ecarts, use_container_width=True)

                st.markdown("### " + tr("📌 Top codes en erreur", "📌 Top foutcodes"))
                top_codes = (df_ecarts.groupby("Code")
                             .size()
                             .sort_values(ascending=False)
                             .head(15)
                             .reset_index(name="Occurrences"))
                st.dataframe(top_codes, use_container_width=True)

        # Tab 3: Codes inconnus
        with tab3:
            rows = []
            for inv_ in invoices:
                if "CODE_ARTICLE_INCONNU" not in (inv_.get("flags") or []):
                    continue
                pe = inv_.get("pe_selected") or inv_.get("pe") or ""
                items = (inv_.get("items") or {}).get("items") or []
                for it in items:
                    if (it.get("item_price_status") or "").upper() != "CODE_INCONNU":
                        continue
                    rows.append({
                        "PE": pe,
                        "Code inconnu": it.get("item_code"),
                        "Description": it.get("description"),
                        "PU (SOCOMO)": it.get("unit_price"),
                    })

            if not rows:
                st.success(tr("Aucun code inconnu ✅", "Geen onbekende codes ✅"))
            else:
                df_unk = pd.DataFrame(rows)
                st.dataframe(df_unk, use_container_width=True)

                st.markdown("### " + tr("📌 Top codes inconnus", "📌 Top onbekende codes"))
                top_unk = (df_unk.groupby("Code inconnu")
                           .size()
                           .sort_values(ascending=False)
                           .head(20)
                           .reset_index(name="Occurrences"))
                st.dataframe(top_unk, use_container_width=True)

        # Tab 4: Doublons
        with tab4:
            dups = [inv_ for inv_ in invoices if "DOUBLON" in (inv_.get("flags") or [])]
            if not dups:
                st.success(tr("Aucun doublon ✅", "Geen dubbels ✅"))
            else:
                df_dups = pd.DataFrame([invoice_row(inv_) for inv_ in dups])
                st.dataframe(df_dups, use_container_width=True)


# ===============================
# PAGE: EXPORT
# ===============================
elif page_id == "export":
    st.title(tr("📦 Export ZIP", "📦 ZIP exporteren"))

    if not st.session_state.invoices_data:
        st.warning(tr("⚠️ Aucune donnée à exporter. Lance d'abord le traitement.", "⚠️ Geen data om te exporteren. Start eerst de verwerking."))
        st.stop()

    invoices = st.session_state.invoices_data
    st.markdown(f"**{len(invoices)} factures à exporter**")

    st.markdown("""
    Ce téléchargement génère **un seul fichier ZIP** qui peut contenir :
    - PDFs renommés (PE.pdf)
    - Excel complet 
    """)

    # ✅ Export minimal : rapport complet + PDFs renommés
    zip_buf = export_bundle_zip(
        st.session_state.invoices_data,
        include_full_excel=True,
        include_extraits_excel=False,
        include_pdfs=True,
        include_erreurs_par_pe=False,
        include_csv=False,
    )

    st.download_button(
        tr("📦 Télécharger ZIP", "📦 ZIP downloaden"),
        data=zip_buf,
        file_name=generate_report_filename("export_VD", date_only=True) + ".zip",
        mime="application/zip",
        type="primary",
        use_container_width=True
    )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(tr("**100% Local** - Aucune donnée envoyée sur internet", "**100% Lokaal** - Geen gegevens worden naar internet gestuurd"))
st.sidebar.markdown("v1.2 - Contrôle Factures")
st.sidebar.markdown("**SOCOMO/VANDAMME 🍌**")
