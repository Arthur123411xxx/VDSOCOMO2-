# 📄 Invoice Control App - Contrôle de Factures Scannées

Application locale Python/Streamlit pour contrôler des factures scannées (PDF image) en les comparant à un référentiel interne.

## ✨ Fonctionnalités

- **OCR robuste** : Extraction de texte depuis des PDFs scannés via Tesseract
- **Extraction PE** : Détection automatique des codes PE (PE123456) avec scoring
- **Extraction montants** : Détection intelligente du total facture (NET A PAYER, TOTAL TTC, etc.)
- **Comparaison référentiel** : Jointure par PE et calcul des écarts
- **Détection doublons** : Identification des PE/factures en double
- **Corrections manuelles** : Interface pour corriger PE et montants avec recalcul instantané
- **Exports** : Rapport Excel multi-onglets + CSV

## 🔧 Installation

### 1. Prérequis système

#### Tesseract OCR

**Windows:**
```bash
# Télécharger l'installateur depuis:
# https://github.com/UB-Mannheim/tesseract/wiki
# Ajouter au PATH: C:\Program Files\Tesseract-OCR
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-fra poppler-utils
```

#### Poppler (pour pdf2image)

**Windows:**
```bash
# Télécharger depuis: https://github.com/osber/poppler-windows/releases
# Ajouter au PATH: C:\path\to\poppler\bin
```

**macOS:**
```bash
brew install poppler
```

**Linux:**
```bash
sudo apt-get install poppler-utils
```

### 2. Installation Python

```bash
# Créer un environnement virtuel (recommandé)
python -m venv venv

# Activer l'environnement
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Vérification de l'installation

```bash
# Vérifier Tesseract
tesseract --version

# Vérifier Poppler
pdftoppm -h
```

## 🚀 Lancement

```bash
streamlit run app.py
```

L'application s'ouvrira automatiquement dans votre navigateur à l'adresse `http://localhost:8501`

## 📁 Structure du projet

```
python-invoice-app/
├── app.py              # Interface Streamlit principale
├── ocr.py              # PDF → images → prétraitement → OCR
├── extract.py          # Extraction PE + montants + champs bonus
├── compare.py          # Jointure référentiel + écarts + doublons
├── export.py           # Export Excel multi-onglets + CSV
├── utils.py            # Utilitaires (normalisation, scoring, parsing)
├── requirements.txt    # Dépendances Python
├── README.md           # Ce fichier
└── templates/          # (Futur) Templates fournisseurs
```

## 📊 Format du référentiel

Le fichier CSV/XLSX du référentiel doit contenir au minimum :

| Colonne | Description | Obligatoire |
|---------|-------------|-------------|
| PE | Code PE (ex: PE123456) | ✅ Oui |
| montant_total_attendu | Montant attendu | ✅ Oui |
| devise | Devise (EUR par défaut) | ❌ Non |
| client | Nom du client | ❌ Non |
| date_validite | Date de validité | ❌ Non |
| commentaire | Notes | ❌ Non |

## 🔍 Statuts de comparaison

| Statut | Description |
|--------|-------------|
| ✅ OK | Écart dans les tolérances |
| ⚠️ ECART | Écart hors tolérances |
| ❓ INCOMPLET | Montant facture ou attendu manquant |
| 🔴 MANQUANT_PE | Aucun PE détecté |
| 🟡 MULTI_PE | Plusieurs PE détectés |
| 🟠 TOTAL_AMBIGU | Ambiguïté sur le montant total |
| 👥 DOUBLON_PE_LOT | Même PE dans plusieurs factures |

## ⚙️ Paramètres

- **Tolérance €** : Écart absolu accepté (défaut: 0.05€)
- **Tolérance %** : Écart relatif accepté (défaut: 0.5%)
- **Prétraitement renforcé** : Amélioration d'image pour OCR difficile

## 🔒 Confidentialité

**100% local** - Aucune donnée n'est envoyée sur internet. Tout le traitement se fait sur votre machine.

## 🐛 Dépannage

### "Tesseract not found"
Vérifiez que Tesseract est installé et dans le PATH système.

### "poppler not found"
Installez poppler-utils (Linux/Mac) ou téléchargez poppler (Windows).

### OCR de mauvaise qualité
Activez le "Prétraitement renforcé" dans les paramètres.

## 📝 Licence

Usage interne uniquement.
