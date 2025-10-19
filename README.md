# 🧩 AI Matching

End-to-end supervision matching demo used in the `ai_matching/` package.  
The repository is Colab-ready so anyone can run generation, training, and scoring jobs in a hosted notebook.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Clem085/AI_Matching/blob/main/Collab_Setup.ipynb)

---

## 🚀 Run in Google Colab

Click the badge above or open `Collab_Setup.ipynb` directly in Colab. The notebook:
- Clones this repository into the Colab runtime.
- Installs the lightweight dependencies from `requirements.txt`.
- Executes `python ai_matching/supervision_tool.py generate build train score` to generate synthetic data, train the model, and score matches.
- Provides links to the richer analysis notebook (`ai_matching/learn_from_outputs.ipynb`) once the assets are created.

### Manual Colab steps

```python
!git clone https://github.com/Clem085/AI_Matching.git /content/AI_Matching
%cd /content/AI_Matching
!pip install -r requirements.txt
!python ai_matching/supervision_tool.py generate build train score
```

---

## 📁 Project layout

- `ai_matching/` – source package, CLI (`supervision_tool.py`), synthetic data generator, model training code, and analysis notebooks.
- `requirements.txt` – minimal dependency set validated in Colab (pandas, numpy, scikit-learn, matplotlib, joblib, scipy).
- `Collab_Setup.ipynb` – quick-launch notebook for Colab users.

All paths inside the package use relative paths (`Path(...)`) so they run the same locally or in Colab.

---

## 🧪 Local usage

```bash
python -m venv .venv
source .venv/bin/activate        # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python ai_matching/supervision_tool.py generate build train score
```

Outputs (CSV + model) are written inside `ai_matching/` by default.

---

## ✅ Colab readiness checklist

- [x] **requirements.txt** present with minimal packages.
- [x] **Clear entry point** – run `ai_matching/supervision_tool.py`.
- [x] **Sample data** – `generate` command produces synthetic CSVs.
- [x] **No hardcoded local paths** – everything uses repo-relative paths.
- [x] **No secrets/environment requirements** – nothing sensitive baked in.

Enjoy building and demoing the matcher!
