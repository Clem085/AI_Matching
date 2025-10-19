# 🧩 AI_Matching

AI Matching is an intelligent matching system for supervised learning workflows.  
This repository provides command-line and Colab support for running experiments, training, and scoring.

---

## 🚀 Quick Start (Colab)

You can run everything in Google Colab — no installation required.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/Clem085/AI_Matching/blob/main/Collab_Setup.ipynb)

### Steps for manual use in Colab (if not using the badge)
```python
!git clone -b main https://github.com/Clem085/AI_Matching.git /content/AI_Matching
%cd /content/AI_Matching
!pip install -r requirements.txt
!python supervision_tool.py generate build train score
