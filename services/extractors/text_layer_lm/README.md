# Character 5-gram LMs for PDF text-layer quality gating (en/zh/ru/vi).
#
# Data: Wikipedia only (~0.15–0.3 MB text/lang via MediaWiki API).
#   pip install requests
#   python scripts/train_text_layer_lm.py
#
# Runtime: *.npz + thresholds.json (no download at serve time).
