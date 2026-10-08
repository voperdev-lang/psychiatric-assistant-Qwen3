"""
Извлечение текста из PDF гайдлайнов в .txt для BM25-индекса (tools_real.py).

Вход:  data/guidelines/src/*.pdf
Выход: data/guidelines/<имя>.txt  (те самые файлы, которые читает ChunkIndex.add_txt_dir)
"""

import os
from pypdf import PdfReader

SRC = os.path.join(os.path.dirname(__file__), "src")
DST = os.path.dirname(__file__)

for fname in sorted(os.listdir(SRC)):
    if not fname.lower().endswith(".pdf"):
        continue
    out_name = os.path.splitext(fname)[0] + ".txt"
    out_path = os.path.join(DST, out_name)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10_000:
        print("skip:", out_name)
        continue
    try:
        reader = PdfReader(os.path.join(SRC, fname))
        parts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            parts.append(t)
        text = "\n\n".join(parts)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"{out_name}: {len(reader.pages)} стр., {len(text)//1024} KB текста")
    except Exception as e:
        print(f"FAIL {fname}: {type(e).__name__} {e}")
