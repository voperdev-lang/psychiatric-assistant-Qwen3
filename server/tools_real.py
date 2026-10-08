"""
Реальные инструменты для server.py.

search_icd11      — BM25 по icd11_cddr_chunks.jsonl (готово к работе сейчас)
search_guidelines — тот же BM25-движок; заработает, когда в data/guidelines/
                    появятся .txt файлы гайдлайнов (МЗ РФ / NICE / Maudsley)
drug_info         — локальная SmPC-база: data/smpc/<drug>.json
calculate         — арифметика (в server.py)

Индекс строится один раз при старте и держится в памяти.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

GUIDELINES_DIR = DATA_DIR / "guidelines"
SMPC_DIR = DATA_DIR / "smpc"

# ICD-11 CDDR: основной файл (server/*.jsonl), извлечённый текст PDF или jsonl в data/
ICD11_CANDIDATES = [
    BASE_DIR / "icd11_cddr_chunks.jsonl",
    DATA_DIR / "icd11_cddr_chunks.jsonl",
    DATA_DIR / "icd11_cddr_2024_en.txt",
    Path(r"C:\ML_learning\research\psychiatry_assistant\server\icd11_cddr_chunks.jsonl"),
]

# ------------------------- токенизация -------------------------

_WORD = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)

# Русско-английский словарь частых клинических терминов: вопрос врача на русском,
# корпус на английском. Расширяй по мере эксплуатации.
RU_EN_TERMS = {
    "депрессия": "depression", "депрессивн": "depression",
    "тревог": "anxiety", "паническ": "panic",
    "шизофрени": "schizophrenia", "биполярн": "bipolar",
    "мани": "mania", "гипомани": "hypomania",
    "обсессивн": "obsessive", "окомпульсивн": "compulsive",
    "птср": "ptsd", "посттравматическ": "posttraumatic",
    "бессонниц": "insomnia", "инсомни": "insomnia",
    "расстройств": "disorder", "синдром": "syndrome",
    "суицид": "suicide", "самоповрежден": "selfharm",
    "резистентн": "resistant", "терапи": "treatment",
    "препарат": "drug", "доз": "dose", "дозировк": "dose",
    "критери": "diagnostic criteria", "диагноз": "diagnosis",
    "дифференциальн": "differential", "беременност": "pregnancy",
    "ребен": "children", "подростк": "adolescents",
    "пожил": "older adults", "деменц": "dementia",
    "эпилепси": "epilepsy", "печен": "hepatic", "почк": "renal",
    "отмен": "withdrawal", "побочн": "adverse",
}


def tokenize(text: str) -> list[str]:
    text = text.lower()
    # разворачиваем русские термины в английские эквиваленты
    for ru, en in RU_EN_TERMS.items():
        if ru in text:
            text += " " + en
    return _WORD.findall(text)


# ------------------------- BM25-индекс -------------------------


class ChunkIndex:
    """BM25 поверх jsonl-чанков или txt-файлов. Строится один раз при старте."""

    def __init__(self, name: str):
        self.name = name
        self.docs: list[dict] = []  # {"text": ..., "meta": ...}
        self._bm25 = None

    def add_jsonl(self, path: Path, meta: str):
        with open(path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.docs.append({"text": obj["text"], "meta": f"{meta}#чанк{i}"})

    def add_txt_dir(self, folder: Path, meta: str):
        if not folder.exists():
            return
        for fp in sorted(folder.glob("*.txt")):
            text = fp.read_text(encoding="utf-8", errors="ignore")
            # режем длинные файлы на абзацы ~2000 символов
            for j, part in enumerate(
                [text[k:k + 2000] for k in range(0, len(text), 1800)]
            ):
                if len(part.strip()) > 200:
                    self.docs.append(
                        {"text": part, "meta": f"{meta}:{fp.stem}#part{j}"}
                    )

    def build(self):
        if not self.docs:
            self._bm25 = None
            return self
        from rank_bm25 import BM25Okapi

        corpus = [tokenize(d["text"]) for d in self.docs]
        self._bm25 = BM25Okapi(corpus)
        return self

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in ranked[:top_k]:
            if scores[i] <= 0:
                continue
            hits.append(
                {
                    "source": self.docs[i]["meta"],
                    "text": self.docs[i]["text"][:1400],
                    "score": round(float(scores[i]), 2),
                }
            )
        return hits


# ------------------------- SmPC -------------------------


class DrugDB:
    """SmPC-база: data/smpc/<drug>.json. Формат файла см. data/smpc/sertraline.json."""

    def __init__(self, folder: Path):
        self.folder = folder
        self._cache: dict[str, dict] = {}

    def lookup(self, drug: str) -> dict:
        key = re.sub(r"[^a-z0-9]", "", drug.lower())
        if not key:
            return {"error": "пустое название препарата"}
        if key in self._cache:
            return self._cache[key]
        path = self.folder / f"{key}.json"
        if not path.exists():
            return {
                "drug": drug,
                "status": "not_found",
                "note": (
                    f"Инструкция '{drug}' отсутствует в базе SmPC. НЕ называть дозы по памяти: "
                    "порекомендовать свериться с официальной инструкцией."
                ),
            }
        doc = json.loads(path.read_text(encoding="utf-8"))
        self._cache[key] = {"status": "ok", **doc}
        return self._cache[key]


# ------------------------- синглтоны -------------------------

def build_indexes() -> dict:
    icd11 = ChunkIndex("icd11_cddr")
    icd11_file = next((p for p in ICD11_CANDIDATES if p.exists()), None)
    if icd11_file and icd11_file.suffix == ".jsonl":
        icd11.add_jsonl(icd11_file, meta="МКБ-11 CDDR")
    elif icd11_file:
        # извлечённый текст CDDR: режем на разделы по двойным переводам строк
        text = icd11_file.read_text(encoding="utf-8", errors="ignore")
        for j, part in enumerate(text.split("\n\n")):
            part = part.strip()
            if len(part) > 300:
                icd11.docs.append({"text": part, "meta": f"МКБ-11 CDDR 2024#sec{j}"})
    icd11.build()

    guidelines = ChunkIndex("guidelines")
    guidelines.add_txt_dir(GUIDELINES_DIR, meta="гайдлайн")
    guidelines.build()

    drugs = DrugDB(SMPC_DIR)

    print(
        f"[tools_real] индексы: icd11={len(icd11.docs)} чанков, "
        f"guidelines={len(guidelines.docs)} чанков, smpc={len(list(SMPC_DIR.glob('*.json'))) if SMPC_DIR.exists() else 0} препаратов"
    )
    return {"icd11": icd11, "guidelines": guidelines, "drugs": drugs}


if __name__ == "__main__":
    idx = build_indexes()
    # самопроверка ретрива по всем трём источникам
    tests = [
        ("icd11", "bipolar disorder hypomania diagnostic criteria"),
        ("icd11", "биполярное расстройство критерии гипомании"),
        ("guidelines", "first line treatment moderate depression SSRI"),
        ("guidelines", "treatment resistant depression next step"),
        ("guidelines", "lamotrigine titration rash"),
    ]
    for src, q in tests:
        hits = idx[src].search(q, top_k=2)
        print(f"\n[{src}] Q: {q} -> {len(hits)} hits")
        for h in hits:
            print(f"  [{h['score']}] {h['source']}: {h['text'][:90]}...")
    print("\ndrug_info sertraline:", json.dumps(idx["drugs"].lookup("sertraline"), ensure_ascii=False)[:150])
    print("drug_info sertraline:", json.dumps(idx["drugs"].lookup("carbamazepine"), ensure_ascii=False)[:150])
