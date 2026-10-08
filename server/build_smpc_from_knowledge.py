"""
Сборка SmPC-базы drug_info из knowledge-файлов psychiatry_qa20k
(01_antidepressants, 02_antipsychotics, 03_mood_stabilizers).

Каждый препарат -> data/smpc/<inn>.json. Обогащаем red_flags из
06_special_populations (серотониновый синдром, ЗПС и т.п.).

Запуск: py -3.13 build_smpc_from_knowledge.py
"""

import json
import os
import re

KD = r"C:\ML_learning\psychiatry_qa20k\knowledge"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "smpc")

RED_FLAGS = {
    "ssri_snri": [
        "суицидальные мысли в первые недели у пациентов до 25 лет",
        "серотониновый синдром при комбинациях (клонус, гиперрефлексия, температура)",
        "гипонатриемия у пожилых (первый месяц)",
        "сыпь/аллергия/кровоточивость — оценить и отменить при необходимости",
    ],
    "antipsychotic": [
        "ЗПС: акатизия, дистония, паркинсонизм (не путать с обострением психоза)",
        "злокачественный нейролептический синдром: температура + ригидность + КФК",
        "метаболический синдром: вес, глюкоза, липиды — контроль каждые 3-6 мес",
        "удлинение QTc: ЭКГ при комбинациях и дозах выше средних",
    ],
    "mood_stabilizer": [
        "см. индивидуальные red_flags в инструкции конкретного препарата",
    ],
}

CLASS_FLAG = re.compile(r"СИОЗС|SSRI|СИОЗСН|SNRI|антидепрессант|трициклик", re.I)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower().split("_")[0])


def main():
    os.makedirs(OUT, exist_ok=True)
    made = 0
    for fn in ["01_antidepressants.json", "02_antipsychotics.json", "03_mood_stabilizers.json"]:
        d = json.load(open(os.path.join(KD, fn), encoding="utf-8"))
        for dr in d.get("drugs", []):
            name_en = dr.get("inn_en") or dr.get("id")
            if not name_en:
                continue
            doc = {
                "drug": name_en,
                "inn_ru": dr.get("inn_ru", ""),
                "class": dr.get("class", ""),
                "mechanism": dr.get("mechanism", ""),
                "indications": dr.get("indications", ""),
                "dosing": dr.get("dosing") or dr.get("doses") or dr.get("initiation", ""),
                "target_dose": dr.get("target_dose", ""),
                "max_dose": dr.get("max_dose", ""),
                "titration": dr.get("titration", ""),
                "adverse_effects": dr.get("adverse_effects", ""),
                "monitoring": dr.get("monitoring", ""),
                "cautions": dr.get("cautions", ""),
                "withdrawal": dr.get("withdrawal", dr.get("taper", "")),
                "source_file": fn,
                "disclaimer": "Собрано из knowledge-корпуса psychiatry_qa20k; перед клиническим применением сверить с официальной инструкцией (SmPC).",
            }
            # red flags по классу
            cls = str(dr.get("class", ""))
            if CLASS_FLAG.search(cls):
                doc["red_flags"] = RED_FLAGS["ssri_snri"]
            elif "антипсихот" in cls.lower() or "antipsychot" in cls.lower():
                doc["red_flags"] = RED_FLAGS["antipsychotic"]
            # убираем пустые поля
            doc = {k: v for k, v in doc.items() if v not in ("", [], {}, None)}
            path = os.path.join(OUT, slug(name_en) + ".json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=1)
            made += 1
            print(f"  + {slug(name_en)}.json  ({name_en})")
    print(f"\nитого: {made} препаратов в {OUT}")


if __name__ == "__main__":
    main()
