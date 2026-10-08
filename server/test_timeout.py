# Тест длинных запросов: ловим 502 на тяжёлом вопросе и меряем время каждого этапа
import json
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server  # noqa: E402

QUESTIONS = [
    ("короткий", "Привет"),
    ("средний", "Какая стартовая доза сертралина при паническом расстройстве?"),
    ("длинный", "Пациентка 32 лет, депрессивный эпизод средней тяжести, первый эпизод. "
                "Биполярное расстройство у матери. Ранее психиатрического лечения не было. "
                "Соматика: гипотиреоз, принимает левотироксин. Планируем флуоксетин. "
                "Какие проверки необходимы перед стартом терапии и с какой дозы начинать?"),
    ("очень длинный", "Пациент 45 лет, хронический алкоголизм, сейчас abstinent 5 дней. "
                "Развивается тремор, потливость, тревога, бессонница 2 дня. Давление 150/95, пульс 100. "
                "Эпилептических приступов не было. Чем купировать риск алкогольной абстиненции, "
                "какую схему выбрать, какие препараты противопоказаны и что мониторить?"),
]

for label, q in QUESTIONS:
    t0 = time.time()
    try:
        resp = server.run_agent(q)
        dt = time.time() - t0
        tools = [tc["tool"] for tc in resp.tool_calls_made]
        print(f"[{label}] OK {dt:.0f}s, tools={tools}, len(reply)={len(resp.reply)}")
        print(f"   {resp.reply[:150].replace(chr(10), ' ')}")
    except Exception as e:
        dt = time.time() - t0
        print(f"[{label}] FAIL {dt:.0f}s: {type(e).__name__}: {str(e)[:200]}")
    print()
