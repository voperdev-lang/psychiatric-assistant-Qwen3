# Определения инструментов QA-ассистента психиатра (Qwen3, Hermes-style tool calling).
# Описания инструментов — часть промпта: от них зависит, КОГДА модель решает вызвать
# инструмент и с какими аргументами. Пишем их так же тщательно, как сами ответы.
#
# Правила, которые здесь зашиты:
#  - запросы в поисковые инструменты — на английском (корпуса DSM-5 / ICD-11 / гайдлайны англоязычные);
#  - drug_info — единственный источник доз, старт/максимум, взаимодействий;
#  - calculate — только арифметика, без клинической логики;
#  - имена функций: [a-z0-9_] (лёгкий вызов из Python-диспетчера на сервере).

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "drug_info",
            "description": (
                "Официальная инструкция препарата (SmPC): показания, стартовые и максимальные дозы, "
                "титрация, противопоказания, взаимодействия, коррекция при почечной/печёночной "
                "недостаточности, беременность. Вызывай ПРИ ЛЮБОМ упоминании дозы, смене препарата "
                "или комбинации — не отвечай о дозах по памяти."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "drug": {
                        "type": "string",
                        "description": "МНН препарата латиницей, напр. 'sertraline', 'lamotrigine'",
                    },
                },
                "required": ["drug"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_guidelines",
            "description": (
                "Поиск по клиническим рекомендациям и гайдлайнам (МЗ РФ, NICE, Maudsley): линии терапии, "
                "алгоритмы при резистентности, скрининг перед назначением, ведение особых групп. "
                "Вызывай для вопросов 'что назначить', 'что дальше', 'что проверить перед стартом'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Поисковый запрос на английском, напр. "
                            "'first line treatment moderate depression', 'treatment resistant depression next step'"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_icd11",
            "description": (
                "Поиск по МКБ-11 CDDR: диагностические критерии, дифференциальный диагноз, границы "
                "расстройств. Вызывай для вопросов классификации и дифференциации ('это ГТР или депрессия')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Поисковый запрос на английском, напр. "
                            "'generalized anxiety disorder diagnostic requirements', 'bipolar I differential'"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Калькулятор: только арифметика (числа и + - * / ( )). Для пересчёта доз (мг/кг), "
                "градиентов титрации, коррекций уровня препарата. Не знает клиники — используй "
                "только после того, как формула получена из drug_info или гайдлайна."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Арифметическое выражение, напр. '25 * 2' или '48 / 1.73'",
                    },
                },
                "required": ["expression"],
            },
        },
    },
]

# --- Самопроверка имён (валидация перед использованием в датасете/сервере) ---
import re

NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

def validate(tools=TOOLS):
    names = []
    for t in tools:
        fn = t["function"]
        name = fn["name"]
        assert NAME_RE.match(name), f"имя '{name}' не соответствует ^[a-z][a-z0-9_]*$"
        assert "description" in fn and len(fn["description"]) > 20, f"у '{name}' нет описания"
        assert "parameters" in fn, f"у '{name}' нет схемы параметров"
        names.append(name)
    assert len(names) == len(set(names)), "дубликаты имён"
    return names

if __name__ == "__main__":
    from transformers import AutoTokenizer

    names = validate()
    print("Инструменты валидны:", names)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    messages = [
        {"role": "system", "content": "Ты — клинический ассистент психиатра. Отвечай на русском."},
        {"role": "user", "content": "Пациент на карбамазепине, думаю добавить кветиапин 300 мг. Ок?"},
    ]
    text = tok.apply_chat_template(messages, tools=TOOLS, add_generation_prompt=True, tokenize=False)
    print("\nПромпт собрался, длина:", len(text), "символов")
    assert "<tools>" in text and "drug_info" in text
    print("Рендер с tools работает.")
