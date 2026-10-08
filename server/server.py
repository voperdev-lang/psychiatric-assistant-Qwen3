"""
QA-ассистент психиатра: FastAPI + HF Inference Endpoint (Qwen3) + tool calling.

Архитектура:
  браузер (static/index.html) -> этот сервер -> HF Inference Endpoint (OpenAI API)
                                     ^                    |
                                     |   tool_calls       v
                                     +--- диспетчер инструментов (drug_info,
                                          search_guidelines, search_icd11, calculate)

Инструменты исполняются ЗДЕСЬ, их результат возвращается модели как role="tool".
Цикл ограничен MAX_TOOL_ROUNDS, чтобы модель не зациклилась.

Запуск:
    pip install fastapi uvicorn openai
    export HF_ENDPOINT_URL="https://xxxx.endpoints.huggingface.cloud/v1"
    export HF_TOKEN="hf_..."
    python server.py            # http://localhost:8000
"""

import json
import os
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel, Field

# --- загрузка .env без зависимостей ---
from pathlib import Path as _P
_env_file = _P(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

sys.path.insert(0, str(Path(__file__).parent.parent))  # tools.py лежит на уровень выше
from tools import TOOLS

# ------------------------- конфигурация -------------------------

ENDPOINT_URL = os.environ.get(
    "HF_ENDPOINT_URL", ""
).rstrip("/")  # напр. https://xxxx.endpoints.huggingface.cloud/v1
MODEL_NAME = os.environ.get("HF_MODEL_NAME", "")  # обычно любое/имя модели; см. доку endpoint'а
HF_TOKEN = os.environ.get("HF_TOKEN", "")
MAX_TOOL_ROUNDS = 4          # максимум циклов модель->инструмент->модель на один вопрос
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "8196"))   # потолок на ответ; при нехватке контекста урежется автоматически
MAX_TOOL_RESULT_CHARS = int(os.environ.get("MAX_TOOL_RESULT_CHARS", "6000"))  # лимит на один результат инструмента
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS", "16000"))  # предохранитель: слишком длинный вопрос режем ДО отправки
TEMPERATURE = 0.7            # для клинического ассистента: низкая, минимум креатива

SYSTEM_PROMPT = (
    "Ты — клинический ассистент психиатра для врачей. Правила:\n"
    "1. Дозы, титрацию, взаимодействия и показания проверяй инструментом drug_info — "
    "не отвечай по памяти.\n"
    "2. Диагностические критерии и дифференциальный диагноз — search_icd11.\n"
    "3. Линии терапии, резистентность, ведение особых групп — search_guidelines.\n"
    "4. Арифметику (пересчёт доз, градиенты) — calculate.\n"
    "5. Если инструментов недостаточно или данных о пациенте не хватает — прямо скажи "
    "об этом и перечисли, что нужно уточнить. Не выдумывай факты, дозы и PMID.\n"
    "6. Вопросы вне психиатрии — вежливо отклони, укажи нужного специалиста.\n"
    "7. ФОРМАТ ОТВЕТА — строго один финальный ответ на вопрос врача:\n"
    "   - никакой служебной разметки (XML-теги, секции, slug-подчёркивания вида "
    "биполярное_I_расстройство — запрещены, пиши обычными словами);\n"
    "   - обычный связный текст на русском: краткая оценка → что проверить → вывод;\n"
    "   - без внутренних меток, без полей 'Final Answer:', без заголовков-тегов.\n"
    "8. Заканчивай одним напоминанием, что решение принимает лечащий врач.\n"
    "/no_think"  # мягкий выключатель thinking-режима Qwen3: длинные вопросы иначе уходят в <think> и обрезаются
)

if not ENDPOINT_URL or not HF_TOKEN:
    raise SystemExit(
        "Задайте переменные окружения HF_ENDPOINT_URL и HF_TOKEN.\n"
        "  export HF_ENDPOINT_URL='https://xxxx.endpoints.huggingface.cloud/v1'\n"
        "  export HF_TOKEN='hf_...'"
    )

client = OpenAI(
    base_url=ENDPOINT_URL,
    api_key=HF_TOKEN,
    max_retries=2,
    timeout=httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=15.0),
)

app = FastAPI(title="Psychiatry QA Assistant")

# ------------------------- инструменты -------------------------

_SAFE_EXPR = re.compile(r"^[\d\s\.\+\-\*/\(\)]+$")

# Реальные индексы (BM25 по МКБ-11 CDDR + VA/DoD гайдлайны + SmPC-база)
import tools_real  # noqa: E402

IDX = tools_real.build_indexes()


def tool_drug_info(drug: str) -> dict:
    """Поиск в локальной SmPC-базе (data/smpc/*.json, собрана из knowledge-корпуса)."""
    return IDX["drugs"].lookup(drug)


def _search_index(index_key: str, query: str, top_k: int = 3) -> dict:
    hits = IDX[index_key].search(query, top_k=top_k)
    return {
        "source": index_key,
        "query": query,
        "results": hits,
        "note": "Цитируй только из results; если пусто — так и скажи, не восполняй по памяти.",
    }


def tool_calculate(expression: str) -> dict:
    expr = expression.strip()
    if not _SAFE_EXPR.match(expr) or not re.search(r"\d", expr):
        return {"error": "выражение должно содержать только числа и + - * / ( )"}
    try:
        # eval после жёсткой проверки символов; builtins вырезаны
        value = eval(expr, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as e:
        return {"error": f"не удалось вычислить: {e}"}
    return {"expression": expr, "result": value}


def execute_tool(name: str, arguments: dict) -> dict:
    """Диспетчер: имя из tool_call -> реальный инструмент."""
    try:
        if name == "drug_info":
            return tool_drug_info(drug=str(arguments.get("drug", "")).strip())
        if name == "search_guidelines":
            return _search_index("guidelines", str(arguments.get("query", "")).strip())
        if name == "search_icd11":
            return _search_index("icd11", str(arguments.get("query", "")).strip())
        if name == "calculate":
            return tool_calculate(str(arguments.get("expression", "")).strip())
        return {"error": f"неизвестный инструмент: {name}"}
    except Exception as e:
        return {"error": f"ошибка исполнения инструмента: {e}"}


# ------------------------- цикл с инструментами -------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=100000)


class ChatResponse(BaseModel):
    reply: str
    tool_calls_made: list[dict]


# ------------------------- очистка ответа от XML/служебной разметки -------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S)        # пустые/остаточные think-блоки
_ORPHAN_CLOSE_THINK = re.compile(r"</?think>")
_UNCLOSED_THINK = re.compile(r"<think>.*\Z", re.S)   # ответ оборвался внутри размышлений

# Открывающий тег <tag ...> или </tag> — срезаем вместе с атрибутами.
# Закрывающие теги в датасете часто терялись, поэтому нельзя просто удалять
# только парные: срезаем ЛЮБые угловые конструкции, похожие на теги.
_ANY_TAG = re.compile(r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9_-]*(\s[^<>]*)?>")

# Теги, которые превращаем в markdown-заголовок: <assessment>, <next-steps> и т.п.
_KNOWN_SECTIONS = {
    "assessment": "**Оценка**",
    "next-steps": "**Что проверить / следующие шаги**",
    "recommendations": "**Рекомендации**",
    "plan": "**План**",
    "monitoring": "**Мониторинг**",
    "red-flags": "**Красные флаги**",
    "summary": "**Итог**",
}

# slug-токены датасета: биполярное_I_расстройство, риск_самооценки и т.п. —
# внутренние метки генератора, в человеческом тексте им не место.
_SLUG = re.compile(r"\b[а-яёa-z]+(?:_[а-яёa-z0-9]+)+\b", re.IGNORECASE)

# Повторяющийся дисклеймер датасета — оставляем максимум один в конце.
_DISCLAIMER = re.compile(
    r"(?im)^(справочная информация[,.:]?|рекомендации требуют сопоставления[^\n]*)\s*$\n?",
    re.MULTILINE,
)

# Служебные строки ReAct/тегов, оставшиеся от формата датасета
_REACT_LINE = re.compile(
    r"(?im)^\s*(?:"
    r"(?:thought|action|observation|final\s*answer)\s*:.*$"
    r"|search_icd11\s*\[[^\]]*\]\.?\s*$"
    r"|search_pubmed\s*\[[^\]]*\]\.?\s*$"
    r"|drug_info\s*\[[^\]]*\]\.?\s*$"
    r"|search_guidelines\s*\[[^\]]*\]\.?\s*$"
    r"|calculate\s*\[[^\]]*\]\.?\s*$"
    r")\s*$\n?",
    re.MULTILINE,
)


def _de_slug(match: re.Match) -> str:
    """биполярное_I_расстройство -> биполярное I расстройство"""
    return match.group(0).replace("_", " ")


def clean_reply(text: str) -> str:
    """Убираем XML-разметку и служебные метки, оставляя чистый финальный ответ."""
    if not text:
        return ""
    # 1. think-блоки (даже незакрытые) — не показываем
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)
    text = _ORPHAN_CLOSE_THINK.sub("", text)

    # 2. служебные строки ReAct-формата (Thought:/Action:/Observation:/Final Answer:,
    #    вызовы инструментов) — срезаем целиком
    text = _REACT_LINE.sub("", text)

    # 3. известные секционные теги -> markdown-заголовки
    def _section(m: re.Match) -> str:
        name = m.group(1).lower()
        return f"\n\n{_KNOWN_SECTIONS.get(name, '')}\n" if name in _KNOWN_SECTIONS else ""

    text = re.sub(r"<\s*([a-zA-Z][a-zA-Z0-9_-]*)\s*(?:\s[^<>]*)?>", _section, text)
    # все оставшиеся теги (закрывающие, неизвестные, с атрибутами) — срезаем
    text = _ANY_TAG.sub("", text)

    # 4. slug-метки датасета -> обычные слова
    text = _SLUG.sub(_de_slug, text)

    # 5. повторяющийся дисклеймер — один раз в самом конце
    disclaimer = "Решение принимает лечащий врач с учётом осмотра и действующих клинических рекомендаций."
    text = _DISCLAIMER.sub("", text)
    text = text.rstrip()
    if not text.endswith(disclaimer):
        text += "\n\n" + disclaimer

    # 6. причёсываем
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class ContextTooLong(Exception):
    """Запрос не помещается в контекст endpoint'а даже после всех урезаний."""


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " …[обрезано]"


def _strip_think(text: str) -> str:
    """Срезаем think-блоки ДО отправки в историю: иначе длинные вопросы на Qwen3
    разрастаются в контексте, а остатки <think> ломают следующие раунды tool-calling."""
    text = _THINK_BLOCK.sub("", text)
    text = _UNCLOSED_THINK.sub("", text)
    return _ORPHAN_CLOSE_THINK.sub("", text)


def _is_context_error(err: APIStatusError) -> bool:
    msg = str(err).lower()
    return any(k in msg for k in ("token", "context", "too long", "max_total", "maximum"))


def _fit_max_tokens(err_text: str, current: int):
    """Достаём из ошибки TGI/vLLM реальный лимит и считаем, сколько токенов осталось на ответ."""
    # TGI: "must be <= 4096. Given: 3500 inputs tokens and 1200 max_new_tokens"
    m = re.search(r"<=\s*(\d+)\.\s*Given:\s*(\d+)\s*inputs?\s*tokens", err_text)
    if m:
        return int(m.group(1)) - int(m.group(2)) - 16
    # vLLM: "maximum context length is 8192 tokens ... (7000 in the messages, ..."
    m = re.search(r"maximum context length is (\d+).*?\((\d+) in the messages", err_text, re.S)
    if m:
        return int(m.group(1)) - int(m.group(2)) - 16
    return current // 2


def _shrink_tool_results(messages: list) -> bool:
    """Вдвое урезаем самые длинные результаты инструментов. True, если что-то урезали."""
    changed = False
    for m in messages:
        if m["role"] == "tool" and len(m["content"]) > 800:
            m["content"] = _clip(m["content"], len(m["content"]) // 2)
            changed = True
    return changed


def _call_model(messages: list, use_tools: bool, max_tokens: int):
    """Один вызов модели В РЕЖИМЕ СТРИМА: соединение не простаивает, поэтому прокси
    HF Endpoints не рвёт длинные генерации по таймауту. Собираем ответ по кускам."""
    kwargs = dict(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=max_tokens,
        stream=True,
    )
    if use_tools:
        kwargs.update(tools=TOOLS, tool_choice="auto")

    parts: list[str] = []
    calls: dict[int, dict] = {}
    finish = None
    for chunk in client.chat.completions.create(**kwargs):
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        delta = choice.delta
        if delta.content:
            parts.append(delta.content)
        for tc in delta.tool_calls or []:
            slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function:
                if tc.function.name and not slot["name"]:
                    slot["name"] = tc.function.name
                if tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        if choice.finish_reason:
            finish = choice.finish_reason

    tool_calls = []
    for i in sorted(calls):
        c = calls[i]
        c["id"] = c["id"] or f"call_{i}"
        tool_calls.append(c)
    return "".join(parts), tool_calls, finish


def _chat(messages: list, use_tools: bool = True):
    """Вызов модели с авто-восстановлением: холодный старт, перегрузка, переполнение контекста."""
    max_tokens = MAX_NEW_TOKENS
    for attempt in range(8):
        try:
            return _call_model(messages, use_tools, max_tokens)
        except APIStatusError as e:
            if e.status_code in (429, 503):  # endpoint просыпается / перегружен
                time.sleep(min(5 * (attempt + 1), 30))
                continue
            if e.status_code in (400, 413, 422) and _is_context_error(e):
                fit = _fit_max_tokens(str(e), max_tokens)
                if fit is not None and fit >= 256:
                    max_tokens = min(max_tokens, fit)       # хватает места хотя бы на ответ
                    continue
                if _shrink_tool_results(messages):          # режем накопленные результаты инструментов
                    max_tokens = max(512, max_tokens // 2)
                    continue
                raise ContextTooLong(str(e))
            raise
    raise ContextTooLong("endpoint недоступен или не отвечает после нескольких попыток")


def run_agent(user_message: str) -> ChatResponse:
    user_message = _clip(user_message.strip(), MAX_INPUT_CHARS)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    tool_log: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        content, tool_calls, finish = _chat(messages, use_tools=True)

        # Модель не хочет инструменты — это финальный ответ.
        # НО: если текст пуст после срезки <think>, а инструменты доступны — это
        # затык thinking-режима, а не ответ. Даём один ретрай без инструментов.
        if not tool_calls:
            reply = clean_reply(content)
            if not reply:
                content, _, finish = _chat(messages, use_tools=False)
                reply = clean_reply(content) or "Модель не смогла сформировать ответ, переформулируйте вопрос."
            if finish == "length":
                reply = reply.rstrip() + "\n\n[Ответ обрезан по лимиту токенов — задайте вопрос уже или увеличьте MAX_NEW_TOKENS.]"
            return ChatResponse(reply=reply, tool_calls_made=tool_log)

        # Прокидываем ход ассистента с tool_calls обратно в историю.
        # content чистим от <think> и вырезаем до 2000 симв: Qwen3 в thinking-режиме
        # кладёт туда простыню размышлений, которая раздувает контекст следующих раундов.
        messages.append(
            {
                "role": "assistant",
                "content": _clip(_strip_think(content or "").strip(), 2000),
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": tc["arguments"]},
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Исполняем каждый вызов, отдаём результаты как role="tool" (с ограничением размера)
        for tc in tool_calls:
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(tc["name"], args)
            tool_log.append({"tool": tc["name"], "arguments": args, "result": result})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": _clip(json.dumps(result, ensure_ascii=False), MAX_TOOL_RESULT_CHARS),
                }
            )

    # Исчерпали лимит циклов — просим модель ответить без инструментов
    content, _, finish = _chat(messages, use_tools=False)
    reply = clean_reply(content)
    if finish == "length":
        reply = reply.rstrip() + "\n\n[Ответ обрезан по лимиту токенов — задайте вопрос уже или увеличьте MAX_NEW_TOKENS.]"
    return ChatResponse(reply=reply, tool_calls_made=tool_log)


# ------------------------- HTTP -------------------------

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/chat")
def chat(req: ChatRequest) -> ChatResponse:
    try:
        return run_agent(req.message)
    except ContextTooLong as e:
        raise HTTPException(
            status_code=413,
            detail="Запрос не помещается в контекст модели. Увеличьте Max Input Length / "
                   f"Max Number of Tokens в настройках endpoint или сократите запрос. ({e})",
        )
    except APITimeoutError:
        raise HTTPException(status_code=504, detail="Endpoint не ответил вовремя (таймаут).")
    except APIConnectionError as e:
        raise HTTPException(status_code=502, detail=f"Нет соединения с endpoint: {e}")
    except APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Endpoint вернул {e.status_code}: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
