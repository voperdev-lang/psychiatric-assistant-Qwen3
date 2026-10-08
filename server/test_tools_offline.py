# Тест диспетчера инструментов без LLM (endpoint фейковый).
import json
import os
import sys

os.environ["HF_ENDPOINT_URL"] = "https://fake.example/v1"
os.environ["HF_TOKEN"] = "test_dummy"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import server  # noqa: E402

tests = [
    ("drug_info", {"drug": "lithium"}),
    ("drug_info", {"drug": "xanax"}),  # нет в базе — должен честный not_found
    ("search_guidelines", {"query": "acute mania treatment lithium valproate"}),
    ("search_icd11", {"query": "hypomanic episode diagnostic requirements"}),
    ("calculate", {"expression": "0.6 * 70"}),
]
for name, args in tests:
    res = server.execute_tool(name, args)
    s = json.dumps(res, ensure_ascii=False)
    print(f"[{name}] {args}")
    print(f"   -> {s[:220]}")
    print()
