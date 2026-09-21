"""
homework01 · агент в терминале.

Команды:
  python agent.py test                 # тесты инструментов (нужен интернет)
  python agent.py ask "вопрос" [...]   # один прогон агента + трейс
  python agent.py measure              # Шаг 3: полный замер -> csv/md/png

Пути ищутся автоматически по папке data/fresh_2026.jsonl
"""

import os, re, sys, json, time, ast, operator, subprocess, argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
from num2words import num2words

import requests
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # рисуем в файл, без окна GUI (важно для терминала)
import matplotlib.pyplot as plt

from pydantic import BaseModel, Field

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*a, **k):
        return False


# ======================================================================
# Пути: ищем корень проекта по наличию data/*.jsonl
# ======================================================================
def find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        d = p / "data"
        if d.is_dir() and any(d.glob("*.jsonl")):
            return p
    return start


HERE = Path(__file__).resolve().parent
ROOT = find_root(HERE)

DATA_DIR   = ROOT / "data"
TRACES     = ROOT / "traces"
IMG        = ROOT / "img"
TRACES.mkdir(parents=True, exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)


def resolve_dataset() -> Path:
    # Список возможных имен файла датасета
    candidates = ["fresh_2026.jsonl"] 
    
    for name in candidates:
        cand = DATA_DIR / name
        
        # ВАЖНО: is_file() проверяет, что это именно ФАЙЛ, а не папка
        if cand.is_file(): 
            return cand
            
    raise FileNotFoundError(
        f"Не найден валидный файл датасета в {DATA_DIR}. "
        f"Ожидались файлы: {candidates}. Проверьте, нет ли там папок с такими именами."
    )


def load_env():
    # .env может лежать рядом со скриптом, в корне или в cwd — берём первый
    for cand in (HERE / ".env", ROOT / ".env", Path.cwd() / ".env"):
        if cand.exists():
            load_dotenv(cand)
            return
    load_dotenv()

load_env()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables (.env)")

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "https://github.com/sv-shnekutis/homework01",
    "X-Title": "agents-course-homework01",
}

MODELS = {
    "cheap":  "openai/gpt-4o-mini",
    "mid":    "anthropic/claude-haiku-4.5",
    "strong": "anthropic/claude-sonnet-4.6",
}
COLORS = {
    "violet": "#5436A3", "amber": "#F09000", "teal": "#00838F",
    "red": "#C43C3C", "grey": "#787882",
}


@dataclass
class Ledger:
    calls: list = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(c["cost"] for c in self.calls)


ledger = Ledger()


def chat(messages, model, tools=None, tag="chat", temperature=0.0):
    payload = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        payload["tools"] = tools

    started = time.perf_counter()
    for attempt in range(3):
        try:
            r = requests.post(CHAT_URL, headers=HEADERS, json=payload, timeout=90)
            if r.status_code == 200:
                data = r.json()
                usage = data.get("usage", {}) or {}
                cost = float(usage.get("cost", 0) or 0)
                ledger.calls.append({
                    "tag": tag,
                    "model": model.split("/")[-1],
                    "prompt": int(usage.get("prompt_tokens", 0) or 0),
                    "completion": int(usage.get("completion_tokens", 0) or 0),
                    "cost": cost,
                    "seconds": round(time.perf_counter() - started, 2),
                })
                return data["choices"][0]["message"]
            # 429 / 5xx — повторить
            time.sleep(1.0 + attempt)
        except Exception as e:
            print(f"Chat error: {e}")
            time.sleep(1.0 + attempt)
    raise RuntimeError("OpenRouter недоступен после 3 попыток")


# ======================================================================
# Википедия
# ======================================================================
WIKI = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "agents-course-homework01/1.0 (https://postypashki.ru; educational project)"
}


def wiki(params, attempts=3):
    """Универсальная обертка для запросов к Википедии."""
    for attempt in range(attempts):
        try:
            r = requests.get(
                WIKI, params={**params, "format": "json"},
                headers=WIKI_HEADERS, timeout=15,
            )
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                return r.json()["query"]
            time.sleep(1.0 + attempt)
        except Exception as e:
            print(f"Wiki error: {e}")
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Википедия недоступна после {attempts} попыток")


# Инструменты
class SearchArgs(BaseModel):
    query: str = Field(description="короткий поисковый запрос: имя, название, термин")


class PageFindArgs(BaseModel):
    title: str = Field(description="точный заголовок статьи Википедии")
    keywords: str = Field(description="ключевые слова через запятую для поиска по тексту статьи")


class CalcArgs(BaseModel):
    expression: str = Field(description="арифметическое выражение: числа, скобки, + - * / ** %")


class ExecArgs(BaseModel):
    code: str = Field(description="код на Python; результат надо напечатать через print")


def web_search(query: str) -> str:
    res = wiki({
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
    })
    hits = res.get("search", [])
    if not hits:
        return "No data"
    title = hits[0]["title"]
    page = wiki({
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "titles": title,
        "redirects": 1,
    })
    pages = page.get("pages", {})
    if not pages:
        return f"[{title}] Page not found."
    extract = list(pages.values())[0].get("extract", "")
    return f"[{title}] {extract[:5000]}"


def page_find(title: str, keywords: str) -> str:
    """Читает полную статью и возвращает строки с ключевыми словами."""
    res = wiki({
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "titles": title,
        "redirects": 1,
    })
    pages = res.get("pages", {})
    if not pages:
        return "Page not found."

    page_id = list(pages.keys())[0]
    page = pages[page_id]
    if page.get("missing") is not None or str(page_id).startswith("-"):
        return "Page not found."

    text = page.get("extract", "")
    if not text:
        return "Empty content."

    kw_list = [k.lower().strip() for k in keywords.split(",") if k.strip()]
    relevant = []
    for line in text.split("\n"):
        line_lower = line.lower()
        if any(kw in line_lower for kw in kw_list):
            relevant.append(line.strip())

    if not relevant:
        sections = [l.strip() for l in text.split("\n") if l.strip().startswith("==")]
        return (f"No exact matches for '{keywords}'.\n"
                f"Sections: {sections[:20]}\n"
                f"Start: {text[:500]}...")
    return "\n".join(relevant[:10])[:3000]

OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("недопустимое выражение")


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree))
    except Exception as e:
        return f"ошибка вычисления: {e}"


def python_exec(code: str) -> str:
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=5,
        )
    except subprocess.TimeoutExpired:
        return "Код превысил 5 сек"
    out = (r.stdout + r.stderr).strip()
    return out[:5000] if out else "Код исполнился но результат пустой"


TOOLS: Dict[str, dict] = {}


def register(fn, args_model, description):
    TOOLS[fn.__name__] = {
        "fn": fn,
        "args": args_model,
        "schema": {
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": description,
                "parameters": args_model.model_json_schema(),
            },
        },
    }


register(web_search, SearchArgs,
         "Searches Wikipedia for article titles matching the query.")
register(page_find, PageFindArgs,
         "Reads a specific Wikipedia article and extracts paragraphs containing given keywords. Use AFTER web_search.")
register(calculator, CalcArgs,
         "Считает арифметическое выражение: числа, скобки, + - * / ** %")
register(python_exec, ExecArgs,
         "Выполняет код на Python в отдельном процессе и возвращает то, что он напечатал")


def run_tool(name, arguments):
    spec = TOOLS.get(name)
    if spec is None:
        return f"Неизвестный инструмент: {name}"
    try:
        args = spec["args"](**arguments)
        return spec["fn"](**args.model_dump())
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}"


# Агент: цикл + мягкое завершение + детектор повторов
SYSTEM = (
    "Ты решаешь задачи. Если нужно посчитать или найти факт, вызывай инструменты, "
    "а не угадывай. Когда ответ готов, напиши его последней строкой в формате "
    "FINAL: <ответ>. Для числовых задач в FINAL только число, для вопросов о фактах "
    "короткая фраза.\n"
    "Если вопрос о событии 2026 года в конкретной стране или теме, сначала попробуй "
    "page_find с title=\"2026 in <Страна>\" или \"2026 in <тема>\", а не общий web_search."
)

@dataclass
class Run:
    question: str
    answer: str
    steps: int
    messages: list = field(default_factory=list)
    cost: float = 0.0
    seconds: float = 0.0


def agent_loop(messages, model, tool_names, max_steps):
    seen = set()
    schemas = [TOOLS[n]["schema"] for n in tool_names if n in TOOLS] or None

    for step in range(1, max_steps + 1):
        msg = chat(messages, model, tools=schemas, tag="agent")
        messages.append(msg)

        calls = msg.get("tool_calls") or []
        if not calls:
            break  # модель дала финальный ответ

        responses_this_step = []
        for call in calls:
            fn_name = call["function"]["name"]
            try:
                args = json.loads(call["function"]["arguments"])
            except json.JSONDecodeError:
                args = {}
            result = run_tool(fn_name, args)
            responses_this_step.append(result)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": str(result)[:2000],
            })

        # Детектор зацикливания -> мягкое завершение
        sig = tuple(responses_this_step)
        if sig in seen:
            messages.append({
                "role": "user",
                "content": "You repeated the same action. Please provide your final answer now starting with FINAL: ",
            })
            msg = chat(messages, model, tag="soft_stop")
            messages.append(msg)
            break
        seen.add(sig)

    return messages[-1].get("content", ""), step


def agent(question, model, tool_names, max_steps=8):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    before, started = ledger.total, time.perf_counter()
    answer, steps = agent_loop(messages, model, tool_names, max_steps)
    return Run(
        question=question,
        answer=answer,
        steps=steps,
        messages=messages,
        cost=ledger.total - before,
        seconds=time.perf_counter() - started,
    )


def show_trace(run: Run):
    for m in run.messages:
        calls = "; ".join(
            f"{c['function']['name']}{c['function']['arguments']}"
            for c in (m.get("tool_calls") or [])
        )
        text = " ".join((m.get("content") or "").split())[:180]
        print(f"{m['role']:9s}| {text} {calls}")
    print(f"шагов: {run.steps}, цена: {run.cost * 100:.3f} ¢, время: {run.seconds:.1f} c")

_WORD2DIGIT = {num2words(n): str(n) for n in range(101)}
_STOP = {"a", "an", "the"}

def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)          # 1,000 -> 1000
    text = re.sub(r"[^\w\s]", " ", text)                 # пунктуация
    # six -> 6, twenty one -> 21
    toks, out, i = text.split(), [], 0
    while i < len(toks):
        for j in range(min(3, len(toks) - i), 0, -1):
            chunk = " ".join(toks[i:i + j])
            if chunk in _WORD2DIGIT:
                out.append(_WORD2DIGIT[chunk])
                i += j
                break
        else:
            out.append(toks[i])
            i += 1
    return " ".join(w for w in out if w not in _STOP)


def final_answer(text: str) -> str:
    m = re.search(r"FINAL:\s*(.+)", text or "")
    return m.group(1).strip() if m else (text or "").strip()


def is_correct(task: Dict, answer: str) -> bool:
    gold_raw = final_answer(str(task["answer"]).strip())
    pred_raw = final_answer(str(answer).strip())

    # Числовой эталон (включая "36°C"): сравниваем множества чисел
    if re.fullmatch(r"-?\d+(?:\.\d+)?\s*°?\s*[CF]?", gold_raw.strip(), re.I):
        def nums(t):
            t = re.sub(r"(?<=\d),(?=\d)", "", t)
            return {round(float(x), 4) for x in re.findall(r"-?\d+(?:\.\d+)?", t)}
        g, p = nums(gold_raw), nums(pred_raw)
        return bool(g) and g.issubset(p)

    # Текстовое сравнение: subset токенов вместо подстроки
    g = set(normalize(gold_raw).split())
    p = set(normalize(pred_raw).split())
    return bool(g) and g.issubset(p)


# ======================================================================
# Датасет и прогон
# ======================================================================
def load_tasks(path) -> List[Dict]:
    path = Path(path)
    tasks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


CONFIGS = {
    "без инструментов": [],
    "поиск": ["web_search", "calculator", "python_exec"],
    "поиск+чтение": ["web_search", "page_find", "calculator", "python_exec"],
}

MEASURE_CONFIGS = [
    ("дешёвая, поиск и чтение страницы",      "cheap",  ["web_search", "page_find", "calculator", "python_exec"]),
]


def run_tasks(tasks, model, tool_names, config) -> pd.DataFrame:
    folder = TRACES / config / model.split("/")[-1]
    folder.mkdir(parents=True, exist_ok=True)
    rows = []

    for i, task in enumerate(tasks):
        try:
            run = agent(task["question"], model, tool_names)
        except Exception as e:
            run = Run(task["question"], f"ошибка: {e}", 0, [])

        ok = is_correct(task, final_answer(run.answer))
        rows.append({
            "config": config,
            "model": model.split("/")[-1],
            "id": task["id"],
            "source": task.get("source", ""),
            "correct": ok,
            "steps": run.steps,
            "tool_calls": sum(m["role"] == "tool" for m in run.messages),
            "cost": run.cost,
            "seconds": round(run.seconds, 1),
            "answer": final_answer(run.answer)[:60],
            "gold": task["answer"],
        })

        safe_id = str(task["id"]).replace("/", "_").replace("\\", "_")
        with open(folder / f"{safe_id}.json", "w", encoding="utf-8") as tf:
            json.dump(
                {"task": task, "prediction": run.answer, "correct": ok, "trace": run.messages},
                tf, ensure_ascii=False, indent=2,
            )

        if (i + 1) % 10 == 0:
            acc = sum(r["correct"] for r in rows) / len(rows)
            print(f"  [{config}] {i + 1}/{len(tasks)}, acc={acc:.2f}")

    return pd.DataFrame(rows)


def report(results: pd.DataFrame) -> pd.DataFrame:
    g = results.groupby(["config", "model"], sort=False)
    out = g.agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        cost_per_task=("cost", "mean"),
        avg_steps=("steps", "mean"),
        avg_seconds=("seconds", "mean"),
        total_cost=("cost", "sum"),
        n_correct=("correct", "sum"),
    ).reset_index()
    out["cost_per_correct"] = out.apply(
        lambda r: r["total_cost"] / r["n_correct"] if r["n_correct"] else float("inf"), axis=1
    )
    return out.drop(columns=["total_cost", "n_correct"])


def money_chart(table: pd.DataFrame, path: str):
    labels = [f"{r.config}\n{r.model}" for r in table.itertuples()]
    x = range(len(table))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].bar(x, table["accuracy"], color=COLORS["teal"])
    axes[0].set_ylabel("доля верных")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3, axis="y")

    axes[1].bar(x, table["cost_per_correct"], color=COLORS["amber"])
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    axes[1].set_ylabel("$ за верный ответ")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def cmd_test(_args):
    print("Производим тест инструментов...")

    # calculator: норма / пустой / ошибка
    assert calculator("2+2") == "4"
    assert calculator("2**10") == "1024"
    assert "ошибка" in calculator("")
    assert "ошибка" in calculator("__import__('os')")

    # web_search: норма / пустой / ошибка
    assert web_search("Scott Derrickson").startswith("[Scott Derrickson]")
    assert web_search("qwxzv 1234567 nonsense") == "No data"
    assert "Tool error" in run_tool("web_search", {})

    # page_find: норма / пустой / ошибка
    assert len(page_find("Earth", "planet")) > 50
    assert page_find("Earth", "zzzz_nonexistent_keyword").startswith("No exact matches")
    assert page_find("QzxvNonexistentTestArticle123456789", "anything") == "Page not found."

    # python_exec: норма / пустой / ошибка / таймаут
    assert python_exec("print(sum(range(10)))") == "45"
    assert python_exec("x = 1") == "Код исполнился но результат пустой"
    assert "ZeroDivisionError" in python_exec("1/0")
    assert "Код превысил 5 сек" in python_exec("import time; time.sleep(10)")
    print("инструменты работают")


def cmd_ask(args):
    tool_names = [t.strip() for t in args.tools.split(",") if t.strip()] if args.tools else []
    model = MODELS[args.model]
    print(f"model={args.model} ({model})  tools={tool_names}\n")
    run = agent(args.question, model, tool_names, max_steps=args.max_steps)
    show_trace(run)
    print("\nFINAL:", final_answer(run.answer))


def cmd_measure(args):
    dataset = resolve_dataset()
    tasks = load_tasks(dataset)
    print(f"Датасет: {dataset}  задач всего: {len(tasks)}")

    if args.limit:
        tasks = tasks[: args.limit]
        print(f"SMOKE-режим: берём первые {len(tasks)} задач\n")

    configs = MEASURE_CONFIGS
    if args.only:
        configs = [c for c in MEASURE_CONFIGS if c[0] == args.only]
        assert configs, f"Нет такой конфигурации: {args.only}. Доступны: {[c[0] for c in MEASURE_CONFIGS]}"
        print(f"Фильтр по конфигурации: {args.only}\n")

    start_cost = ledger.total
    frames = []
    for config_name, model_key, tool_names in configs:
        print(f"=== {config_name} | {model_key} ===")
        frames.append(run_tasks(tasks, MODELS[model_key], tool_names, config_name))

    results = pd.concat(frames, ignore_index=True)
    spent = ledger.total - start_cost
    print(f"\nПотрачено: {spent:.4f} $")

    table = report(results)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)
    print("\nТаблица:")
    print(table.to_string(index=False))

    # В smoke-режиме показываем построчно, чтобы видеть провалы
    if args.limit:
        print("\nПострочно:")
        print(results.to_string(index=False))
        results.to_csv(ROOT / "results_raw.csv", index=False)
        (ROOT / "results.md").write_text(table.to_string(index=False), encoding="utf-8")
        money_chart(table, str(IMG / "homework.png"))
        print(f"\nСохранено: results_raw.csv, results.md, img/homework.png")
        wrong = results[~results["correct"]]
        if not wrong.empty:
            print(f"\nПервый провал: id={wrong.iloc[0]['id']}")
            print(f"  gold:   {wrong.iloc[0]['gold']}")
            print(f"  answer: {wrong.iloc[0]['answer']}")
            print(f"  трейс:  {TRACES / wrong.iloc[0]['config'] / wrong.iloc[0]['model'] / (str(wrong.iloc[0]['id']).replace('/', '_') + '.json')}")

    # Полные артефакты пишем только при полном прогоне, чтобы не затирать results.md
    if not args.limit and not args.only:
        results.to_csv(ROOT / "results_raw.csv", index=False)
        (ROOT / "results.md").write_text(table.to_string(index=False), encoding="utf-8")
        money_chart(table, str(IMG / "homework.png"))
        print(f"\nСохранено: results_raw.csv, results.md, img/homework.png")

def main():
    parser = argparse.ArgumentParser(description="homework01 агент в терминале")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_test = sub.add_parser("test", help="тесты инструментов (нужен интернет)")
    p_test.set_defaults(func=cmd_test)

    p_ask = sub.add_parser("ask", help="один прогон агента")
    p_ask.add_argument("question")
    p_ask.add_argument("--model", default="cheap", choices=list(MODELS))
    p_ask.add_argument("--tools", default="web_search,page_find,calculator,python_exec")
    p_ask.add_argument("--max-steps", type=int, default=8)
    p_ask.set_defaults(func=cmd_ask)

    p_meas = sub.add_parser("measure", help="Шаг 3: замер -> csv/md/png")
    p_meas.add_argument("--limit", type=int, default=0, help="взять первые N задач (smoke)")
    p_meas.add_argument("--only", type=str, default="", help="прогнать только одну конфигурацию")
    p_meas.set_defaults(func=cmd_measure)

    args = parser.parse_args()
    if not hasattr(args, "func"):      # чтобы любая забытая set_defaults давала помощь, а не падение
        parser.print_help()
        raise SystemExit(1)
    args.func(args)



if __name__ == "__main__":
    main()