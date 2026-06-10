"""Quick sanity check: API key valid, correct model responds."""
import os, sys
sys.path.insert(0, ".")

key = os.environ.get("ANTHROPIC_API_KEY", "")
model = os.environ.get("REGIONALS_LLM_MODEL", "claude-opus-4-8")

print(f"ANTHROPIC_API_KEY : {'SET (' + key[:12] + '...)' if key else 'NOT SET'}")
print(f"REGIONALS_LLM_MODEL: {model}")
print()

try:
    import anthropic
except ImportError:
    print("ERROR: пакет anthropic не установлен — pip install anthropic")
    sys.exit(1)

if not key:
    print("ERROR: ANTHROPIC_API_KEY не задан")
    sys.exit(1)

client = anthropic.Anthropic()
print(f"Отправляем тестовый запрос к {model}...")
try:
    resp = client.messages.create(
        model=model,
        max_tokens=20,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), "").strip()
    tokens_in  = resp.usage.input_tokens
    tokens_out = resp.usage.output_tokens
    print(f"Ответ    : {text!r}")
    print(f"Модель   : {resp.model}")
    print(f"Токены   : in={tokens_in} out={tokens_out}")
    print()
    print("OK — ключ работает, модель отвечает.")
except anthropic.AuthenticationError:
    print("ERROR: ключ недействителен (AuthenticationError)")
    sys.exit(1)
except anthropic.NotFoundError:
    print(f"ERROR: модель {model!r} не найдена (NotFoundError) — проверь REGIONALS_LLM_MODEL")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    sys.exit(1)
