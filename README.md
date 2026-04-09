# martin-outfitters-app

AI-powered customer assistant for Martin Outfitters, built with the Anthropic Python SDK.

## Usage optimizations

| Technique | Where | Savings |
|---|---|---|
| **Prompt caching** | System prompt + product catalog cached with `cache_control: ephemeral` | ~90% on repeated input tokens |
| **Model routing** | `claude-haiku-4-5` for intent classification, `claude-sonnet-4-6` for full chat | Haiku is 5× cheaper than Sonnet |
| **Batch API** | Bulk product-description generation via `messages.batches` | 50% off all token costs |
| **Token counting** | Pre-flight `count_tokens` call before large requests | Avoids surprise overages |
| **Right-sized max_tokens** | 16 for labels, 256 for short descriptions, 1024 for chat | Caps wasted output budget |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
python app.py
```

## Key files

- `app.py` — main application with chat loop, intent classifier, and batch helpers