"""
Martin Outfitters — AI Assistant
Usage-optimized Claude integration:
  - Prompt caching on the shared system prompt + product catalog
  - claude-haiku-4-5 for cheap intent classification
  - claude-sonnet-4-6 for full chat responses (cached prefix)
  - Batch API for bulk product-description generation (50% off)
"""

import json
import time
import anthropic

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Shared context (cached — this is the expensive part that we pay once)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful sales assistant for Martin Outfitters, a
premium outdoor gear and apparel store. You help customers find the right
equipment for hiking, camping, climbing, fishing, and hunting.

Be friendly, knowledgeable, and concise. When recommending products, mention
key features and why they suit the customer's needs. If you're unsure about
current inventory or pricing, say so and offer to connect them with a store
associate.

Store policies:
- Free shipping on orders over $75
- 30-day hassle-free returns on unworn/unused gear
- Price-match guarantee against major outdoor retailers
- Loyalty points: 1 point per $1 spent, redeemable for store credit
"""

PRODUCT_CATALOG = """
CURRENT PRODUCT CATALOG:

Footwear:
- Trail Runner X3 ($129) — lightweight, waterproof, Vibram sole, sizes 6-14
- Summit Boot Pro ($249) — full-grain leather, crampon-compatible, insulated
- Camp Moccasin ($59) — packable camp shoe, EVA sole

Apparel:
- MernoTech Base Layer ($89) — 200g merino/nylon blend, odor-resistant
- StormShield Jacket ($199) — Gore-Tex, packable, 3-layer waterproof/breathable
- FlexHike Pants ($79) — 4-way stretch, zip-off legs, 6 pockets

Camping:
- UltraLight 2P Tent ($349) — 2.1 lb, freestanding, 3-season, footprint included
- 20°F Sleeping Bag ($179) — 650-fill duck down, water-resistant shell
- Titanium Cook Set ($49) — 750ml pot + lid + cup, compatible with canister stoves

Climbing:
- Harness Elite ($89) — UIAA-certified, gear loops x4, ice clipper slots
- Belay Device ATC-XP ($22) — assisted-braking mode, compatible 8.7-11mm ropes

Fishing:
- 9' 5wt Fly Rod Combo ($219) — 4-piece graphite, reel + WF5F line included
- Chest Waders ($149) — 5mm neoprene, gravel guards, boot size 7-13

Hunting:
- Camo Soft Shell ($159) — scent-control fabric, fleece-lined, quiet stretch panels
- Ground Blind Pop-Up ($129) — 180° view, 2-person, hub design, carry bag
"""


def make_cached_params(messages: list[dict]) -> dict:
    """Build message params with prompt caching on the shared prefix."""
    return {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        # Cache the system prompt + product catalog together.
        # Any byte change here would bust the cache, so we keep it stable.
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT + "\n\n" + PRODUCT_CATALOG,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Intent classifier (haiku — cheap, fast)
# ---------------------------------------------------------------------------

INTENTS = ["product_question", "order_help", "return_request", "general_chat", "other"]


def classify_intent(user_message: str) -> str:
    """Use Haiku to cheaply route the message before the full response."""
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=16,  # just the label
        messages=[
            {
                "role": "user",
                "content": (
                    f"Classify this customer message into exactly one of: "
                    f"{', '.join(INTENTS)}.\n"
                    f"Reply with only the label.\n\n"
                    f"Message: {user_message}"
                ),
            }
        ],
    )
    label = response.content[0].text.strip().lower()
    return label if label in INTENTS else "other"


# ---------------------------------------------------------------------------
# Chat assistant (Sonnet + cached prefix)
# ---------------------------------------------------------------------------

def chat(conversation: list[dict], user_message: str) -> tuple[str, dict]:
    """
    Send a user message, get a response.
    Returns (assistant_text, usage_stats).
    """
    conversation.append({"role": "user", "content": user_message})

    params = make_cached_params(conversation)
    response = client.messages.create(**params)

    assistant_text = response.content[0].text
    conversation.append({"role": "assistant", "content": assistant_text})

    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read_tokens": getattr(response.usage, "cache_read_input_tokens", 0),
        "cache_write_tokens": getattr(response.usage, "cache_creation_input_tokens", 0),
    }
    return assistant_text, usage


# ---------------------------------------------------------------------------
# Batch API — bulk product description generation (50% off)
# ---------------------------------------------------------------------------

def generate_product_descriptions_batch(products: list[dict]) -> str:
    """
    Submit a batch job to generate SEO product descriptions.
    Returns the batch ID to poll later.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = [
        Request(
            custom_id=f"product-{p['sku']}",
            params=MessageCreateParamsNonStreaming(
                model="claude-haiku-4-5",
                max_tokens=256,
                system=(
                    "You write short, compelling SEO product descriptions "
                    "for an outdoor gear store. 2-3 sentences max."
                ),
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Write a product description for: {p['name']}\n"
                            f"Price: {p['price']}\n"
                            f"Key features: {p['features']}"
                        ),
                    }
                ],
            ),
        )
        for p in products
    ]

    batch = client.messages.batches.create(requests=requests)
    print(f"Batch submitted: {batch.id}  (status: {batch.processing_status})")
    return batch.id


def poll_batch(batch_id: str, poll_interval: int = 10) -> dict[str, str]:
    """Poll until the batch is done, return {custom_id: text}."""
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        print(f"  Batch {batch_id}: {batch.processing_status} — waiting {poll_interval}s …")
        time.sleep(poll_interval)

    results = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            text = next(
                (b.text for b in result.result.message.content if b.type == "text"),
                "",
            )
            results[result.custom_id] = text
        else:
            results[result.custom_id] = f"[ERROR: {result.result.type}]"
    return results


# ---------------------------------------------------------------------------
# Token counter — check cost before a big request
# ---------------------------------------------------------------------------

def estimate_tokens(messages: list[dict]) -> int:
    """Count tokens for the cached params without sending the actual request."""
    params = make_cached_params(messages)
    count = client.messages.count_tokens(
        model=params["model"],
        system=params["system"],
        messages=params["messages"],
    )
    return count.input_tokens


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------

def demo_chat():
    print("=" * 60)
    print("Martin Outfitters — AI Assistant Demo")
    print("=" * 60)

    conversation: list[dict] = []

    turns = [
        "Hi! I'm planning a 3-day backpacking trip and need a tent recommendation.",
        "What sleeping bag would pair well with that?",
        "Do you price-match? I saw the tent cheaper on another site.",
    ]

    for msg in turns:
        intent = classify_intent(msg)
        print(f"\n[Customer] {msg}")
        print(f"  (intent: {intent})")

        reply, usage = chat(conversation, msg)
        print(f"[Assistant] {reply}")

        cached = usage["cache_read_tokens"]
        fresh = usage["input_tokens"]
        print(
            f"  tokens → input:{fresh} output:{usage['output_tokens']} "
            f"cache_read:{cached} cache_write:{usage['cache_write_tokens']}"
        )
        if cached > 0:
            savings_pct = round(cached / (fresh + cached) * 100)
            print(f"  💰 {savings_pct}% of input tokens served from cache")


def demo_batch():
    print("\n" + "=" * 60)
    print("Batch API — Bulk Product Description Generation")
    print("=" * 60)

    products = [
        {
            "sku": "TRX3",
            "name": "Trail Runner X3",
            "price": "$129",
            "features": "lightweight, waterproof, Vibram sole",
        },
        {
            "sku": "SSB",
            "name": "Summit Boot Pro",
            "price": "$249",
            "features": "full-grain leather, crampon-compatible, insulated",
        },
        {
            "sku": "UL2P",
            "name": "UltraLight 2P Tent",
            "price": "$349",
            "features": "2.1 lb, freestanding, 3-season, footprint included",
        },
    ]

    print(f"Submitting {len(products)} descriptions as a batch (50% cheaper than sync)…")
    batch_id = generate_product_descriptions_batch(products)

    print("Polling for results…")
    descriptions = poll_batch(batch_id, poll_interval=5)

    for sku, text in descriptions.items():
        print(f"\n  {sku}: {text}")


def demo_token_estimate():
    print("\n" + "=" * 60)
    print("Token Counting — Pre-flight Cost Estimate")
    print("=" * 60)

    sample_messages = [{"role": "user", "content": "What tents do you carry?"}]
    tokens = estimate_tokens(sample_messages)
    est_cost = tokens * 3 / 1_000_000  # Sonnet input = $3/M
    print(f"Estimated input tokens: {tokens}")
    print(f"Estimated input cost:   ${est_cost:.5f}")
    print("(After first request, cached tokens cost ~90% less)")


if __name__ == "__main__":
    demo_token_estimate()
    demo_chat()
    # demo_batch()   # uncomment to run the batch demo (submits real API calls)
