"""USD prices per 1M tokens (input, output) used for cost tracking and budgets.
Indicative values - verify against current provider pricing and update this table."""

PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1": (2.00, 8.00),
    "claude-haiku": (1.00, 5.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-opus": (5.00, 25.00),
    "gemini-2.5-flash": (0.30, 2.50),
    "offline-extractive": (0.0, 0.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    m = model.lower()
    price = next((p for k, p in sorted(PRICES.items(), key=lambda kv: -len(kv[0])) if k in m), (0.0, 0.0))
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 6)
