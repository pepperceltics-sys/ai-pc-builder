@st.cache_data(show_spinner=False, ttl=3600)
def ai_analyze_builds_cached(sig: str, builds_minimal: list[dict], industry: str, budget: float, model: str):
    """
    Returns dict: {rank(int): {"pros":[...], "cons":[...]}}
    Cached by signature for 1 hour.

    IMPORTANT:
    - Catches rate limits and returns {"__error__": "..."} instead of crashing the app.
    - Retries with exponential backoff on RateLimitError / transient errors.
    """
    client, err = get_openai_client()
    if err:
        return {"__error__": err}

    system = (
        "You are a PC-building assistant. "
        "You will be given up to 5 candidate PC builds with limited fields. "
        "Only use the provided fields; if something isn't provided, treat it as unknown. "
        "Write concise, practical pros/cons for the target industry."
    )

    user = {
        "task": "Analyze each build and produce 2-3 pros and 2-3 cons. Focus on price-performance, suitability for the industry, and any obvious risk (e.g., low PSU headroom if draw close to PSU wattage).",
        "industry": industry,
        "budget_usd": float(budget),
        "builds": builds_minimal,
        "output_format": {
            "type": "json",
            "schema": [
                {"rank": 1, "pros": ["..."], "cons": ["..."]}
            ]
        }
    }

    # Reduce response size (helps avoid TPM/rate constraints)
    max_tokens = 450  # small + sufficient for 5 builds

    # Retry policy for rate limiting / transient failures
    max_attempts = 4
    base_delay = 1.0

    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(user)}
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )

            text = completion.choices[0].message.content or ""

            # Try parsing strict JSON
            try:
                data = json.loads(text)
            except Exception:
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1 and end > start:
                    data = json.loads(text[start:end+1])
                else:
                    return {"__error__": "AI response was not valid JSON. Try again."}

            out = {}
            for item in data:
                try:
                    r = int(item.get("rank"))
                    pros = item.get("pros", []) or []
                    cons = item.get("cons", []) or []
                    out[r] = {"pros": pros[:3], "cons": cons[:3]}
                except Exception:
                    continue
            return out

        except Exception as e:
            last_err = e
            msg = str(e).lower()

            # Detect rate-limit-ish problems robustly across SDK versions
            is_rate_limit = ("ratelimit" in msg) or ("rate limit" in msg) or ("429" in msg)

            # Exponential backoff retry on rate limit / transient network-ish issues
            if is_rate_limit or ("timeout" in msg) or ("temporarily" in msg) or ("connection" in msg):
                if attempt < max_attempts:
                    sleep_s = base_delay * (2 ** (attempt - 1))
                    time.sleep(sleep_s)
                    continue

            # Non-retryable error: return friendly message
            return {"__error__": f"OpenAI call failed: {type(e).__name__}"}

    # If all retries failed, show a rate limit friendly message
    return {"__error__": "Rate limit hit. Please try again in ~30–60 seconds."}
