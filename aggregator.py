def aggregate(decomp: dict, scoring: dict, wc_a: int = None, wc_b: int = None) -> dict:
    """
    decomp  — output of Prompt 1 (subclaims + weights)
    scoring — output of Prompt 2 (matched pairs + unmatched ids)
    """
    pairs     = scoring["matched_pairs"]
    u_a       = scoring["unmatched_a"]
    u_b       = scoring["unmatched_b"]
    subclaims_a = {s["id"]: s for s in decomp["reader_a"]["subclaims"]}
    subclaims_b = {s["id"]: s for s in decomp["reader_b"]["subclaims"]}

    # ── STEP 4: aggregate ────────────────────────────────────────────────────

    think_r = think_c = think_d = 0.0
    feel_r  = feel_c  = feel_d  = 0.0
    total_w = 0.0

    for p in pairs:
        w = min(p["weight_a"], p["weight_b"])
        think_r += w * p["think"]["R"]
        think_c += w * p["think"]["C"]
        feel_r  += w * p["feel"]["R"]
        feel_c  += w * p["feel"]["C"]
        total_w += w

    for aid in u_a:
        w = subclaims_a[aid]["weight"]
        think_d += w; feel_d += w; total_w += w

    for bid in u_b:
        w = subclaims_b[bid]["weight"]
        think_d += w; feel_d += w; total_w += w

    def to_pct(r, c, d):
        r_i, c_i, d_i = round(r/total_w*100), round(c/total_w*100), round(d/total_w*100)
        diff = 100 - (r_i + c_i + d_i)
        # apply rounding remainder to largest
        largest = max((r_i, "r"), (c_i, "c"), (d_i, "d"), key=lambda x: x[0])[1]
        if   largest == "r": r_i += diff
        elif largest == "c": c_i += diff
        else:                d_i += diff
        return r_i, c_i, d_i

    tr, tc, td = to_pct(think_r, think_c, think_d)
    fr, fc, fd = to_pct(feel_r,  feel_c,  feel_d)

    def dominant(r, c, d):
        return max(("resonate", r), ("contradict", c), ("diverge", d), key=lambda x: x[1])[0]

    # ── STEP 5: confidence ───────────────────────────────────────────────────

    if wc_a is None:
        wc_a = sum(len(s["claim"].split()) for s in decomp["reader_a"]["subclaims"])
    if wc_b is None:
        wc_b = sum(len(s["claim"].split()) for s in decomp["reader_b"]["subclaims"])
    depth = min(wc_a, wc_b) / max(wc_a, wc_b)

    n_a = len(subclaims_a)
    n_b = len(subclaims_b)
    coverage = len(pairs) / max(n_a, n_b) if (n_a + n_b) > 0 else 0.0

    if pairs:
        mapping_quality = sum(p["gate_confidence"] for p in pairs) / len(pairs)
    else:
        mapping_quality = 0.50

    confidence = 0.30 * depth + 0.40 * coverage + 0.30 * mapping_quality
    confidence = round(max(0.20, min(0.95, confidence)), 2)

    return {
        "passage_id":      scoring["passage_id"],
        "character_a":     decomp["reader_a"]["user_id"],
        "character_b":     decomp["reader_b"]["user_id"],
        "think":           {"R": tr, "C": tc, "D": td},
        "feel":            {"R": fr, "C": fc, "D": fd},
        "dominant_think":  dominant(tr, tc, td),
        "dominant_feel":   dominant(fr, fc, fd),
        "match_count":     len(pairs),
        "confidence":      confidence,
    }