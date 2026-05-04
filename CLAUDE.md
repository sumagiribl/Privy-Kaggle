# CLAUDE.md

This file provides guidance to Claude Code when working in this directory.

## What This Is

**Privy-Kaggle** is the reproducible Kaggle submission artifact for the Gemma 4 Good Hackathon. It contains a single self-contained Python script (`privy_kaggle_demo.py`) that demonstrates the Privy privacy pipeline end-to-end using the Google AI Studio API.

The Android app source lives in a separate repo: `C:\Users\pvgir\AndroidStudioProjects\Privy`

---

## Running the Script

```powershell
# Windows (PowerShell) — emoji requires UTF-8 mode
$env:PYTHONUTF8 = "1"
$env:GOOGLE_AI_STUDIO_KEY = "your_key_here"  # or load from AndroidStudioProjects\Privy\local.properties
python privy_kaggle_demo.py
```

```bash
# Linux / Kaggle
pip install requests
export GOOGLE_AI_STUDIO_KEY=your_key_here
python privy_kaggle_demo.py
```

**API key location (local dev):** `C:\Users\pvgir\AndroidStudioProjects\Privy\local.properties`
→ `GOOGLE_AI_STUDIO_KEY=...`

The script exits 0 and produces clean output when the key is valid. Without a key it still runs (all API calls return a placeholder string and are skipped gracefully).

---

## Script Structure (15 cells, `# %%` markers)

| Cell | Purpose |
|------|---------|
| 1 | Markdown overview + pipeline diagram + model architecture table |
| 2 | Imports, API key, `call_gemma()`, model constants, connectivity check |
| 3 | Three synthetic test documents (W-2, health report, lease) |
| 4 | `detect_regex()` — deterministic PII detection, mirrors `RegexPiiDetector.kt` |
| 5 | `detect_gemma_ner()` — context-aware NER via E4B, mirrors `GemmaPiiDetector.kt` |
| 6 | `detect_pii()` — merge + dedup, mirrors `PiiPipeline.kt` |
| 7 | `generate_synthetic()` — fake data substitution, mirrors `SyntheticGenerator.kt` |
| 8 | `MappingVault` + re-personalization smoke test |
| 9 | `route_query()` — intent classification via E4B, mirrors `QueryRouter.kt` |
| 10 | Demo 1: W-2 (full pipeline — routing → NER → synthetic → cloud → repersonalize) |
| 11 | Demo 2: research query (no PII → direct cloud) |
| 12 | Demo 3: simple tip calculator (on-device path, simulated via E4B API) |
| 13 | Privacy audit report |
| 14 | Metrics & benchmarks table |
| 15 | Markdown: on-device screenshots + architecture diagram |

---

## Key Constants (Cell 2)

```python
GEMMA_MODEL           = "gemma-4-26b-a4b-it"   # Cloud reasoning (26B MoE)
GEMMA_MODEL_ON_DEVICE = "gemma-3n-e4b-it"       # Routing + NER simulation (E4B architecture)
```

`gemma-4-4b-it` does not exist on Google AI Studio. `gemma-3n-e4b-it` is the E4B architecture available via API. The real app runs the same architecture locally via AICore/LiteRT with no network call.

---

## Important Behaviors

**`call_gemma()` system_instruction retry:**
`gemma-3n-e4b-it` rejects `system_instruction` with a 400 error. `call_gemma()` automatically retries by inlining the system prompt into the user message. Do not remove this retry logic.

**NER timeout:**
NER uses `timeout=150` (not the default 90). The 26B model took 50–100s for thorough W-2 extraction. The 4B model takes ~6s. Keep the 150s value as a safety margin.

**Confidence threshold:**
`_build_entities()` skips NER entities with `confidence < 0.80`. This filters false positives (e.g. the 4B model tagging `"2025"` as a date with conf=0.70, which would corrupt "Tax Year 2025").

**Bullet-first NER parser:**
The 26B model outputs a reasoning trace + bullets + JSON. The E4B model outputs clean JSON. Both paths are handled: bullet parse (primary), `raw_decode` JSON fallback. Do not remove either path — the 26B model may be used as a fallback.

**`PYTHONUTF8=1`:**
Required on Windows for emoji output. Without it Python's cp1252 console encoding raises `UnicodeEncodeError` on the first emoji. Always set this env var before running on Windows.

---

## What NOT to Change

- Do not change `GEMMA_MODEL` — `gemma-4-26b-a4b-it` matches the hackathon's available model
- Do not change the `# %%` cell markers — Kaggle uses these to split the script into notebook cells
- Do not change `ModelReleaseStage` references in comments — they document the Android app behavior
- Do not add `requirements.txt` — the script self-installs `requests` if missing (Kaggle compatibility)

---

## Relationship to Android App

The Python script is a direct port of the Kotlin pipeline:

| Python (this repo) | Kotlin (AndroidStudioProjects\Privy) |
|--------------------|--------------------------------------|
| `detect_regex()` | `pii/RegexPiiDetector.kt` |
| `detect_gemma_ner()` | `pii/GemmaPiiDetector.kt` |
| `detect_pii()` | `pii/PiiPipeline.kt` |
| `generate_synthetic()` | `pii/SyntheticGenerator.kt` |
| `MappingVault` | `pii/MappingVault.kt` + `pii/Repersonalizer.kt` |
| `route_query()` | `pipeline/QueryRouter.kt` |
| `call_gemma()` | `pipeline/CloudGemmaClient.kt` |

---

## License

CC-BY 4.0 (required by the Gemma 4 Good Hackathon competition rules).
