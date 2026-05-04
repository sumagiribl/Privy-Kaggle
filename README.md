# 🔒 Privy — Your Private AI Workforce

**Kaggle Gemma 4 Good Hackathon Submission**

Privy is a native Android app that gives you a coordinated team of AI agents powered by Gemma 4 — where your personal data **never leaves your device**.

## Repository Contents

| File | Description |
|------|-------------|
| `privy_kaggle_demo.py` | End-to-end pipeline demo — run this on Kaggle |
| `screenshots/` | Device screenshots from real Pixel 8 Pro |
| `CLAUDE.md` | Claude Code guidance for working in this repo |
| `README.md` | This file |
| `LICENSE` | CC-BY 4.0 |

Android source code: [github.com/sumagiribl/Privy](https://github.com/sumagiribl/Privy)

## How to Run the Notebook

### On Kaggle

1. Upload `privy_kaggle_demo.py` to a Kaggle notebook (or open it as a script)
2. Add your Google AI Studio key as a Kaggle Secret named `GOOGLE_AI_STUDIO_KEY`
3. Run all cells — the notebook handles the secret automatically

### Locally

```bash
pip install requests
export GOOGLE_AI_STUDIO_KEY=your_key_here
python privy_kaggle_demo.py
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com).

## Prize Categories

| Prize | Relevance |
|-------|-----------|
| ⚡ **LiteRT Prize ($10K)** | All on-device AI runs via AICore/LiteRT on Pixel |
| 🛡️ **Safety & Trust Prize ($10K)** | PII redaction pipeline + real-time Privacy Dashboard |
| 🏆 **Main Track (up to $50K)** | Novel privacy-preserving multi-agent architecture |

## The Privacy Pipeline

```
User Input
  → Gemma 4 E4B (on-device) classifies intent, PII risk, complexity
  → Regex detects structured PII (SSN, phone, email, account numbers)
  → Gemma 4 E4B NER detects contextual PII (names, addresses, employers)
  → Synthetic substitution replaces PII with structurally-consistent fake data
  → Cloud Gemma 4 reasons over the sanitized query (sees zero real data)
  → Re-personalization restores real data in the response on-device
  → Privacy Dashboard shows full audit trail in real time
```

Zero real PII reaches the cloud. Verified on every query.

## Model Architecture

| Component | Real App (Pixel 8 Pro) | This Notebook |
|-----------|------------------------|---------------|
| Routing (intent + PII risk) | Gemma 4 E4B on-device via AICore/LiteRT | `gemma-3n-e4b-it` via API |
| PII NER detection | Gemma 4 E4B on-device via AICore/LiteRT | `gemma-3n-e4b-it` via API |
| Cloud reasoning | Gemma 4 26B via Google AI Studio | `gemma-4-26b-a4b-it` via API |

`gemma-3n-e4b-it` is the E4B architecture available via Google AI Studio API. The real app runs the same architecture locally via AICore — no network call for routing or NER.

## Note on Reproducibility

The actual Privy app runs Gemma 4 E4B **on-device** via AICore (LiteRT) — Kaggle cannot execute AICore. The notebook demonstrates the **identical pipeline logic** using the Google AI Studio API to simulate both on-device routing/NER and cloud reasoning. On-device benchmarks from the real device are included in Cell 14.

## License

CC-BY 4.0 — see [LICENSE](LICENSE) for details.
