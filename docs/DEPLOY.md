# Deploying FilingLens for free

Limits below were checked on 2026-11-09. Free plans change, so confirm them before relying on this.

## What is and isn't free

| Service | What the provider states | Use here |
|---|---|---|
| Vercel Hobby | $0/month, for personal, non-commercial use | Host the website |
| Hugging Face Spaces | Static Spaces are free. Docker and Gradio Spaces need a paid plan to create, apart from limited ZeroGPU Gradio Spaces | Not suitable for the API on a free account |
| Groq | Has a Free plan, but its pages didn't confirm the free limits, the models included, or whether a card is needed | Check in the Groq console before using |

The API loads embedding and reranking models, so it needs more memory than typical free server tiers provide.

## Recommended setup

1. **Public demo, free:** deploy the website to Vercel with recorded results from real runs, the dashboard, and the evaluation page. It needs no server, API key, or payment card.
2. **Full app, on your Mac:** run the API and Ollama locally, as described in the README, and show the live agent in interviews or a screen recording.

## Use a hosted model instead of Ollama

Any OpenAI-compatible service works by changing `backend/.env`:

```
LLM_PROVIDER=local
LOCAL_LLM_BASE_URL=https://api.groq.com/openai/v1
LOCAL_LLM_API_KEY=your-key
LOCAL_LLM_MODEL=a-tool-capable-model
DAILY_QUESTION_LIMIT=20
ALLOWED_ORIGINS=https://your-site.vercel.app
```

The API itself still needs a host with enough memory.

## Docker

```bash
docker compose up --build
```

Start Ollama on the host first. The Docker files have not been built or tested yet, because Docker wasn't running on the development machine.
