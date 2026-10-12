# Etihad Passenger Travel Assistant — local MVP

This project provides:

- FastAPI backend with JWT authentication and `user` / `admin` RBAC.
- Streamlit chat UI with source passages, feedback, and an admin usage dashboard.
- LangGraph conditional routing and state management.
- Persistent Chroma embeddings plus BM25 retrieval over the Etihad Hub policy library.
- A separately cited official Etihad Economy fare/baggage source for general allowances absent
  from the Hub snapshot.
- Question-focused policy answers: local RAG selects relevant passages, then LangChain
  `ChatOpenAI` refines the answer and attaches source citations.
- A shared Redis Cloud policy-answer cache skips OpenAI for repeated normalized questions across
  users. If Redis is unavailable, answers are not cached; SQLite is never used for this cache.
- User-scoped saved conversations and messages in SQLite, shown in the Streamlit sidebar.
- LangGraph SQLite checkpoints retain each conversation's graph state across backend restarts.
- LangChain `ChatOpenAI` for evergreen travel advice without a RAG match, and for summarizing
  AviationStack MCP results. Unsupported policy questions are refused without an LLM call.
- Optional LangSmith traces, normalized SQLite LLM usage records, and a local JSONL audit log.
- AviationStack-only flight status, schedules, and routes through an in-process MCP adapter, with
  no synthetic fallback.

Production and Azure/Foundry deployment are intentionally out of scope.

## Run locally

Use Python 3.11 or newer in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Edit `.env`, set a strong `JWT_SECRET`, and add:

```dotenv
OPENAI_API_KEY=your-openai-key
OPENAI_CHAT_MODEL=gpt-4o-mini
AVIATIONSTACK_API_KEY=your-aviationstack-key
REDIS_URL=rediss://default:your-password@your-host:your-port
REDIS_REQUIRE_TLS=true
RAG_BACKEND=chroma
CHROMA_PERSIST_DIRECTORY=data/chroma
# Optional; defaults to data/app.checkpoints.db when DATABASE_PATH is data/app.db
LANGGRAPH_CHECKPOINT_PATH=data/app.checkpoints.db
```

The OpenAI key is used for first-time policy-answer refinement, general travel advice without a
RAG match, and AviationStack/MCP summaries. Chroma indexing and semantic queries use the OpenAI
embedding model and therefore make embedding API calls; repeated cached policy questions make
neither embedding nor chat-model calls. The first policy miss builds the local Chroma index,
which can take time and incur embedding
cost. Subsequent restarts reuse it. Do not commit `.env` or a Redis password.

Redis Cloud should provide a TLS `rediss://` URL. A remote plaintext `redis://` URL is rejected
by default, with a visible no-cache warning in the admin dashboard. Only set
`REDIS_REQUIRE_TLS=false` if you understand and accept sending the Redis password and cache data
without transport encryption. The cache TTL defaults to seven days.

Start the backend:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn etihad_assistant.api:app --reload --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal, start the UI:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run ui/streamlit_app.py
```

Open `http://localhost:8501`. Default local accounts are `user` / `user-change-me` and
`admin` / `admin-change-me`; change both passwords before sharing the app.
When the assistant opens after sign-in, a welcome form asks for a name and email. The name is
used in the session greeting and sidebar. Both fields stay in Streamlit session state, are cleared
on sign-out, and are not sent to OpenAI or stored in the chat database or shared answer cache.
After signing in, select a saved conversation under **Recent conversations** in the sidebar
to read it or continue it. **New conversation** starts a separate thread; it does not delete
older conversations. Chats are associated with the account that created them.

## Refresh the Etihad Hub corpus

The checked-in corpus is `data/corpus/etihad_hub_policy_library.json`. It contains the complete
55-document snapshot from all five policy-library tabs, the original policy HTML, normalized text,
outbound links, and 194 retrieval chunks.

`data/corpus/etihad_official_economy_baggage.json` supplements that snapshot with the Economy
baggage table from [Etihad's India fares FAQ](https://www.etihad.com/en-in/help/faq/fares), checked
on 2026-10-07. This source is kept separate so its provenance is clear. For Delhi–Abu Dhabi,
Economy cabin baggage is one bag up to 7 kg; checked baggage varies by fare (Basic: fees apply;
Value: 25 kg; Comfort: 30 kg; Deluxe: 40 kg). The app refines an answer from this source and
then caches it for the same normalized question.

Refresh it from the live public page with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m etihad_assistant.ingestion
```

If a managed corporate TLS inspection certificate is not available to Python, use `--insecure`
for this public page only:

```powershell
python -m etihad_assistant.ingestion --insecure
```

The scraper requires exactly five policy groups and fails rather than silently producing a partial
corpus. Its output records the scrape timestamp and SHA-256 of the source HTML.

## Conditional RAG, MCP, and LLM routing

1. The corpus loader converts chunks to LangChain `Document` objects.
2. Chroma persists embeddings locally and fuses semantic matches with BM25 exact-term ranking.
   If Chroma is unavailable, retrieval falls back to BM25.
3. Before retrieval, LangGraph checks the shared policy cache using the normalized question,
   corpus digest, model, and prompt version. A hit returns the answer and citations with no LLM call.
4. On a miss, the policy node selects a few question-matching excerpts and asks `ChatOpenAI` for
   a concise, source-grounded answer. It stores successful generic policy answers only in Redis.
   If Redis is unavailable, the answer is returned without caching.
5. Audit records distinguish cache hits (`llm_calls: 0`) from refinements (`llm_calls: 1`).

The cache ignores case, repeated whitespace, and trailing sentence punctuation, but does not
reuse an answer for a different question. Questions containing obvious personal/account terms
are not shared across users. A corpus, model, or prompt change produces a new cache key. If the
snapshot only covers a special case—for example, extra-seat cabin baggage rather than a general
baggage allowance—the answer should state that limit instead of inventing general rules.
Recent conversation turns are loaded only from the signed-in user's conversation; context-dependent
follow-ups are not placed in the shared answer cache. Chroma stores policy vectors, not chat memory.
LangGraph uses a SQLite checkpointer (default `data/app.checkpoints.db`) with a thread ID derived
from the authenticated user and conversation. The app's primary SQLite database (`data/app.db`)
holds the full message history and powers the sidebar. These are distinct from the shared Redis
policy-answer cache. Existing conversations saved before checkpointing remain readable and their
state is migrated to a checkpoint when continued.

If no passage answers a policy question, LangGraph explicitly says the verified information is
unavailable; it does not ask OpenAI to guess. Only general, evergreen travel-advice questions may
use the general-answer LLM node. Clearly off-topic requests are politely declined without an LLM
call. The policy prompt requires conditional, source-grounded language for refunds, waivers, and
compensation; an additional output check blocks individualized benefit promises before they can
be served or cached. The policy prompt version changed, invalidating older shared-cache entries.

Flight requests are sent to AviationStack through LangChain's in-process MCP
adapter, and the normalized MCP result is then summarized by the OpenAI node.
For a flight-status question missing a date, reply with `YYYY-MM-DD` in the same chat; the
flight number and intent are retained. The answer displays AviationStack's exact recognized
status (`scheduled`, `active`, `landed`, `cancelled`, `incident`, or `diverted`). If no matching
record exists, it reports that instead of guessing. For future dates, the provider looks up the
flight's departure airport and checks AviationStack's `/flightsFuture` confirmed schedule. A
future `scheduled` answer is marked as schedule information, not live operational status. If
the departure airport cannot be resolved, the assistant asks for its IATA code.
Airport-arrivals requests such as `Show flights arriving at Delhi airport` are routed to the
`list_airport_arrivals` MCP tool, which calls AviationStack's current-day `/timetable` endpoint
with `iataCode=DEL` and `type=arrival`. The answer displays up to 20 matching provider records
and uses OpenAI only to summarize those records. If the airport is unclear, the assistant asks
for its city or IATA code; an empty or unavailable board never falls through to policy RAG.
`First flight from Delhi to London` routes to the `get_first_flight` MCP tool. With no date,
it uses today's Delhi date; London defaults to Heathrow (`LHR`) and the answer explicitly notes
that other London airports are excluded. AviationStack data is filtered by route/date and sorted
by scheduled departure. If data is missing or incomplete, the app reports that and never invents
a flight. A future date uses AviationStack's future schedule rather than a live status.
`Abu Dhabi`, `Abu-Dhabi`, and `abudhabi` resolve to `AUH`. If an airport is missing and the app
asks for it, replies such as `i said abu dhabi` retain the pending flight request and call MCP.
Connection failures show a specific outbound HTTPS/DNS/proxy diagnostic; no flight is fabricated.

## LangSmith and logs

Local audit events are appended to `logs/audit.jsonl`; raw user messages are not written there.
To send graph traces to LangSmith, configure:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=etihad-local-mvp
```

LangSmith is observability only. Enabling it does not add an LLM call.

Every attempted OpenAI call is also stored locally with model, graph-node name, token counts,
cached and reasoning token details when available, latency, status, and error type. Sign in as an
admin and choose **Admin dashboard** to view usage charts, app activity, feedback, recent calls,
policy-cache entries and hits,
and audit events.
The dashboard estimates USD chat-model cost from recorded input, cached-input, and output tokens
using published standard rates for supported models. It shows totals, per-model/graph-node cost,
and daily cost, and flags calls without sufficient usage or pricing. This is not the OpenAI invoice:
embedding calls, AviationStack, taxes, discounts, and historical rate changes are not included.

## API endpoints

| Method | Path | Role | Purpose |
|---|---|---|---|
| `POST` | `/auth/token` | public | Obtain a bearer token |
| `GET` | `/me` | user/admin | Inspect current identity |
| `POST` | `/chat` | user/admin | Retrieve a policy extract or use a flight tool |
| `GET` | `/conversations` | user/admin | List the signed-in user's saved chats |
| `GET` | `/conversations/{conversation_id}` | user/admin | Load messages for an owned chat |
| `POST` | `/feedback` | user/admin | Store thumbs up/down |
| `GET` | `/admin/audit-logs` | admin | Read local JSONL audit events |
| `GET` | `/admin/llm-calls` | admin | Verify normalized provider-call rows |
| `GET` | `/admin/dashboard` | admin | Aggregated LLM and application metrics |
| `GET` | `/health` | public | Health, corpus mode, and flight provider |

## AviationStack

`src/etihad_assistant/flights.py` calls the direct `https://api.aviationstack.com/v1` HTTPS host
and normalizes flight status, schedule, and route responses. It uses the operating system's
trusted certificates for TLS verification, which supports corporate-installed trust roots
without disabling verification. Route schedules and earliest-flight lookups query `/flights`
with `dep_iata` and `arr_iata`; future dates use `/flightsFuture`. The provider returns an
explicit error rather than mock flight data when the API fails.
`src/etihad_assistant/flight_mcp.py` adapts the supplied flight-status, flight-schedule, and
routes tool-manager pattern to the existing FastMCP tools and LangChain `MCPAdapter`. Only flight number,
date, and IATA airport codes are sent. If the key is absent, disabled, rejected, rate-limited, or
unavailable to the selected plan, the assistant returns `tool_unavailable` and never invents data.
The future-schedule endpoint may require a paid AviationStack plan and is distinct from the
`/flights` live/historical endpoint. See [AviationStack's endpoint guidance](https://blog.apilayer.com/introducing-continuous-flight-schedules/).

Run its MCP server with:

```powershell
python -m etihad_assistant.mcp_server
```

Run tests with:

```powershell
pytest
```

## MVP boundaries

- SQLite and local files replace managed databases and blob storage.
- The public policy page can change after the recorded scrape timestamp; refresh the corpus before
  relying on time-sensitive policy details.
- Schedule and route endpoints may require a paid AviationStack plan.
