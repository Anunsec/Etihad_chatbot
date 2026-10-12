# Etihad local assistant — build report

## Delivered routing

- A first-time Etihad Hub RAG match is narrowed to question-matching excerpts and refined by
  LangChain/OpenAI into a concise answer with source citations.
- A repeated normalized policy question uses a shared Redis cache (SQLite fallback) across users, returning the
  saved answer and citations with **zero new LLM calls**.
- A RAG miss is routed by LangGraph to LangChain `ChatOpenAI`.
- Flight status, schedule, and route requests invoke AviationStack through LangChain's in-process
  `MCPAdapter`, then use `ChatOpenAI` to summarize the normalized MCP result.
- Missing or failed AviationStack calls fail closed and do not invent flight data.
- Missing or failed OpenAI calls return `tool_unavailable` and are never disguised as answers.
- Flight-status date replies keep their pending flight intent and go to AviationStack, not the
  generic OpenAI answer node. Recognized provider values are shown verbatim.
- Future dates use AviationStack's `/flightsFuture` schedule after resolving the departure airport
  through `/routes`; a confirmed future schedule is labelled `scheduled`, not a live status.
- Airport-arrivals requests route to AviationStack's current-day `/timetable` via the
  `list_airport_arrivals` MCP tool. Delhi resolves to `DEL`; only matching provider rows appear.
- Earliest-flight requests invoke `get_first_flight` through MCP, filter and sort AviationStack
  results, and fail closed if no complete provider result exists. Unspecified Delhi dates use
  today's Delhi date; London is identified as Heathrow (`LHR`) in the answer.
- Economy baggage questions use a separately cited Etihad India fares FAQ table rather than
  incidental Hub mentions of pets or extra-seat baggage.

## Stack

- FastAPI backend with JWT `user` / `admin` RBAC.
- Streamlit chat UI and an admin observability dashboard.
- Etihad `OIP.webp` logo in the Streamlit header and browser tab icon.
- Etihad-inspired Streamlit styling with dark teal and gold, a responsive welcome panel,
  quick topic prompts, compact source details, and tabbed admin pages.
- LangChain documents, prompts, `ChatOpenAI`, and MCP adapter.
- LangGraph conditional nodes for retrieval, MCP tools, local answers, and LLM answers.
- LangSmith tracing when configured.
- SQLite LLM usage accounting and always-on JSONL audit events.
- Persistent Chroma policy vectors with BM25 fusion; SQLite stores user-scoped chat history.

## Admin dashboard

The admin-only dashboard shows:

- OpenAI, LangSmith, model, and AviationStack configuration state.
- Total/successful/failed LLM calls.
- Input, output, cached-input, reasoning, and total tokens.
- Average latency, daily usage, per-model usage, and per-graph-node usage.
- Users, conversations, messages, answer decisions, and feedback.
- Policy-answer cache entries and cache hits that avoided a new OpenAI call.
- Active cache/RAG backends, newly indexed Chroma chunks, embedding batch/query calls, and
  storage fallback warnings.
- Recent normalized LLM calls and audit events, with JSON downloads.

## Etihad Hub snapshot

- Source: <https://www.etihadhub.com/en-ae/policy-library>
- Documents: **55**
- Retrieval chunks: **194**
- Categories: Book 32, Change 12, GDS best practice 7, Fare audit 3, GTAA Addendum 1.

## Verification

- `pytest`: **39 passed**
- `ruff`: **all checks passed**
- `pip check`: **no broken requirements**
- FastMCP, LangChain MCP adapter, and `ChatOpenAI` imports verified.
- FastAPI live health reports conditional RAG/OpenAI routing and MCP flight transport.
- Redis cache behavior, TLS guard, Chroma persistence/restart, and chat-history isolation tested.
- Streamlit live health: `ok`.
- Regression tests cover the exact first-flight and Economy baggage questions. The latter now
  presents the official Economy fare table to OpenAI and reuses the answer from shared cache
  for an equivalent later question.

## Configuration required

The local settings loader sees configured OpenAI, AviationStack, and LangSmith API keys. Keys are
not included in the deliverable archive. To configure a fresh installation, set:

```dotenv
OPENAI_API_KEY=your-openai-key
OPENAI_CHAT_MODEL=gpt-4o-mini
```

`AVIATIONSTACK_API_KEY` is read from the same `.env` file. Enable hosted LangSmith traces with:

```dotenv
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-key
LANGSMITH_PROJECT=etihad-local-mvp
```

The archive includes a placeholder `.env.example`; copy it to `.env` and replace the sample
values. No real key is included.

Set `REDIS_URL` to a TLS `rediss://` Redis Cloud endpoint. The user-provided URL was a
remote plaintext `redis://` connection, so this build does not embed it and refuses that
transport by default. Until a TLS URL is configured, the existing SQLite cache is used.
Policy vectors persist under `data/chroma`; the first cache miss builds that index with
OpenAI embeddings. Recent chat turns remain in user-scoped SQLite conversations.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn etihad_assistant.api:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run ui/streamlit_app.py
```

Open <http://localhost:8501>. Sign in as an admin and select **Admin dashboard** in the sidebar.

## EY219 follow-up fix

The regression test covers `what the flight status of ey 219` followed by `2026-10-10` in the
same conversation. It verifies that the second turn invokes `get_flight_status`, displays the
provider status, and records one OpenAI summary call. Additional tests ensure no status is
fabricated when AviationStack has no matching record, reports an unrecognized status, or needs
the departure airport for a future schedule. The actual EY219 status on 2026-10-10 was not
rechecked in this update; the earlier check failed before the TLS configuration change.

## Delhi airport arrivals routing fix

`Show flights arriving at Delhi airport.` no longer enters policy RAG. It calls the AviationStack
MCP arrivals tool for `DEL`, displays up to 20 matching current-board rows, and logs an OpenAI
summary call when data is available. Tests cover the exact request, MCP-to-provider integration,
airport clarification, and unavailable data. Arrivals were not rechecked after the TLS fix, so
no actual arrivals are claimed in this report.

## Question-focused RAG refinement and cache

The policy path now checks a shared SQLite answer cache before retrieval. On a miss, local BM25
retrieves policy chunks, a focused excerpt selector supplies only question-matching passages to
OpenAI, and the successful answer is stored with its source IDs. A later user asking the same
normalized question receives the cached answer and citations without a new OpenAI call. Cache
keys include the corpus digest, model, and prompt version; account-specific questions are not
shared. Failures and flight-tool answers are not cached. The admin dashboard shows cache entries
and hits separately from provider-side cached-input tokens.

The Hub corpus does not contain a general Economy baggage allowance. A separate policy chunk
from <https://www.etihad.com/en-in/help/faq/fares> (checked 2026-10-07) supplies the 7 kg cabin
bag and fare-specific checked allowance for Delhi–Abu Dhabi. Its text participates in the corpus
digest, so an older cached answer cannot mask the new evidence. The Economy baggage retriever
focuses on this source instead of irrelevant route or special-case chunks.

## Delhi–Abu Dhabi first-flight routing fix

Reproduced the reported sequence before editing: `whats the first flight from delhi to abudhabi`
resolved `DEL` but not `AUH`, prompting for a destination. The follow-up `i said abu dhabi` lost
the pending flight intent and entered the general LLM path. The router now recognizes Abu Dhabi
without spaces or with a hyphen, accepts short airport clarification replies, and preserves the
pending `first_flight` request. Regression tests cover the exact wording, a clarification
follow-up, and avoidance of an unrelated policy route. Successful MCP responses display the
provider's flight number and available schedule, airline, terminal, gate, and status fields.

The earlier `get_first_flight` connection failed before an API response. The pasted MCP example
used AviationStack's direct host and `/flights` route filters. This update adapts its
`MCPToolManager` categories in `flight_mcp.py`, switches the provider to the direct host, and
uses the operating system's trusted certificate store for HTTPS. Mock fallbacks, hard-coded
API keys, and invented codeshare/route claims from the example were intentionally omitted.

Live verification on 2026-10-07: the `DEL`–`AUH` MCP request returned seven matching provider
records and selected Etihad EY219 as the earliest scheduled departure at
`2026-10-07T04:50:00+00:00`. A full app chat request with the exact reported wording returned
`Decision: answer` and one AviationStack citation after an OpenAI summary. Flight data can
change; this is a debug observation, not a guaranteed future schedule.
