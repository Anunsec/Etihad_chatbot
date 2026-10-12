"""Build the user-facing Etihad Travel Assistant project guide PDF."""

from __future__ import annotations

from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf" / "etihad-travel-assistant-project-guide.pdf"
NAVY = colors.HexColor("#092E3A")
TEAL = colors.HexColor("#0E5260")
GOLD = colors.HexColor("#B18B54")
SAND = colors.HexColor("#F7F4EE")
PALE = colors.HexColor("#EAF2F3")
INK = colors.HexColor("#20353D")
MUTED = colors.HexColor("#53666D")
LINE = colors.HexColor("#D7E0E1")
WHITE = colors.white

PAGE_W, PAGE_H = A4
MARGIN = 43
CONTENT_W = PAGE_W - 2 * MARGIN

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverKicker", fontName="Helvetica-Bold", fontSize=10, leading=13,
    textColor=GOLD, spaceAfter=16, alignment=TA_LEFT,
))
styles.add(ParagraphStyle(
    name="CoverTitle", fontName="Helvetica-Bold", fontSize=29, leading=34,
    textColor=NAVY, spaceAfter=14,
))
styles.add(ParagraphStyle(
    name="CoverDeck", fontName="Helvetica", fontSize=12, leading=18,
    textColor=MUTED, spaceAfter=20,
))
styles.add(ParagraphStyle(
    name="Section", fontName="Helvetica-Bold", fontSize=16, leading=20,
    textColor=NAVY, spaceBefore=14, spaceAfter=10, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="Subsection", fontName="Helvetica-Bold", fontSize=11.5, leading=15,
    textColor=TEAL, spaceBefore=12, spaceAfter=6, keepWithNext=True,
))
styles.add(ParagraphStyle(
    name="BodyCustom", fontName="Helvetica", fontSize=9.2, leading=14,
    textColor=INK, spaceAfter=7,
))
styles.add(ParagraphStyle(
    name="SmallCustom", fontName="Helvetica", fontSize=8, leading=11.3,
    textColor=INK, spaceAfter=3,
))
styles.add(ParagraphStyle(
    name="MutedCustom", fontName="Helvetica", fontSize=8, leading=11.5,
    textColor=MUTED, spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="TableHeadCustom", fontName="Helvetica-Bold", fontSize=8.2, leading=11,
    textColor=WHITE,
))
styles.add(ParagraphStyle(
    name="TableCustom", fontName="Helvetica", fontSize=8, leading=11.2,
    textColor=INK,
))
styles.add(ParagraphStyle(
    name="TableFileCustom", fontName="Helvetica-Bold", fontSize=8, leading=11.2,
    textColor=NAVY,
))
styles.add(ParagraphStyle(
    name="QuoteCustom", fontName="Helvetica-Bold", fontSize=10, leading=15,
    textColor=NAVY, alignment=TA_CENTER,
))


def para(text: str, style: str = "BodyCustom") -> Paragraph:
    return Paragraph(text, styles[style])


def plain(text: str, style: str = "BodyCustom") -> Paragraph:
    return para(escape(text), style)


def heading(title: str, level: int = 1) -> Paragraph:
    return plain(title, "Section" if level == 1 else "Subsection")


def bullet(text: str) -> Paragraph:
    return para("<font color='#B18B54'><b>-</b></font>  " + text)


def grid(headers: list[str], rows: list[list[str]], widths: list[float]) -> LongTable:
    data = [[plain(item, "TableHeadCustom") for item in headers]]
    for row in rows:
        data.append([
            plain(value, "TableFileCustom" if column == 0 else "TableCustom")
            for column, value in enumerate(row)
        ])
    table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SAND]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, GOLD),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, LINE),
    ]))
    return table


def callout(label: str, text: str, background=PALE) -> Table:
    body = para(f"<b>{escape(label)}</b><br/>{text}", "BodyCustom")
    table = Table([[body]], colWidths=[CONTENT_W])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), background),
        ("BOX", (0, 0), (-1, -1), 0.5, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    return table


class ArchitectureDiagram(Flowable):
    """Vector architecture diagram sized for an A4 portrait page."""

    def __init__(self) -> None:
        super().__init__()
        self.width = CONTENT_W
        self.height = 376

    def draw(self) -> None:
        c = self.canv

        def box(x: float, y: float, w: float, h: float, lines: list[str], fill,
                text_color=INK, size=8.2) -> None:
            c.setFillColor(fill)
            c.setStrokeColor(LINE)
            c.roundRect(x, y, w, h, 6, fill=1, stroke=1)
            c.setFillColor(text_color)
            c.setFont("Helvetica-Bold", size)
            gap = 11
            first_y = y + h / 2 + (len(lines) - 1) * gap / 2 - 3
            for index, line in enumerate(lines):
                c.drawCentredString(x + w / 2, first_y - index * gap, line)

        def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
            c.setStrokeColor(MUTED)
            c.setFillColor(MUTED)
            c.setLineWidth(1.15)
            c.line(x1, y1, x2, y2)
            if abs(x2 - x1) >= abs(y2 - y1):
                direction = 1 if x2 > x1 else -1
                p = c.beginPath()
                p.moveTo(x2, y2)
                p.lineTo(x2 - 5 * direction, y2 + 3)
                p.lineTo(x2 - 5 * direction, y2 - 3)
            else:
                direction = 1 if y2 > y1 else -1
                p = c.beginPath()
                p.moveTo(x2, y2)
                p.lineTo(x2 - 3, y2 - 5 * direction)
                p.lineTo(x2 + 3, y2 - 5 * direction)
            p.close()
            c.drawPath(p, fill=1, stroke=0)

        top_y = 323
        top_w = 112
        box(0, top_y, top_w, 39, ["Passenger / admin"], PALE)
        box(154, top_y, top_w, 39, ["Streamlit UI"], PALE)
        box(308, top_y, top_w, 39, ["FastAPI + JWT", "role checks"], PALE)
        arrow(112, top_y + 19, 151, top_y + 19)
        arrow(266, top_y + 19, 305, top_y + 19)

        box(123, 259, 262, 40, ["AssistantService + LangGraph", "intent and state routing"], NAVY, WHITE)
        arrow(364, top_y, 364, 302)

        box(7, 210, 215, 31, ["Policy path"], TEAL, WHITE)
        box(286, 210, 215, 31, ["Flight path"], TEAL, WHITE)
        arrow(201, 259, 115, 244)
        arrow(308, 259, 394, 244)

        box(7, 161, 215, 38, ["Redis answer cache", "hit: return cited answer"], SAND)
        box(286, 161, 215, 38, ["FastMCP tools", "LangChain MCP adapter"], SAND)
        arrow(115, 210, 115, 202)
        arrow(394, 210, 394, 202)

        box(7, 109, 215, 38, ["Chroma + BM25", "Etihad corpus and supplements"], SAND)
        box(286, 109, 215, 38, ["AviationStack API", "validated live / schedule data"], SAND)
        arrow(115, 161, 115, 150)
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 7)
        c.drawString(124, 151, "miss")
        arrow(394, 161, 394, 150)

        box(7, 58, 215, 36, ["OpenAI refines policy evidence"], PALE)
        box(286, 58, 215, 36, ["OpenAI summarizes tool result"], PALE)
        arrow(115, 109, 115, 97)
        arrow(394, 109, 394, 97)

        box(123, 3, 262, 37, ["Response with decision + citations", "SQLite memory / usage; JSONL audit; optional LangSmith"], NAVY, WHITE, 7.6)
        arrow(115, 58, 175, 43)
        arrow(394, 58, 334, 43)


story: list[Flowable] = []

# Cover
story.extend([
    Spacer(1, 36),
    para("PROJECT ARCHITECTURE AND OPERATIONS", "CoverKicker"),
    para("Etihad Travel Assistant", "CoverTitle"),
    para("A complete guide to the local MVP: architecture, request paths, data stores, "
         "technology choices, source files, quality checks and a practical roadmap.", "CoverDeck"),
    Spacer(1, 17),
    callout("At a glance",
            "Policy facts come from local RAG; flight facts come from AviationStack through MCP. "
            "OpenAI refines new grounded answers. Redis shares eligible policy answers; "
            "SQLite preserves user-scoped chat history and usage."),
    Spacer(1, 23),
    grid(["Project snapshot", "Current implementation"], [
        ["Application", "FastAPI backend and Streamlit user/admin interface"],
        ["Policy corpus", "55 Etihad Hub documents, 194 chunks, plus 7 official supplement chunks"],
        ["Flight data", "AviationStack only; no synthetic flight fallback"],
        ["Model", "Configurable OpenAI chat model; default gpt-4o-mini; no fine-tuning"],
        ["Verification", "47 tests passed; Ruff and dependency checks passed on 10 October 2026"],
    ], [122, CONTENT_W - 122]),
    Spacer(1, 24),
    para("Code review date: 10 October 2026  |  Scope: local development MVP", "MutedCustom"),
    PageBreak(),
])

# Architecture
story.extend([
    heading("1. Architecture at a glance"),
    para("The web interface never calls a flight provider or OpenAI directly. It sends an "
         "authenticated chat request to FastAPI. The service loads conversation context, runs "
         "the LangGraph workflow, then persists the response and operational metadata."),
    ArchitectureDiagram(),
    Spacer(1, 8),
    callout("Three different stores",
            "Redis caches reusable policy answers. Chroma stores policy vectors, not chat memory. "
            "SQLite stores users, conversation state, messages, feedback and LLM usage."),
    heading("2. Technology stack"),
    grid(["Technology", "Why it is here"], [
        ["Python 3.11+ / Pydantic", "Implementation language and validated API models."],
        ["FastAPI / Uvicorn", "HTTP backend, authentication dependencies and local server."],
        ["Streamlit / CSS", "Chat UI, citations, feedback, login and admin dashboard."],
        ["LangGraph", "Conditional paths for policy, flight, clarification and refusal."],
        ["LangChain / OpenAI", "Prompts, chat-model calls, embeddings and MCP adapter."],
        ["Chroma + BM25", "Persistent semantic policy search fused with exact-term ranking."],
        ["Redis Cloud", "Shared answer cache with expiration; no SQLite cache fallback."],
        ["SQLite", "Local account, conversation, feedback and usage persistence."],
        ["FastMCP / AviationStack", "Live flight-status, schedule, arrival and route tools."],
        ["LangSmith / JSONL", "Optional hosted traces and always-on local audit records."],
    ], [130, CONTENT_W - 130]),
    heading("Why the components are separated", 2),
    para("Policy content, live operations data and user-specific conversation state have "
         "different freshness and privacy needs. Keeping those paths separate prevents "
         "a static policy excerpt from being mistaken for a current flight status, and "
         "prevents one user's chat history from entering the shared policy index."),
    bullet("Source changes update the corpus digest and invalidate old policy-answer cache keys."),
    bullet("Flight provider failures are explicit; they do not fall back to synthetic records."),
    bullet("Optional tracing can be disabled without losing the local audit or usage records."),
    PageBreak(),
])

# Runtime paths
story.extend([
    heading("3. Request lifecycle and model calls"),
    grid(["Request", "Execution path", "Chat-model call?"], [
        ["Repeated generic policy question", "Redis returns saved answer and source IDs.", "No"],
        ["New policy question with evidence", "Hybrid retrieval selects passages; OpenAI refines; eligible answer is cached.", "Yes"],
        ["Policy question without evidence", "Honest refusal; no unsupported policy is invented.", "No, except permitted general advice"],
        ["Valid flight request", "MCP invokes AviationStack; OpenAI summarizes validated data.", "Yes"],
        ["Missing flight detail", "Clarify and persist pending intent/slot for the next turn.", "No"],
        ["Off-topic or unavailable provider", "Decline or report no reliable provider data.", "No"],
    ], [132, 273, CONTENT_W - 405]),
    Spacer(1, 10),
    para("Embedding calls are separate from chat-model calls. On a policy cache miss, Chroma may "
         "embed the question and, on first use or corpus change, missing policy chunks. A Redis "
         "hit skips retrieval and both kinds of call."),
    heading("4. How policy RAG works"),
    para("The scraper converts the five Etihad Hub policy groups into titled, versioned chunks "
         "with source metadata. The repository adds separately identified official-source "
         "supplements. A BM25 lexical retriever ranks chunks; Chroma adds semantic matches and "
         "fuses both rankings. The graph then narrows the result to question-matching passages "
         "before requesting a concise, source-grounded OpenAI answer."),
    bullet("A cache key includes the normalized question, corpus digest, chat model and prompt version."),
    bullet("Account-specific questions and context-dependent follow-ups are not shared through Redis."),
    bullet("Source IDs are stored with cached answers so citations can be reconstructed."),
    bullet("If evidence is inadequate, the assistant refuses rather than guessing a policy."),
    para("Targeted source handling currently improves Economy baggage, Platinum benefits and "
         "onboard-meal questions. This is retrieval-augmented generation, not fine-tuning: "
         "the model weights are unchanged."),
    heading("5. How flight questions work"),
    para("The graph parses flight intent and available slots. Missing flight number, date or "
         "airport triggers a clarification; the pending intent is saved in SQLite. Once complete, "
         "the graph invokes an in-process FastMCP tool through LangChain's MCP adapter. The "
         "AviationStack adapter validates inputs, calls the provider, normalizes results, and "
         "returns a structured success, not-found, clarify or unavailable status."),
    para("Successful tool data is summarized by OpenAI, while key facts such as recognized "
         "flight status or earliest flight details are inserted from the validated tool result. "
         "Future confirmed schedules are labeled as schedules, not live operational status. "
         "A failed or empty provider result never becomes a synthetic flight."),
    PageBreak(),
])

# Storage, safety, API
story.extend([
    heading("6. Memory, cache and observability"),
    grid(["Component", "Stored information and boundary"], [
        ["SQLite", "Argon2-hashed accounts; user-owned conversations; recent messages; pending flight state; feedback; normalized LLM usage."],
        ["Redis", "Eligible policy answer draft plus cited source IDs, default seven-day TTL. Remote plaintext connections are rejected by default."],
        ["Chroma", "Persistent policy embeddings and metadata. It is not the conversation-memory store."],
        ["Local JSONL", "Turn decisions, IDs, retrieval metadata, latency and feedback events; not a raw chat transcript."],
        ["LangSmith", "Optional graph and model traces when configured; enabling it does not itself add an LLM call."],
    ], [105, CONTENT_W - 105]),
    Spacer(1, 9),
    para("The default context window loads up to eight recent messages from the signed-in "
         "user's conversation. Streamlit's New conversation control starts a new conversation ID; "
         "it does not erase prior SQLite rows. The current UI has no conversation-history browser."),
    heading("7. Authentication, guardrails and API"),
    para("FastAPI uses bearer JWTs and checks roles against the stored user record. Both user "
         "and admin roles can chat and provide feedback; audit, call records and the dashboard "
         "require admin. Guardrails decline clear off-topic requests, refuse unsupported policy "
         "claims and block individualized refund, waiver or compensation promises. These checks "
         "are rule-based and require continued evaluation."),
    grid(["Endpoint", "Role", "Purpose"], [
        ["POST /auth/token", "Public", "Sign in and obtain bearer token."],
        ["GET /me", "User / admin", "Read current identity."],
        ["POST /chat", "User / admin", "Run a policy or flight conversation turn."],
        ["POST /feedback", "User / admin", "Record helpful / needs-work feedback."],
        ["GET /admin/dashboard", "Admin", "Aggregated usage, cache and application metrics."],
        ["GET /admin/llm-calls", "Admin", "Recent normalized provider-call records."],
        ["GET /admin/audit-logs", "Admin", "Recent local JSONL audit records."],
        ["GET /health", "Public", "Configuration and application health summary."],
    ], [154, 78, CONTENT_W - 232]),
    PageBreak(),
])

# Source map
story.extend([
    heading("8. File-by-file reference"),
    para("Paths are relative to the project root. Source modules are listed individually; "
         "generated dependency and export files are grouped because they are not edited "
         "as application source."),
    heading("Backend package: src/etihad_assistant", 2),
    grid(["File", "Responsibility"], [
        ["__init__.py", "Package marker and version."],
        ["config.py", "Settings from environment/.env; model, key, storage, Redis TLS and provider options."],
        ["schemas.py", "Pydantic role, decision, citation, chat, feedback and flight contracts."],
        ["api.py", "FastAPI lifecycle, CORS, login, chat, feedback, health and admin endpoints."],
        ["auth.py", "JWT creation/validation and role-based dependencies."],
        ["db.py", "SQLite tables and queries for accounts, conversations, messages, feedback and usage."],
        ["service.py", "Constructs dependencies, runs turns, saves state and citations, writes audit events."],
        ["graph.py", "LangGraph intent parsing, cache, RAG, flight, clarification and refusal paths."],
        ["guardrails.py", "Travel-scope, unavailable-information and benefit-promise rules."],
        ["retrieval.py", "Corpus loader, PolicyChunk model, citation conversion and BM25 retrieval."],
        ["chroma_retrieval.py", "Persistent vector index, embeddings and rank fusion with BM25."],
        ["cache.py", "Redis-only policy-answer cache, TLS guard, retries, TTL and stats."],
        ["ingestion.py", "Etihad Hub parser and versioned corpus builder."],
        ["flights.py", "AviationStack HTTP client and normalized status/schedule/arrival/route data."],
        ["flight_mcp.py", "FastMCP tool definitions and in-process LangChain MCP adapter."],
        ["mcp_server.py", "Optional standalone MCP server entrypoint."],
        ["llm.py", "ChatOpenAI invocation and SQLite model-usage logging."],
        ["audit.py", "Append-only JSONL audit writer and tail reader."],
    ], [126, CONTENT_W - 126]),
    PageBreak(),
])

story.extend([
    heading("8. File-by-file reference (continued)"),
    heading("Interface and root", 2),
    grid(["File", "Responsibility"], [
        ["ui/streamlit_app.py", "Login, chat, citations, feedback, sidebar and admin dashboard."],
        ["ui/theme.css", "Etihad-inspired visual treatment and button/sidebar styles."],
        [".streamlit/config.toml", "Base Streamlit theme and toolbar options."],
        ["OIP.webp", "Brand image and browser tab icon."],
        ["pyproject.toml", "Package metadata, dependencies, scripts and test/lint configuration."],
        ["README.md", "Local setup, flow and endpoint documentation; one setup step is stale."],
        [".gitignore", "Excludes .env, virtual environment, databases, Chroma and audit logs."],
        ["txt.example", "Older sample settings, partly outdated."],
        ["n.txt", "Ancillary notes containing a credential-like key; staged in Git. Do not distribute."],
        ["outputs/BUILD_REPORT.md", "Historical development report; several cache descriptions are outdated."],
    ], [145, CONTENT_W - 145]),
    heading("Policy data and tests", 2),
    grid(["File", "Responsibility"], [
        ["data/corpus/etihad_hub_policy_library.json", "6 October 2026 Etihad Hub snapshot: 55 records, 194 chunks."],
        ["data/corpus/etihad_official_economy_baggage.json", "One official Economy baggage supplement chunk."],
        ["data/corpus/etihad_official_guest_platinum.json", "Three official Platinum-benefit supplement chunks."],
        ["data/corpus/etihad_official_onboard_meals.json", "Three official onboard/special-meal supplement chunks."],
        ["tests/test_api.py", "Authentication, roles and chat endpoint tests."],
        ["tests/test_core.py", "Routing, RAG, cache, guardrail, follow-up and usage regressions."],
        ["tests/test_aviationstack.py", "Provider and MCP normalization, error and no-synthetic-data tests."],
        ["tests/test_storage.py", "Redis, Chroma and user-scoped history tests."],
    ], [221, CONTENT_W - 221]),
    heading("Generated or local working files", 2),
    para("data/app.db is the local SQLite store; data/chroma/ is the persisted vector index; "
         "logs/audit.jsonl is the audit stream. work/policy-library.html is a saved source-page "
         "capture; the other work/*.db and work/*.jsonl files are local working artifacts. "
         "The seven outputs/*.zip files are historical export snapshots, not runtime dependencies. "
         "The .venv/, .pytest_cache/, .ruff_cache/ and .git/ directories are tooling state."),
    PageBreak(),
])

# Operations and roadmap
story.extend([
    heading("9. Running and verifying locally"),
    para("The app uses two terminals. Start FastAPI with: "
         "<font name='Courier'>.\\.venv\\Scripts\\Activate.ps1</font>, then "
         "<font name='Courier'>uvicorn etihad_assistant.api:app --reload --host 127.0.0.1 --port 8000</font>. "
         "Start Streamlit in another terminal with: "
         "<font name='Courier'>streamlit run ui/streamlit_app.py</font>. "
         "Open http://localhost:8501; inspect http://127.0.0.1:8000/health for backend status."),
    para("Required or optional .env variable names include OPENAI_API_KEY, AVIATIONSTACK_API_KEY, "
         "REDIS_URL, REDIS_REQUIRE_TLS, RAG_BACKEND, LANGSMITH_TRACING, LANGSMITH_API_KEY and "
         "JWT_SECRET. Do not copy secret values into source files or exported ZIPs."),
    callout("Documentation mismatch",
            "README.md asks new users to copy .env.example, but that file is absent from the "
            "current working tree. Use the existing ignored .env for this installation and "
            "restore a safe example file before distributing the project.", SAND),
    heading("10. Current boundaries"),
    bullet("No Azure Foundry, model fine-tuning, booking transactions or account-linked airline actions."),
    bullet("Policy sources are snapshots; publication may have changed since their checked dates."),
    bullet("The intent detector and promise guardrail are rules, not comprehensive classifiers."),
    bullet("The UI does not provide a conversation-history browser even though SQLite stores history."),
    bullet("The admin dashboard tracks usage, not authoritative billing cost or provider uptime."),
    bullet("Tests use controlled fakes and do not confirm live external-account access."),
    heading("11. Prioritized improvement plan"),
    grid(["Priority", "Recommendation and reason"], [
        ["P0 - security", "Rotate the credential-looking key in staged n.txt; remove it from Git/index and shared archives. Replace demo secrets; add a real password-update flow and rate limiting."],
        ["P1 - correctness", "Build representative evaluations for intent, MCP arguments, retrieval relevance, citation grounding, refusals and multi-turn follow-ups."],
        ["P1 - freshness", "Refresh policy sources with review, record effective dates, detect changes and expose source age in answers."],
        ["P1 - flight UX", "Handle airport/time-zone ambiguity more broadly and render validated provider data deterministically if OpenAI summarization fails."],
        ["P2 - RAG", "Check individual high-stakes claims against cited passages; tune retrieval thresholds; version cache against retrieval/index changes."],
        ["P2 - interface", "Add conversation listing/resume, useful feedback reasons and admin quality metrics."],
        ["Later", "Only if moving beyond local MVP: managed secrets, migrations/backups, retention, monitoring and deployment hardening."],
    ], [96, CONTENT_W - 96]),
    PageBreak(),
])

# Final notes and references
story.extend([
    heading("12. Key takeaways"),
    callout("In one sentence",
            "This is a policy-grounded, flight-data-aware local assistant: RAG supplies policy "
            "evidence, AviationStack supplies flight facts, OpenAI refines first-time answers, "
            "Redis reuses eligible answers, and SQLite remembers each user's conversation."),
    Spacer(1, 13),
    para("The highest-priority action is security cleanup of the staged credential-like text. "
         "The best quality investment after that is a repeatable evaluation dataset built from "
         "real questions and failures; fine-tuning is not a prerequisite. A source-grounded "
         "answer can still be wrong if retrieval chooses the wrong evidence or a source is stale."),
    heading("Source and review notes"),
    para("This document summarizes the files in the local project as inspected on "
         "10 October 2026. Source code is the authority where README.md or BUILD_REPORT.md "
         "describe older behavior. No live flight status, Redis Cloud connection or current "
         "OpenAI response was tested for this document."),
    para("Official Etihad policy sources recorded by the project:<br/>"
         "https://www.etihadhub.com/en-ae/policy-library<br/>"
         "https://www.etihad.com/en-in/help/faq/fares<br/>"
         "https://www.etihad.com/en/etihadguest/tiers-and-benefits<br/>"
         "https://www.etihad.com/en/plan/fly-with-etihad/onboard-services/meals<br/>"
         "https://www.etihad.com/en/help/special-assistance/special-meals", "SmallCustom"),
    para("Further reading for the roadmap:<br/>"
         "https://developers.openai.com/api/docs/guides/evaluation-best-practices<br/>"
         "https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html", "SmallCustom"),
    Spacer(1, 17),
    para("Verification: 47 pytest tests passed, Ruff passed, and pip check found no broken "
         "requirements. One LangChain MCP beta warning remains.", "MutedCustom"),
])


def decorate(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, PAGE_H - 37, PAGE_W - MARGIN, PAGE_H - 37)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(TEAL)
    canvas.drawString(MARGIN, PAGE_H - 29, "ETIHAD TRAVEL ASSISTANT")
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 29, "PROJECT GUIDE  |  10 OCT 2026")
    canvas.line(MARGIN, 39, PAGE_W - MARGIN, 39)
    canvas.drawString(MARGIN, 27, "Local MVP - architecture and implementation")
    canvas.drawRightString(PAGE_W - MARGIN, 27, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=52, bottomMargin=51,
        title="Etihad Travel Assistant - Project Guide",
        author="Codex",
        subject="Architecture, implementation, operations and roadmap",
    )
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(OUTPUT)


if __name__ == "__main__":
    main()
