# benchb0t — OSS Roadmap

> Stand: 2026-04-12 · 9 Module · 12 Levels · 3 Harnesses · 0 Tests

---

## Phase 0 — Housekeeping & Tech Debt (jetzt)

Das Fundament muss stabil sein bevor Contributors kommen.

### 0.1 dashboard.py aufbrechen
`dashboard.py` hat 1203 Zeilen — monolithisch. Aufspalten in:
- `framework/web/app.py` — FastAPI-Init, Mount, Startup
- `framework/web/routes_api.py` — REST-Endpoints (`/api/*`)
- `framework/web/routes_ws.py` — WebSocket-Logic (`/ws`)
- `framework/web/routes_html.py` — Template-Serving (`/`, `/builder`, `/analytics`)
- `framework/web/chat.py` — Chat-Kontext-Builder + SSE-Streaming
- `framework/web/providers.py` — Provider-Detection + Credential-Management

Vorteil: jede Datei < 200 LOC, Contributor sieht sofort wo was hin muss.

### 0.2 Duplicate Levels bereinigen
- `l-tatto-studio.yaml` + `l9-tattoo-studio.yaml` → eins davon löschen
- Naming-Convention definieren: `l{nn}-{slug}.yaml`

### 0.3 Type Hints vervollständigen
- `mypy --strict` auf alle Module
- `py.typed` Marker für Package
- Pydantic-Models für alle API Request/Response Bodies (statt rohe dicts)

### 0.4 Config-Validierung
- Pydantic-Model für `config.yaml` + `level.yaml` + `harness.yaml`
- Fehlermeldungen die einem Anfänger sofort sagen was falsch ist
- `benchbot validate levels/` CLI-Befehl

---

## Phase 1 — Testing & CI (Woche 1–2)

### 1.1 Unit Tests
- `test_scorer.py` — Score-Berechnung deterministisch prüfbar (kein Docker nötig)
- `test_store.py` — SQLite in-memory, Migrationen, Queries
- `test_recorder.py` — Agentlog schreiben/lesen roundtrip
- `test_api.py` — Mock-HTTP, Streaming, Fehlerbehandlung
- `test_container.py` — Mock-Docker, `write_file` tar-Logik
- `test_dashboard.py` — FastAPI TestClient für alle 24 Endpoints

### 1.2 Integration Tests
- `test_e2e.py` — `l99-test.yaml` mit Mock-LLM (httpx-mock oder WireMock)
- Score muss deterministisch bei gleicher Tool-Sequenz rauskommen
- Preview-Port muss zugeordnet werden können

### 1.3 CI Pipeline (GitHub Actions)
```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  lint:    ruff check + mypy --strict
  test:    pytest -v --cov=framework (kein Docker nötig)
  e2e:     Docker-in-Docker mit l99-test (nur main/release)
  build:   docker build . (Rauchtest)
```

### 1.4 Pre-commit Hooks
- `ruff check --fix`, `ruff format`
- `mypy`
- YAML-Lint für Levels/Harnesses

---

## Phase 2 — OSS Readiness (Woche 2–3)

### 2.1 Repository-Struktur
```
LICENSE                    (MIT oder Apache-2.0)
CONTRIBUTING.md            (wie Level erstellen, wie Tools hinzufügen)
CODE_OF_CONDUCT.md
CHANGELOG.md
.github/
  ISSUE_TEMPLATE/
    bug_report.md
    feature_request.md
    new_level.md          ← eigenes Template für Level-Contributions
  PULL_REQUEST_TEMPLATE.md
  workflows/ci.yml
```

### 2.2 CONTRIBUTING.md Kernpunkte
1. **Level beitragen** — YAML-Schema-Referenz + `benchbot validate` + PR-Template
2. **Tool beitragen** — Schema in `TOOL_SCHEMAS` + Handler in `dispatch_tool()` + Test
3. **Evaluator beitragen** — `_check_*` Methode in `scorer.py`
4. **Harness beitragen** — YAML + Endpoint-Docs

### 2.3 API-Dokumentation
- FastAPI Auto-Docs (`/docs`) aktivieren (Swagger UI)
- Jeder Endpoint bekommt Docstrings + Pydantic Response-Models
- OpenAPI-Spec wird automatisch generiert

### 2.4 PyPI-Release
- `pyproject.toml` finalisieren (Metadata, Classifiers, URLs)
- `benchbot` CLI-Befehl via Entry-Point
- Versioning: SemVer, Changelog

---

## Phase 3 — Feature-Ausbau (Woche 3–6)

### 3.1 Multi-Model Parallel Runs
- N Modelle gleichzeitig auf dasselbe Level
- `benchbot run --level l4 --harness slavko,hermes,oc-nano --parallel`
- Dashboard zeigt Side-by-Side Comparison in Echtzeit
- Store bekommt `run_group_id` für zusammenhängende Batches

### 3.2 Level-Packs & Kategorien
- Offizielle Packs: `starter-pack` (l1–l3), `webapp-pack` (l4,l6,l7,l8,l9), `data-pack` (l2,l5)
- Community-Pack-Format: `benchbot install-pack <git-url>`
- Tags werden durchsuchbar im Builder

### 3.3 Leaderboard Export & Sharing
- `benchbot export --format html` → Standalone-HTML-Report (offline-fähig)
- `benchbot export --format json` → Maschinenlesbar für CI-Integration
- Optional: Public Leaderboard Submission (opt-in, anonymisiert)

### 3.4 LLM-Judge Evaluator ausbauen
- Aktuell nur `script` und `exact_match` richtig genutzt
- `llm_judge` standardisieren: Rubrik-System (1–5 Sterne pro Criterion)
- Judge-Model konfigurierbar (Claude, GPT-4, lokales Modell)
- Calibration-Level: bekannte Outputs mit Soll-Scores

### 3.5 Snapshot & Replay
- `container.snapshot()` existiert schon — Replay-Funktion bauen
- Agentlog-Datei → Container aufsetzen → Tool-Calls Step-by-Step nachspielen
- Debugging: "Wo genau ist der Agent falsch abgebogen?"

---

## Phase 4 — Advanced Features (Woche 6–12)

### 4.1 Plugin-System für Custom Tools
```python
# plugins/my_tool/tool.py
SCHEMA = { "type": "function", "function": { ... } }
def handle(args: dict, container: LevelContainer) -> tuple[int, str]: ...
```
- `benchbot` lädt automatisch aus `plugins/` Verzeichnis
- Kein Core-Code-Änderung nötig für neue Tools

### 4.2 Multi-Container Levels
- Levels mit mehreren Services (z.B. Frontend + Backend + DB)
- Docker Compose pro Level statt einzelner Container
- Networking zwischen Containern
- Evaluator kann alle Container prüfen

### 4.3 Adaptive Difficulty
- Scorer trackt historische Performance pro Model
- Automatisch nächstes Level vorschlagen (zu leicht → harder, zu schwer → easier)
- "Career Mode" in der UI: Model muss Level-Reihenfolge meistern

### 4.4 Tournament Mode
- Bracket-System: 2 Modelle bekommen dasselbe Level, besserer Score gewinnt
- Round-Robin oder Single-Elimination
- Live-Dashboard mit Bracket-Visualisierung (Pixel-Style)

### 4.5 Cost Tracking
- Token-Nutzung pro Run messen (Input + Output Tokens)
- $/Level Berechnung für Cloud-Endpoints
- Effizienz-Metrik: Score-per-Dollar
- Store: neue Spalten `input_tokens`, `output_tokens`, `cost_usd`

---

## Phase 5 — Community & Ecosystem (ab Woche 12)

### 5.1 Level Marketplace
- GitHub-basiert: Jeder kann ein Level-Repo erstellen
- Registry: `benchbot search levels --tag webapp --difficulty 3-4`
- `benchbot install level <github-url>`
- Bewertungssystem: Community-Stars + durchschnittlicher Model-Score

### 5.2 Harness Registry
- Vorkonfigurierte Harnesses für populäre Modelle
- `benchbot install harness claude-sonnet-4`
- Auto-detect erweitern: nicht nur Port-Probing, auch Model-Discovery (Ollama `/api/tags`)

### 5.3 Dashboard Themes
- Aktuell: Gold/Amber Pixel
- Community-Themes: CRT-Green, Cyberpunk, Minimal-Light
- CSS-Variable-System ist schon vorbereitet (`:root` vars)

### 5.4 Webhook & Notification System
- Slack/Discord Webhook nach Run-Ende
- CI-Integration: benchbot als GitHub Action
- `on_complete: webhook: https://hooks.slack.com/...` in config.yaml

### 5.5 Internationalisierung
- UI aktuell Deutsch/Englisch gemischt
- Entscheidung: Englisch als Default (OSS) mit i18n-System
- Level-Instructions bleiben sprachunabhängig (Code ist universell)

---

## Priorisierte Reihenfolge

| Prio | Was | Warum | Aufwand |
|------|-----|-------|---------|
| 🔴 1 | dashboard.py aufbrechen | Blocker für alles andere — niemand reviewed 1200 LOC | 3–4h |
| 🔴 2 | Unit Tests + CI | Ohne Tests kein sicheres Refactoring | 6–8h |
| 🔴 3 | OSS Files (LICENSE, CONTRIBUTING) | Must-have vor Public Release | 2–3h |
| 🟡 4 | Type Hints + Pydantic Models | Code-Qualität + Auto-Docs | 4–5h |
| 🟡 5 | Config/Level Validation CLI | Contributor-DX | 2–3h |
| 🟡 6 | Multi-Model Parallel Runs | Killer-Feature für Vergleiche | 6–8h |
| 🟡 7 | LLM-Judge Evaluator | Qualitative Bewertung für komplexe Levels | 4–6h |
| 🟢 8 | Plugin-System | Langfristige Extensibility | 8–10h |
| 🟢 9 | Level Marketplace | Community-Growth | 10–12h |
| 🟢 10 | Tournament Mode | Fun-Factor + Marketing | 8–10h |

---

## Architektur-Entscheidungen für OSS

**Entscheidung 1: Kein Backend-Framework-Wechsel**
FastAPI bleibt. Es ist leicht, async-native, auto-documented. Kein Django/Flask Overhead.

**Entscheidung 2: Templates bleiben Inline**
Kein React/Vue/Svelte Build-Step. Ein einziges `pip install benchbot && benchbot dash` soll funktionieren. Die HTML-Templates sind self-contained — das ist ein Feature, kein Bug.

**Entscheidung 3: SQLite bleibt**
Für Single-Node-Benchmarks perfekt. Kein PostgreSQL Setup nötig. WAL-Mode für Concurrent Reads. Export nach JSON/CSV für externe Analyse.

**Entscheidung 4: OpenAI-kompatibel als einziges Interface**
Jeder LLM-Provider der `/v1/chat/completions` spricht funktioniert automatisch. Kein Anthropic-SDK, kein Google-SDK, kein proprietärer Code. Ollama, vLLM, LM Studio, OpenRouter — alles geht über dasselbe Interface.
