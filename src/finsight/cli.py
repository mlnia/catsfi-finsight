"""Command line interface.

    finsight ingest AAPL MSFT NVDA --forms 10-K --limit 1
    finsight ask "What are Apple's main supply chain risks?" --ticker AAPL
    finsight agent "Compare AAPL and MSFT revenue and show AAPL RSI"
    finsight documents
    finsight serve --port 8000
    finsight eval --dataset fixture
"""

from __future__ import annotations

import argparse
import json
import sys

from finsight.config import get_settings


def _container():
    from finsight.container import build_container
    from finsight.observability.tracing import setup_logging

    s = get_settings()
    setup_logging("WARNING")
    return build_container(s)


def cmd_ingest(a) -> int:
    c = _container()
    total = 0
    for t in a.tickers:
        rep = c.ingestion.ingest_ticker(t, tuple(a.forms), a.limit, a.force)
        for f in rep.filings:
            print(f"{f.ticker:6} {f.form:5} FY{f.fiscal_year} {f.accession}  {f.status:8} sections={f.sections:3} "
                  f"chunks={f.chunks:4} {f.seconds:6.1f}s {f.error or ''}")
        total += rep.chunks
    print(f"total chunks indexed: {total}; collection size: {c.store.count()}")
    return 0


def cmd_ask(a) -> int:
    from finsight.models import SearchFilters

    c = _container()
    flt = SearchFilters(ticker=a.ticker, form=a.form, fiscal_year=a.year) if (a.ticker or a.form or a.year) else None
    r = c.rag.answer(a.question, flt, prompt_ref=a.prompt)
    if a.json:
        print(r.model_dump_json(indent=2))
        return 0
    print(f"\n{r.answer}\n")
    for src in r.sources:
        print(f"  [{src.chunk_id}] {src.label}  {src.url or ''}")
    g = r.guardrails
    print(f"\nconfidence={r.confidence} action={g.action} groundedness={g.groundedness} "
          f"latency={r.latency_ms}ms cost=${r.usage.cost_usd} prompt={r.prompt_version} model={r.model}")
    return 0


def cmd_agent(a) -> int:
    c = _container()
    res = c.agent.run(a.message)
    while res.status == "awaiting_approval":
        print("\nApproval required for:")
        for tc in res.pending_approval:
            print(f"  - {tc['name']}({json.dumps(tc['arguments'])})")
        ok = a.yes or input("Approve? [y/N] ").strip().lower() == "y"
        res = c.agent.resume(res.thread_id, ok, "cli", None if ok else "rejected in CLI")
    print(f"\n{res.answer}\n")
    for t in res.tool_trace:
        print(f"  tool: {t['tool']}({json.dumps(t['arguments'])}) ok={t['ok']}")
    print(f"\nstatus={res.status} guardrails={res.guardrails.get('action')} usage={res.usage} latency={res.latency_ms}ms")
    return 0


def cmd_documents(a) -> int:
    for d in _container().store.documents():
        print(f"{d['ticker']:6} {d['form']:5} FY{d.get('fiscal_year')} filed {d.get('filing_date')}  chunks={d['chunks']}")
    return 0


def cmd_serve(a) -> int:
    import uvicorn

    uvicorn.run("finsight.api.main:app", host=a.host, port=a.port, reload=a.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="finsight", description="CATsFi FinSight CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest", help="download + index SEC filings")
    i.add_argument("tickers", nargs="+")
    i.add_argument("--forms", nargs="+", default=["10-K"], choices=["10-K", "10-Q"])
    i.add_argument("--limit", type=int, default=1)
    i.add_argument("--force", action="store_true")
    i.set_defaults(fn=cmd_ingest)

    q = sub.add_parser("ask", help="grounded Q&A")
    q.add_argument("question")
    q.add_argument("--ticker")
    q.add_argument("--form", choices=["10-K", "10-Q"])
    q.add_argument("--year", type=int)
    q.add_argument("--prompt")
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_ask)

    ag = sub.add_parser("agent", help="research agent (tools + human approval)")
    ag.add_argument("message")
    ag.add_argument("-y", "--yes", action="store_true", help="auto-approve gated tools")
    ag.set_defaults(fn=cmd_agent)

    sub.add_parser("documents", help="list indexed filings").set_defaults(fn=cmd_documents)

    sv = sub.add_parser("serve", help="run the API + UI")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--reload", action="store_true")
    sv.set_defaults(fn=cmd_serve)

    ev = sub.add_parser("eval", help="run the evaluation harness (see finsight.eval.runner -h)", add_help=False)
    ev.set_defaults(fn=None)

    args, rest = p.parse_known_args(argv)
    if args.cmd == "eval":
        from finsight.eval.runner import main as eval_main

        return eval_main(rest)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
