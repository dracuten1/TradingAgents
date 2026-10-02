#!/usr/bin/env python3
"""TradingAgents Docker entrypoint — non-interactive, env-configured.

Works on upstream v0.5.2 API (TradingAgentsGraph.propagate with asset_type).
LLM backend is any OpenAI-compatible endpoint via provider `openai_compatible`
(9router :20128, Daoduc, local vLLM, ...).

Required env vars:
  TICKER          - Stock ticker symbol (e.g. NVDA, SPY, ETH-USD)
Optional env vars:
  ANALYSIS_DATE   - Date YYYY-MM-DD (default: today)
  LLM_PROVIDER    - default: openai_compatible
  OPENAI_BASE_URL - OpenAI-compatible endpoint (default: http://host.docker.internal:20128/v1)
  DEEP_MODEL      - Deep thinking model (default: 9agentic)
  QUICK_MODEL     - Quick thinking model (default: 9agentic)
  DEBATE_ROUNDS   - Max debate rounds (default: 1)
  ANALYSTS        - Comma-separated: market,social,news,fundamentals (default: all;
                    fundamentals auto-dropped for crypto)
  OUTPUT_LANG     - Output language (default: English)
"""

import os
import sys
import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv()
load_dotenv(".env.enterprise", override=False)

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from cli.prompts import detect_asset_type


def main():
    ticker = os.environ.get("TICKER", "").strip().upper()
    if not ticker:
        print("ERROR: Set TICKER env var (e.g. TICKER=NVDA)")
        sys.exit(1)

    analysis_date = os.environ.get("ANALYSIS_DATE", datetime.datetime.now().strftime("%Y-%m-%d"))
    provider = os.environ.get("LLM_PROVIDER", "openai_compatible")
    deep_model = os.environ.get("DEEP_MODEL", "9agentic")
    quick_model = os.environ.get("QUICK_MODEL", "9agentic")
    debate_rounds = int(os.environ.get("DEBATE_ROUNDS", "1"))
    output_lang = os.environ.get("OUTPUT_LANG", "English")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://host.docker.internal:20128/v1")

    # Parse analysts (drop fundamentals for crypto — crypto has no company financials)
    analysts_str = os.environ.get("ANALYSTS", "market,social,news,fundamentals")
    selected_analysts = [a.strip().lower() for a in analysts_str.split(",") if a.strip()]
    asset_type = str(detect_asset_type(ticker).value)
    if asset_type == "crypto":
        selected_analysts = [a for a in selected_analysts if a != "fundamentals"]
        print(f"Crypto ticker detected -> asset_type=crypto, analysts={selected_analysts}")

    # API key: provider-specific env var (see tradingagents.llm_clients.api_key_env)
    key_env = {
        "openai_compatible": "OPENAI_COMPATIBLE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider)
    api_key = os.environ.get(key_env, "") if key_env else ""
    if not api_key:
        print(f"ERROR: Set {key_env or 'provider API key env var'} in .env or environment")
        sys.exit(1)

    print(f"""
╔══════════════════════════════════════════════════════╗
║       TradingAgents v0.5.2 · env entrypoint          ║
╠══════════════════════════════════════════════════════╣
║  Ticker:     {ticker:<40s}║
║  Date:       {analysis_date:<40s}║
║  Provider:   {provider:<40s}║
║  Endpoint:   {base_url:<40s}║
║  Deep Model: {deep_model:<40s}║
║  Quick Model:{quick_model:<40s}║
║  Analysts:   {', '.join(selected_analysts):<40s}║
║  Debates:    {debate_rounds:<40d}║
║  Language:   {output_lang:<40s}║
╚══════════════════════════════════════════════════════╝
""")

    # Build config (v0.5.2: TRADINGAGENTS_* env overrides already folded into
    # DEFAULT_CONFIG; we set the values explicitly from our own env vars)
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = provider
    config["deep_think_llm"] = deep_model
    config["quick_think_llm"] = quick_model
    config["backend_url"] = base_url
    config["max_debate_rounds"] = debate_rounds
    config["max_risk_discuss_rounds"] = debate_rounds
    config["output_language"] = output_lang
    config["data_vendors"] = {
        "core_stock_apis": "yfinance",
        "technical_indicators": "yfinance",
        "fundamental_data": "yfinance",
        "news_data": "yfinance",
    }

    # Initialize graph
    print("Initializing TradingAgents graph...")
    ta = TradingAgentsGraph(
        selected_analysts=selected_analysts,
        config=config,
        debug=True,
    )

    # Run analysis
    print(f"\nAnalyzing {ticker} for {analysis_date}...\n")
    state, decision = ta.propagate(ticker, analysis_date, asset_type=asset_type)

    # Full multi-analyst report lives in state; decision is the parsed
    # 5-tier rating (Buy/Overweight/Hold/Underweight/Sell) or "REVIEW"
    full_decision = state.get("final_trade_decision") or str(decision)

    # Print decision
    print("\n" + "=" * 60)
    print("TRADING DECISION")
    print("=" * 60)
    print(decision)

    # Save report
    results_dir = Path(config["results_dir"]) / ticker / analysis_date
    results_dir.mkdir(parents=True, exist_ok=True)
    report_file = results_dir / "decision.md"
    report_file.write_text(
        f"# Trading Analysis: {ticker}\n"
        f"Date: {analysis_date}\n"
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Signal: {decision}\n\n"
        f"## Final Trade Decision\n\n{full_decision}\n",
        encoding="utf-8",
    )
    print(f"\nReport saved to: {report_file}")


if __name__ == "__main__":
    main()
