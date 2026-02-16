"""Streamlit app entry point: aggregates pages and provides main frame."""

from __future__ import annotations

import streamlit as st

from app.services import bootstrap_services
from app.ui import (
    BacktestDependencies,
    BacktestPage,
    ConfigDependencies,
    ConfigPage,
    HistoryDependencies,
    HistoryPage,
    PageHeader,
    PositionPageDependencies,
    PositionsPage,
    WorkflowDependencies,
    WorkflowPage,
)


def main() -> None:
    """Launch Streamlit WebUI."""
    services = st.session_state.get("services")
    if services is None:
        services = bootstrap_services()
        st.session_state["services"] = services
    page_header = PageHeader()

    page_header.render()

    positions_page = PositionsPage(
        PositionPageDependencies(
            db=services.db,
            market=services.market,
            data_access=services.data_access,
            config=services.config,
        )
    )
    workflow_page = WorkflowPage(
        WorkflowDependencies(
            config=services.config,
            data_access=services.data_access,
            db=services.db,
            market=services.market,
        )
    )
    history_page = HistoryPage(HistoryDependencies(db=services.db, market=services.market))
    config_page = ConfigPage(ConfigDependencies(config=services.config))
    backtest_page = BacktestPage(BacktestDependencies(config=services.config))

    pages = {
        "📊 Position Management": positions_page.render,
        "🚀 Strategy Workflow": workflow_page.render,
        "📚 History & Logs": history_page.render,
        "⚙️ System Config": config_page.render,
        "📈 Backtest Analysis": backtest_page.render,
    }

    selected = st.sidebar.radio("Navigation", list(pages.keys()), index=0)
    render_fn = pages[selected]
    render_fn()


if __name__ == "__main__":
    main()
