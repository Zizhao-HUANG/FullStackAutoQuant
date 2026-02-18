"""UI component aggregate exports."""

from .backtest_view import BacktestDependencies, BacktestPage
from .config_view import ConfigDependencies, ConfigPage
from .history_view import HistoryDependencies, HistoryPage
from .layout import PageHeader
from .positions_view import PositionPageDependencies, PositionsPage
from .workflow_view import WorkflowDependencies, WorkflowPage

__all__ = [
    "PageHeader",
    "PositionPageDependencies",
    "PositionsPage",
    "WorkflowDependencies",
    "WorkflowPage",
    "HistoryDependencies",
    "HistoryPage",
    "ConfigDependencies",
    "ConfigPage",
    "BacktestDependencies",
    "BacktestPage",
]
