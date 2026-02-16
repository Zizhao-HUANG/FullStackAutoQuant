"""UI component aggregate exports."""

from .layout import PageHeader
from .positions_view import PositionPageDependencies, PositionsPage
from .workflow_view import WorkflowDependencies, WorkflowPage
from .history_view import HistoryDependencies, HistoryPage
from .config_view import ConfigDependencies, ConfigPage
from .backtest_view import BacktestDependencies, BacktestPage

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
