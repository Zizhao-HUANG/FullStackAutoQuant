"""Page layout component: sets Streamlit page metadata and title."""

from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


@dataclass(frozen=True)
class PageHeader:
    """Configure page title and sidebar description."""

    title: str = "Manual Trading Console"
    caption: str = (
        "Sync positions → Run strategy → View risk & history → Adjust config. All in one page."
    )

    def render(self) -> None:
        """Inject unified page header into Streamlit."""
        st.set_page_config(page_title=self.title, layout="wide")
        st.title(self.title)
        st.caption(self.caption)
        st.sidebar.title(self.title)
