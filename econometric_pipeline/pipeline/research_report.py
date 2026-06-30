"""
research_report.py
════════════════════════════════════════════════════════════════════════
Professional PDF report generator for the Violence → Institutions →
FDI → Growth econometric pipeline.

Design philosophy
─────────────────
• World Bank / IMF Working Paper aesthetic — clean, minimal, authoritative.
• All tabular data presented as proper ReportLab tables (never monospaced text).
• Structured into labelled sections mirroring a formal research article.
• Header / footer on every page; cover page; auto table of contents.
• Colour palette: deep navy headers, alternating grey-white rows, black text.

Public API
──────────
  report = ResearchReport(title=..., ...)
  report.add_cover()                  # full-page cover
  report.add_toc()                    # table of contents
  report.add_section("1. Resumen Ejecutivo")
  report.add_body_text("...")
  report.add_table(headers, rows)     # generic styled table
  report.add_model_table(...)         # OLS coefficient table
  report.add_bootstrap_table(...)     # bootstrap inference table
  report.add_diagnostics_table(...)   # test-value-interpretation table
  report.add_ml_table(...)            # ML feature importance table
  report.add_interpretation_box(...)  # highlighted callout box
  report.add_image(path, caption)     # full-width figure with caption
  report.add_page_break()
  report.save()                       # builds & writes the PDF

Capture API (used by run_pipeline.py)
──────────────────────────────────────
  enable_reporting(report)   # starts stdout/stderr capture
  disable_reporting(report)  # stops capture
  get_active_report()        # returns active instance (or None)
════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import html
import io
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
from matplotlib.figure import Figure

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

# ──────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ──────────────────────────────────────────────────────────────────────────────

C_NAVY       = colors.HexColor("#0F2D52")   # dark navy — section headers, col headers
C_NAVY_LIGHT = colors.HexColor("#1A4A7A")   # medium navy — cover subtitle bar
C_SLATE      = colors.HexColor("#4B5563")   # slate grey — body meta text
C_HEADER_BG  = colors.HexColor("#E8EEF5")   # very light blue — table header bg alt
C_ROW_ALT    = colors.HexColor("#F4F7FB")   # near-white blue tint — alternating rows
C_ROW_BASE   = colors.white
C_GRID       = colors.HexColor("#C7D2E0")   # subtle grid lines
C_ACCENT     = colors.HexColor("#2563EB")   # electric blue — callout borders
C_INTERP_BG  = colors.HexColor("#F0F4FF")   # interpretation box background
C_WARN_BG    = colors.HexColor("#FFFBEB")   # warning-tone box background
C_POSITIVE   = colors.HexColor("#166534")   # positive significance green
C_NEGATIVE   = colors.HexColor("#991B1B")   # negative / insignificant red
C_PAGE_LINE  = colors.HexColor("#CBD5E1")   # header/footer rule line

# ──────────────────────────────────────────────────────────────────────────────
# STYLE SHEET
# ──────────────────────────────────────────────────────────────────────────────

ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")

def _build_styles() -> Any:
    base = getSampleStyleSheet()

    # ── Cover ──────────────────────────────────────────────
    base.add(ParagraphStyle(
        name="CoverTitle",
        fontName="Helvetica-Bold", fontSize=26, leading=32,
        alignment=TA_LEFT, textColor=colors.white, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="CoverSubtitle",
        fontName="Helvetica", fontSize=13, leading=17,
        alignment=TA_LEFT, textColor=colors.HexColor("#BFD7F7"), spaceAfter=6,
    ))
    base.add(ParagraphStyle(
        name="CoverMeta",
        fontName="Helvetica", fontSize=10, leading=14,
        alignment=TA_LEFT, textColor=colors.HexColor("#DDEEFF"), spaceAfter=4,
    ))

    # ── Document headings ──────────────────────────────────
    base.add(ParagraphStyle(
        name="SectionH1",
        fontName="Helvetica-Bold", fontSize=14, leading=18,
        textColor=C_NAVY, spaceBefore=14, spaceAfter=6,
        borderPad=4, keepWithNext=True,
    ))
    base.add(ParagraphStyle(
        name="SectionH2",
        fontName="Helvetica-Bold", fontSize=11, leading=14,
        textColor=C_NAVY_LIGHT, spaceBefore=10, spaceAfter=4, keepWithNext=True,
    ))
    base.add(ParagraphStyle(
        name="SectionH3",
        fontName="Helvetica-BoldOblique", fontSize=10, leading=13,
        textColor=C_SLATE, spaceBefore=8, spaceAfter=3, keepWithNext=True,
    ))

    # ── Body text ──────────────────────────────────────────
    base.add(ParagraphStyle(
        name="BodyPara",
        fontName="Helvetica", fontSize=9.5, leading=13,
        alignment=TA_JUSTIFY, spaceAfter=5,
    ))
    base.add(ParagraphStyle(
        name="Caption",
        fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
        alignment=TA_CENTER, textColor=C_SLATE, spaceBefore=3, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        name="FigureLabel",
        fontName="Helvetica-Bold", fontSize=9, leading=11,
        alignment=TA_CENTER, textColor=C_NAVY, spaceBefore=6, spaceAfter=2,
    ))
    base.add(ParagraphStyle(
        name="TableNote",
        fontName="Helvetica-Oblique", fontSize=8, leading=10,
        alignment=TA_LEFT, textColor=C_SLATE, spaceBefore=2, spaceAfter=6,
    ))

    # ── Interpretation / callout boxes ────────────────────
    base.add(ParagraphStyle(
        name="BoxTitle",
        fontName="Helvetica-Bold", fontSize=9.5, leading=12,
        textColor=C_NAVY, spaceAfter=3,
    ))
    base.add(ParagraphStyle(
        name="BoxBody",
        fontName="Helvetica", fontSize=9, leading=12,
        alignment=TA_JUSTIFY, spaceAfter=2,
    ))

    # ── TOC entry styles ───────────────────────────────────
    base.add(ParagraphStyle(
        name="TOCEntry1",
        fontName="Helvetica", fontSize=10, leading=14,
        leftIndent=0, spaceAfter=2,
    ))
    base.add(ParagraphStyle(
        name="TOCEntry2",
        fontName="Helvetica", fontSize=9, leading=12,
        leftIndent=18, spaceAfter=1, textColor=C_SLATE,
    ))

    return base


# ──────────────────────────────────────────────────────────────────────────────
# TABLE STYLE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _base_table_style(
    n_rows: int,
    n_cols: int,
    header_bg: Any = C_NAVY,
    header_fg: Any = colors.white,
    font_size: float = 8.5,
    padding: int = 5,
) -> TableStyle:
    """Return a styled TableStyle with alternating row colours."""
    cmds = [
        # Header row
        ("BACKGROUND",  (0, 0), (-1, 0),  header_bg),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  header_fg),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, 0),  font_size),
        ("ALIGN",       (0, 0), (-1, 0),  "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",  (0, 0), (-1, -1), padding),
        ("BOTTOMPADDING",(0, 0),(-1, -1), padding),
        ("LEFTPADDING", (0, 0), (-1, -1), padding + 1),
        ("RIGHTPADDING",(0, 0), (-1, -1), padding + 1),
        # Body rows
        ("FONTNAME",    (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",    (0, 1), (-1, -1), font_size - 0.5),
        # Grid
        ("LINEBELOW",   (0, 0), (-1, 0),  1.0, C_NAVY),
        ("LINEBELOW",   (0, -1),(-1, -1), 0.5, C_GRID),
        ("LINEBEFORE",  (0, 0), (0, -1),  0,   colors.transparent),
        ("LINEAFTER",   (-1, 0),(-1, -1), 0,   colors.transparent),
        ("GRID",        (0, 1), (-1, -1), 0.3, C_GRID),
        # Outer border
        ("BOX",         (0, 0), (-1, -1), 0.8, C_NAVY),
    ]
    # Alternating row backgrounds
    for r in range(1, n_rows):
        bg = C_ROW_ALT if r % 2 == 0 else C_ROW_BASE
        cmds.append(("BACKGROUND", (0, r), (-1, r), bg))
    return TableStyle(cmds)


def _p(text: str, style: Any) -> Paragraph:
    """Create a Paragraph, escaping HTML special chars."""
    safe = html.escape(str(text), quote=False)
    return Paragraph(safe, style)


def _sigstars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def _fmt(val: Any, decimals: int = 4) -> str:
    if val is None:
        return "n/a"
    try:
        f = float(val)
        if abs(f) < 1e-4 and f != 0:
            return f"{f:.2e}"
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


# ──────────────────────────────────────────────────────────────────────────────
# STDOUT CAPTURE
# ──────────────────────────────────────────────────────────────────────────────

class TeeStream:
    def __init__(self, stream: Any, report: "ResearchReport") -> None:
        self._stream = stream
        self._report = report

    def write(self, data: str) -> int:
        if not data:
            return 0
        try:
            self._stream.write(data)
        except UnicodeEncodeError:
            enc = getattr(self._stream, "encoding", None) or "utf-8"
            safe = data.encode(enc, errors="replace").decode(enc, errors="replace")
            self._stream.write(safe)
        try:
            self._stream.flush()
        except Exception:
            pass
        self._report.capture_text(data)
        return len(data)

    def writelines(self, lines: List[str]) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self) -> int:
        return getattr(self._stream, "fileno", lambda: -1)()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CLASS
# ──────────────────────────────────────────────────────────────────────────────

class ResearchReport:
    """
    Professional PDF report builder for econometric panel pipelines.

    Parameters
    ----------
    title           : Main title of the report (cover page H1).
    subtitle        : Subtitle (e.g., geographic scope).
    author          : Author name(s).
    script_name     : Name of the master script (displayed on cover).
    observations    : Number of panel observations.
    countries       : Number of unique countries / entities.
    year_range      : Tuple (start_year, end_year).
    pipeline_version: Pipeline version string.
    output_path     : Where to write the PDF.
    """

    def __init__(
        self,
        title: str = "Econometric Pipeline Report",
        subtitle: str = "Violence → Institutions → FDI → Economic Growth",
        author: str = "Pipeline Auto-generated",
        script_name: str = "run_pipeline.py",
        observations: Optional[int] = None,
        countries: Optional[int] = None,
        year_range: Optional[Tuple[int, int]] = None,
        pipeline_version: str = "1.0",
        output_path: Optional[Path] = None,
    ) -> None:
        self.title            = title
        self.subtitle         = subtitle
        self.author           = author
        self.script_name      = script_name
        self.observations     = observations
        self.countries        = countries
        self.year_range       = year_range
        self.pipeline_version = pipeline_version
        self.output_path      = Path(output_path or Path.cwd() / "econometric_report.pdf")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.story: List[Any] = []
        self._console_lines: List[str] = []
        self._pending_line = ""
        self._active = False
        self._styles = _build_styles()
        self._toc = TableOfContents()
        self._toc_entries: List[Tuple[int, str]] = []
        self._section_counter = 0

        # Patch matplotlib and pandas
        self._patch_matplotlib_savefig()
        self._patch_pandas_repr()

        # Redirect tracking
        self._stdout_redirector = None
        self._stderr_redirector = None

    # ──────────────────────────────────────────────────────────
    # PATCHES
    # ──────────────────────────────────────────────────────────

    def _patch_matplotlib_savefig(self) -> None:
        if getattr(Figure, "_research_report_patched", False):
            return
        original_savefig = Figure.savefig

        def wrapped(self_fig: Figure, *args: Any, **kwargs: Any) -> Any:
            result = original_savefig(self_fig, *args, **kwargs)
            report = get_active_report()
            if report is not None and report._active:
                # Only add if saved to a file path (not BytesIO), handled separately
                if args and isinstance(args[0], (str, Path)):
                    report.add_image(args[0])
            return result

        Figure.savefig = wrapped
        Figure._research_report_patched = True

    def _patch_pandas_repr(self) -> None:
        if getattr(pd.DataFrame, "_research_report_patched", False):
            return
        original_repr = pd.DataFrame.__repr__

        def wrapped(self_df: pd.DataFrame) -> str:
            text = original_repr(self_df)
            report = get_active_report()
            if report is not None and report._active:
                report.add_dataframe(self_df)
            return text

        pd.DataFrame.__repr__ = wrapped
        pd.DataFrame._research_report_patched = True

    # ──────────────────────────────────────────────────────────
    # CAPTURE
    # ──────────────────────────────────────────────────────────

    def start_capture(self) -> None:
        if self._active:
            return
        self._active = True
        self._stdout_redirector = TeeStream(sys.stdout, self)
        self._stderr_redirector = TeeStream(sys.stderr, self)
        sys.stdout = self._stdout_redirector
        sys.stderr = self._stderr_redirector

    def stop_capture(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._stdout_redirector is not None:
            sys.stdout = self._stdout_redirector._stream
        if self._stderr_redirector is not None:
            sys.stderr = self._stderr_redirector._stream
        self._stdout_redirector = None
        self._stderr_redirector = None

    def capture_text(self, data: str) -> None:
        if not data:
            return
        self._pending_line += data
        parts = self._pending_line.split("\n")
        self._pending_line = parts.pop()
        for part in parts:
            clean = ANSI_RE.sub("", part).rstrip("\r").rstrip()
            self._console_lines.append(clean)

    def flush_pending(self) -> None:
        if self._pending_line:
            clean = ANSI_RE.sub("", self._pending_line).rstrip()
            self._console_lines.append(clean)
            self._pending_line = ""

    # ──────────────────────────────────────────────────────────
    # COVER PAGE
    # ──────────────────────────────────────────────────────────

    def add_cover(self) -> None:
        """Full-page professional cover with navy background band."""
        now = datetime.now()

        # ── Top navy band ──
        band_data = [[""]]
        band = Table(band_data, colWidths=[7.5 * inch], rowHeights=[0.08 * inch])
        band.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_NAVY),
        ]))
        self.story.append(band)
        self.story.append(Spacer(1, 0.5 * inch))

        # ── Title block ──
        title_data = [
            [_p(self.title.upper(), self._styles["CoverTitle"])],
            [_p(self.subtitle, self._styles["CoverSubtitle"])],
        ]
        title_tbl = Table(title_data, colWidths=[7.5 * inch])
        title_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), C_NAVY),
            ("LEFTPADDING", (0, 0), (-1, -1), 20),
            ("RIGHTPADDING", (0, 0), (-1, -1), 20),
            ("TOPPADDING", (0, 0), (-1, -1), 14),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 18),
        ]))
        self.story.append(title_tbl)
        self.story.append(Spacer(1, 0.4 * inch))

        # ── Meta info box ──
        obs_str    = str(self.observations) if self.observations is not None else "N/D"
        cty_str    = str(self.countries)    if self.countries    is not None else "N/D"
        yr_str     = (f"{self.year_range[0]}–{self.year_range[1]}"
                      if self.year_range else "N/D")

        meta_rows = [
            ["Autor",             self.author],
            ["Fecha",             now.strftime("%d %B %Y")],
            ["Observaciones",     obs_str],
            ["Países / Unidades", cty_str],
            ["Período temporal",  yr_str],
            ["Script maestro",    self.script_name],
            ["Versión pipeline",  self.pipeline_version],
        ]
        s = self._styles
        meta_table_data = [
            [_p(k, s["BoxTitle"]), _p(v, s["BodyPara"])]
            for k, v in meta_rows
        ]
        meta_tbl = Table(meta_table_data, colWidths=[2.2 * inch, 4.5 * inch])
        meta_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), C_ROW_ALT),
            ("BACKGROUND",   (0, 0), (0, -1),  C_HEADER_BG),
            ("BOX",          (0, 0), (-1, -1),  1.0, C_NAVY),
            ("LINEAFTER",    (0, 0), (0, -1),   0.5, C_GRID),
            ("LINEBELOW",    (0, 0), (-1, -1),  0.3, C_GRID),
            ("LEFTPADDING",  (0, 0), (-1, -1),  8),
            ("RIGHTPADDING", (0, 0), (-1, -1),  8),
            ("TOPPADDING",   (0, 0), (-1, -1),  6),
            ("BOTTOMPADDING",(0, 0), (-1, -1),  6),
            ("VALIGN",       (0, 0), (-1, -1),  "MIDDLE"),
        ]))
        self.story.append(meta_tbl)
        self.story.append(Spacer(1, 0.6 * inch))

        # ── Bottom thin rule ──
        rule_data = [[""]]
        rule = Table(rule_data, colWidths=[7.5 * inch], rowHeights=[3])
        rule.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), C_NAVY)]))
        self.story.append(rule)

        self.story.append(PageBreak())

    # ──────────────────────────────────────────────────────────
    # TABLE OF CONTENTS
    # ──────────────────────────────────────────────────────────

    def add_toc(self) -> None:
        """Insert a manually-built table of contents page."""
        self.add_section_h1("Índice", register_toc=False)
        # Placeholder; will be filled in add_section_h1 calls
        self._toc_placeholder_idx = len(self.story)
        self.story.append(Spacer(1, 0.2 * inch))
        self.story.append(PageBreak())

    # ──────────────────────────────────────────────────────────
    # HEADINGS / SECTIONS
    # ──────────────────────────────────────────────────────────

    def add_section_h1(self, title: str, register_toc: bool = True) -> None:
        """Primary numbered section heading (e.g., '1. Resumen Ejecutivo')."""
        safe = html.escape(title, quote=False)
        self.story.append(_p(safe, self._styles["SectionH1"]))
        self.story.append(Spacer(1, 0.05 * inch))
        if register_toc:
            self._toc_entries.append((1, title))

    def add_section_h2(self, title: str) -> None:
        """Secondary sub-section heading."""
        safe = html.escape(title, quote=False)
        self.story.append(_p(safe, self._styles["SectionH2"]))
        self.story.append(Spacer(1, 0.03 * inch))

    def add_section_h3(self, title: str) -> None:
        """Tertiary heading."""
        safe = html.escape(title, quote=False)
        self.story.append(_p(safe, self._styles["SectionH3"]))

    # Backward-compat alias used by run_pipeline.py
    def add_heading(self, title: str) -> None:
        self.add_section_h1(title)

    # Backward-compat alias
    def add_title(self) -> None:
        self.add_cover()

    # ──────────────────────────────────────────────────────────
    # BODY TEXT
    # ──────────────────────────────────────────────────────────

    def add_body_text(self, text: str) -> None:
        """Add a justified body paragraph."""
        if not text or not text.strip():
            return
        for chunk in self._split_text(html.escape(text, quote=False)):
            try:
                self.story.append(_p(chunk, self._styles["BodyPara"]))
            except Exception:
                pass
        self.story.append(Spacer(1, 0.04 * inch))

    # Backward-compat
    def add_text(self, text: str, monospaced: bool = False) -> None:
        self.add_body_text(text)

    def add_summary(self, text: str) -> None:
        """Silently discard raw console summaries — data comes from JSON."""
        pass  # Intentionally suppressed; tables built from structured JSON

    def _split_text(self, text: str, max_chars: int = 2000) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        parts = []
        while text:
            if len(text) <= max_chars:
                parts.append(text); break
            split_at = text.rfind(" ", 0, max_chars)
            if split_at <= 0:
                split_at = max_chars
            parts.append(text[:split_at])
            text = text[split_at:].lstrip()
        return parts

    # ──────────────────────────────────────────────────────────
    # GENERIC TABLE
    # ──────────────────────────────────────────────────────────

    def add_table(
        self,
        data: Any,
        col_widths: Optional[List[float]] = None,
        font_size: float = 8.5,
        note: str = "",
    ) -> None:
        """
        Render a generic table.

        Parameters
        ----------
        data       : list-of-lists OR pd.DataFrame (first row = headers).
        col_widths : column widths in inches (auto-distributed if None).
        font_size  : font size for body cells.
        note       : footnote text appended below table.
        """
        if isinstance(data, pd.DataFrame):
            self.add_dataframe(data, col_widths=col_widths, note=note)
            return
        if not data:
            return
        n_rows = len(data)
        n_cols = max(len(r) for r in data)
        if col_widths:
            widths = [w * inch for w in col_widths]
        else:
            total = 7.2
            widths = [total / n_cols * inch] * n_cols

        # Build Paragraph-wrapped cells
        s = self._styles
        tbl_data = []
        for ri, row in enumerate(data):
            style = s["BoxTitle"] if ri == 0 else s["BodyPara"]
            tbl_data.append([_p(str(c), style) for c in row])

        tbl = Table(tbl_data, colWidths=widths, repeatRows=1)
        tbl.setStyle(_base_table_style(n_rows, n_cols, font_size=font_size))
        self.story.append(KeepTogether([tbl]))
        if note:
            self.story.append(_p(f"Nota: {note}", s["TableNote"]))
        self.story.append(Spacer(1, 0.1 * inch))

    def add_dataframe(
        self,
        df: pd.DataFrame,
        col_widths: Optional[List[float]] = None,
        note: str = "",
    ) -> None:
        if df.empty:
            return
        headers = [str(c) for c in df.columns]
        rows = df.round(4).astype(str).values.tolist()
        self.add_table([headers] + rows, col_widths=col_widths, note=note)

    # ──────────────────────────────────────────────────────────
    # MODEL COEFFICIENT TABLE
    # ──────────────────────────────────────────────────────────

    def add_model_table(
        self,
        dep_var: str,
        params: Dict[str, float],
        std_errs: Dict[str, float],
        pvalues: Dict[str, float],
        n_obs: int,
        r2_within: Optional[float] = None,
        estimator: str = "Two-Way FE (CL-SE)",
        key_vars: Optional[List[str]] = None,
        note: str = "",
    ) -> None:
        """
        Render a polished OLS/FE coefficient table.

        Columns: Variable | Coef. | Std. Err. | t-stat | p-value | Sig. | [95% CI]
        """
        headers = ["Variable", "Coef.", "Std. Err.", "t-stat", "p-value", "Sig."]
        rows = [headers]
        for var in params:
            coef = params[var]
            se   = std_errs.get(var, float("nan"))
            pval = pvalues.get(var, float("nan"))
            try:
                t = coef / se if se and se != 0 else float("nan")
            except Exception:
                t = float("nan")
            stars = _sigstars(pval)
            rows.append([
                var,
                _fmt(coef, 4),
                _fmt(se, 4),
                _fmt(t, 3),
                _fmt(pval, 4),
                stars,
            ])

        # Footer rows
        rows.append(["", "", "", "", "", ""])
        rows.append([f"Dep. variable: {dep_var}", "", "", "", "", ""])
        rows.append([f"Estimador: {estimator}", "", "", f"N = {n_obs}", "", ""])
        if r2_within is not None:
            rows.append([f"R² Within = {r2_within:.4f}", "", "", "", "", ""])

        n_data_rows  = len(rows)
        col_widths_in = [2.4, 0.9, 0.9, 0.75, 0.85, 0.5]
        widths = [w * inch for w in col_widths_in]

        s = self._styles
        tbl_data = []
        for ri, row in enumerate(rows):
            if ri == 0:
                tbl_data.append([_p(str(c), s["BoxTitle"]) for c in row])
            elif ri >= n_data_rows - 4:
                # footer meta rows — span or italic
                tbl_data.append([_p(str(c), s["TableNote"]) for c in row])
            else:
                var_label = row[0]
                is_key = key_vars and var_label in key_vars
                var_style = s["BoxTitle"] if is_key else s["BodyPara"]
                row_cells = [_p(var_label, var_style)]
                row_cells += [_p(str(c), s["BodyPara"]) for c in row[1:]]
                tbl_data.append(row_cells)

        tbl = Table(tbl_data, colWidths=widths, repeatRows=1)
        style_cmds = _base_table_style(n_data_rows, 6, font_size=8.5)
        # Right-align numeric columns
        for col in range(1, 6):
            style_cmds.add("ALIGN", (col, 1), (col, -1), "RIGHT")
        # Shade key variable rows
        if key_vars:
            for ri, row in enumerate(rows[1:], start=1):
                if row[0] in key_vars:
                    style_cmds.add("BACKGROUND", (0, ri), (-1, ri), C_INTERP_BG)
        # Footer styling
        footer_start = n_data_rows - 4
        style_cmds.add("BACKGROUND", (0, footer_start), (-1, -1), C_ROW_ALT)
        style_cmds.add("SPAN", (0, footer_start + 1), (-1, footer_start + 1))
        style_cmds.add("SPAN", (0, footer_start + 2), (2, footer_start + 2))
        style_cmds.add("SPAN", (0, footer_start + 3), (-1, footer_start + 3))

        tbl.setStyle(style_cmds)
        self.story.append(KeepTogether([tbl]))
        full_note = "*** p<0.01  ** p<0.05  * p<0.10. " + note
        self.story.append(_p(full_note, s["TableNote"]))
        self.story.append(Spacer(1, 0.12 * inch))

    # ──────────────────────────────────────────────────────────
    # BOOTSTRAP TABLE
    # ──────────────────────────────────────────────────────────

    def add_bootstrap_table(
        self,
        eq_label: str,
        beta: float,
        se_cl: float,
        t_stat: float,
        p_boot: float,
        ci_lo: float,
        ci_hi: float,
        n_valid: int,
        note: str = "",
    ) -> None:
        """Render a compact wild-cluster bootstrap results row."""
        headers = ["Ecuación", "β (Full sample)", "SE (CL)", "t", "p (boot)", "CI 95% (boot)", "Réplicas"]
        row = [
            eq_label,
            _fmt(beta, 4),
            _fmt(se_cl, 4),
            _fmt(t_stat, 3),
            _fmt(p_boot, 4),
            f"[{_fmt(ci_lo, 4)}, {_fmt(ci_hi, 4)}]",
            str(n_valid),
        ]
        self.add_table([headers, row], note=note)

    # ──────────────────────────────────────────────────────────
    # DIAGNOSTICS TABLE
    # ──────────────────────────────────────────────────────────

    def add_diagnostics_table(
        self,
        rows: List[Tuple[str, str, str, str]],
        note: str = "",
    ) -> None:
        """
        Render a test-results table.

        Parameters
        ----------
        rows : list of (Test name, Statistic, p-value, Interpretation)
        """
        headers = ["Test", "Estadístico", "p-valor", "Interpretación"]
        data = [headers] + [list(r) for r in rows]
        col_widths_in = [2.2, 1.2, 1.0, 2.8]
        self.add_table(data, col_widths=col_widths_in, note=note)

    # ──────────────────────────────────────────────────────────
    # ML / FEATURE IMPORTANCE TABLE
    # ──────────────────────────────────────────────────────────

    def add_ml_table(
        self,
        rows: List[Tuple],
        metric_label: str = "Importancia",
        note: str = "",
    ) -> None:
        """
        Render feature importance / ML results table.

        Parameters
        ----------
        rows : list of (Variable, Gini Importance, Permutation Imp., SHAP Imp., Rank SHAP)
        """
        headers = ["Variable", "Gini Imp.", "Perm. Imp.", "SHAP Imp.", "Rango SHAP"]
        data = [headers] + [list(r) for r in rows]
        col_widths_in = [2.5, 1.1, 1.1, 1.1, 1.2]
        self.add_table(data, col_widths=col_widths_in, note=note)

    # ──────────────────────────────────────────────────────────
    # INTERPRETATION BOX
    # ──────────────────────────────────────────────────────────

    def add_interpretation_box(
        self,
        title: str,
        body: str,
        style: str = "info",
    ) -> None:
        """
        Render a highlighted callout / interpretation box.

        Parameters
        ----------
        title : Bold label (e.g., 'Interpretación', 'Nota metodológica').
        body  : Body text.
        style : 'info' (blue), 'warning' (amber), or 'neutral'.
        """
        bg = {
            "info":    C_INTERP_BG,
            "warning": C_WARN_BG,
            "neutral": C_ROW_ALT,
        }.get(style, C_INTERP_BG)

        border_color = {
            "info":    C_ACCENT,
            "warning": colors.HexColor("#D97706"),
            "neutral": C_GRID,
        }.get(style, C_ACCENT)

        s = self._styles
        inner = [
            [_p(f"▶ {title}", s["BoxTitle"])],
            [_p(body, s["BoxBody"])],
        ]
        inner_tbl = Table(inner, colWidths=[6.8 * inch])
        inner_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, -1), bg),
            ("LEFTPADDING",  (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING",   (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 7),
            ("LINEABOVE",    (0, 0), (-1, 0),  0, colors.transparent),
            ("LINEBEFORE",   (0, 0), (0, -1),  3, border_color),
            ("BOX",          (0, 0), (-1, -1), 0.5, border_color),
        ]))
        self.story.append(KeepTogether([inner_tbl]))
        self.story.append(Spacer(1, 0.1 * inch))

    # ──────────────────────────────────────────────────────────
    # FIGURES / IMAGES
    # ──────────────────────────────────────────────────────────

    def add_image(
        self,
        source: Union[str, Path, io.BytesIO],
        caption: str = "",
        label: str = "",
        width_inch: float = 6.8,
        height_inch: float = 4.4,
    ) -> None:
        """
        Insert a figure with an optional label and caption.

        Parameters
        ----------
        source     : File path or BytesIO buffer.
        caption    : Figure caption text (italicised, centred).
        label      : Short figure label, e.g., 'Figura 3'.
        width_inch : Display width in inches.
        height_inch: Display height in inches.
        """
        if isinstance(source, Path):
            source = str(source)
        if isinstance(source, str) and not Path(source).exists():
            return
        s = self._styles
        elements: List[Any] = []
        if label:
            elements.append(_p(label, s["FigureLabel"]))
        img = Image(source, width=width_inch * inch, height=height_inch * inch)
        img.hAlign = "CENTER"
        elements.append(img)
        if caption:
            elements.append(_p(caption, s["Caption"]))
        elements.append(Spacer(1, 0.08 * inch))
        self.story.append(KeepTogether(elements))

    def add_figure_from_path(self, path: Union[Path, str], caption: str = "") -> None:
        """Convenience wrapper for run_pipeline.py compatibility."""
        self.add_image(path, caption=caption)

    def add_figure(self, fig: Figure, caption: str = "") -> None:
        """Add a Matplotlib Figure object directly."""
        if fig is None:
            return
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        self.add_image(buf, caption=caption)

    # ──────────────────────────────────────────────────────────
    # PAGE BREAK / SPACER
    # ──────────────────────────────────────────────────────────

    def add_page_break(self) -> None:
        self.story.append(PageBreak())

    def add_spacer(self, height_inch: float = 0.15) -> None:
        self.story.append(Spacer(1, height_inch * inch))

    # ──────────────────────────────────────────────────────────
    # APPENDIX (console log)
    # ──────────────────────────────────────────────────────────

    def add_annex(self) -> None:
        """Append a compact appendix containing the raw console log."""
        self.story.append(PageBreak())
        self.add_section_h1("Apéndice — Log de Consola", register_toc=True)
        self.add_body_text(
            "El siguiente es el registro textual completo generado por el pipeline. "
            "Contiene todos los mensajes de consola, advertencias y métricas intermedias."
        )
        self.story.append(Spacer(1, 0.1 * inch))

        # Present console log in a styled single-column table
        log_text = "\n".join(self._console_lines)
        # Chunk into manageable blocks
        lines = self._console_lines
        chunk_size = 60
        for i in range(0, len(lines), chunk_size):
            block = lines[i:i + chunk_size]
            rows = [[html.escape(ln or " ", quote=False)] for ln in block]
            tbl_data = [[_p(r[0], self._styles["TableNote"])] for r in rows]
            tbl = Table(tbl_data, colWidths=[7.2 * inch])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                ("FONTNAME",   (0, 0), (-1, -1), "Courier"),
                ("FONTSIZE",   (0, 0), (-1, -1), 7.5),
                ("LEFTPADDING",(0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING",(0,0),(-1,-1),  1),
                ("BOX",        (0, 0), (-1, -1), 0.5, C_GRID),
            ]))
            self.story.append(tbl)
            self.story.append(Spacer(1, 0.02 * inch))

    # ──────────────────────────────────────────────────────────
    # HEADER / FOOTER CANVAS CALLBACKS
    # ──────────────────────────────────────────────────────────

    def _draw_cover_page(self, canvas: Any, doc: Any) -> None:
        """No header/footer on cover page."""
        pass

    def _draw_header_footer(self, canvas: Any, doc: Any) -> None:
        canvas.saveState()

        # ── Header ──
        canvas.setFillColor(C_NAVY)
        canvas.rect(
            0.6 * inch, 10.35 * inch,
            7.8 * inch, 0.02 * inch,
            fill=1, stroke=0,
        )
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(C_SLATE)
        # Left: short title
        short = self.title[:60] + ("…" if len(self.title) > 60 else "")
        canvas.drawString(0.65 * inch, 10.42 * inch, short)
        # Right: date
        canvas.drawRightString(
            8.4 * inch, 10.42 * inch,
            datetime.now().strftime("%Y-%m-%d"),
        )

        # ── Footer ──
        canvas.setFillColor(C_NAVY)
        canvas.rect(
            0.6 * inch, 0.6 * inch,
            7.8 * inch, 0.02 * inch,
            fill=1, stroke=0,
        )
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(C_SLATE)
        canvas.drawString(0.65 * inch, 0.45 * inch,
                          "Econometric Pipeline — Confidential")
        canvas.drawRightString(8.4 * inch, 0.45 * inch,
                               f"Página {doc.page}")

        canvas.restoreState()

    # ──────────────────────────────────────────────────────────
    # BUILD PDF
    # ──────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        self.flush_pending()
        if path is not None:
            self.output_path = Path(path)
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.story:
            self.add_cover()

        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=letter,
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
            topMargin=0.85 * inch,
            bottomMargin=0.75 * inch,
            title=self.title,
            author=self.author,
            subject="Econometric Research Report",
        )

        # Page callbacks: skip header/footer on cover (page 1)
        def on_page(canvas: Any, doc: Any) -> None:
            if doc.page == 1:
                self._draw_cover_page(canvas, doc)
            else:
                self._draw_header_footer(canvas, doc)

        doc.build(self.story, onFirstPage=on_page, onLaterPages=on_page)
        return self.output_path


# ──────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON HELPERS
# ──────────────────────────────────────────────────────────────────────────────

_active_report: Optional[ResearchReport] = None


def set_active_report(report: Optional[ResearchReport]) -> None:
    global _active_report
    _active_report = report


def get_active_report() -> Optional[ResearchReport]:
    return _active_report


def enable_reporting(report: ResearchReport) -> None:
    set_active_report(report)
    report.start_capture()


def disable_reporting(report: ResearchReport) -> None:
    report.stop_capture()
    set_active_report(None)
