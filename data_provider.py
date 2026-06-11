"""Robust yfinance price fetching with retry + delisting-aware classification.

`yf.download` intermittently returns empty/all-NaN columns for perfectly healthy
tickers (AAPL, V, MCD, ...) when Yahoo rate-limits a burst of requests — e.g. the
~100 back-to-back downloads of a full experiment run. yfinance prints
``$TICKER: possibly delisted; no price data found`` for *any* fetch failure, so a
throttled-but-healthy ticker is indistinguishable from a genuinely delisted one
(only WBA, taken private in 2025, is truly dead in this project's universe).

`fetch_close_prices` fixes this by (1) retrying with exponential backoff so
transient throttling recovers, and (2) classifying tickers that never recover as
``delisted`` (a hard 404 / "no timezone found" signal) vs ``unavailable``
(transient / unknown). The retry-survival behavior is the primary guarantee and
is independent of log parsing: a recoverable ticker is never dropped, so the worst
failure mode of a yfinance log-format change is a dead ticker labeled
``unavailable`` instead of ``delisted``.

Reliability notes (verified against yfinance 1.4.1):
- ``yfinance.shared._ERRORS`` / ``_TRACEBACKS`` are cleared mid-call and read back
  empty, so they cannot be used for post-hoc classification.
- The reliable per-ticker failure signal is the ``yfinance`` logger at WARNING
  level (the raw ``HTTP Error 404 ... "Quote not found for symbol: WBA"`` line plus
  a grouped ``['WBA']: no timezone found`` summary). DEBUG is avoided because it
  disables yfinance's download threading.
- Single-ticker downloads also yield MultiIndex columns, so ``data["Close"]`` is
  uniformly a ticker-keyed frame; the Series shape is still guarded defensively.
"""

import io
import logging
import re
import time
from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass
class FetchResult:
    """Outcome of a (retried) price fetch.

    ``close`` is cleaned the same way the legacy call sites cleaned it: columns
    that are entirely NaN, or whose most recent row is NaN, are dropped.
    """

    close: pd.DataFrame
    requested: list
    delisted: list      # confirmed dead (hard 404 / "no timezone found")
    unavailable: list   # still missing after retries, with no hard-404 signal

    @property
    def valid(self):
        return list(self.close.columns)

    @property
    def dropped(self):
        return sorted(set(self.requested) - set(self.valid))


# Grouped failure summary line, e.g. "['WBA']: no timezone found" or
# "['AAA', 'BBB']: no price data found (period=...)". yfinance has already
# stripped the leading "$SYM: " prefix at this point.
_GROUP_RE = re.compile(r"^\[(?P<syms>.*?)\]:\s*(?P<rationale>.+)$")
_NOTFOUND_RE = re.compile(r"Quote not found for symbol:\s*([A-Za-z0-9.\-]+)")


def _extract_close(data, requested):
    """Return a ticker-keyed ``Close`` DataFrame from a ``yf.download`` result.

    Handles both the MultiIndex-column case (single or multi ticker) and the
    degenerate single-level / Series shapes defensively.
    """
    if data is None or len(data) == 0:
        return pd.DataFrame()
    cols = data.columns
    if isinstance(cols, pd.MultiIndex):
        if "Close" not in cols.get_level_values(0):
            return pd.DataFrame()
        close = data["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame()
    else:
        if "Close" not in cols:
            return pd.DataFrame()
        close = data["Close"]
        if isinstance(close, pd.Series):
            name = requested[0] if len(requested) == 1 else (close.name or "Close")
            close = close.to_frame(name=name)
    return close


def _download_capturing_log(tickers, start, end, *, auto_adjust, progress):
    """Run ``yf.download`` while capturing the ``yfinance`` logger output.

    Returns ``(log_text, data)``. A temporary WARNING-level ``StreamHandler`` is
    attached (and removed in ``finally``) so per-ticker failure rationales can be
    parsed without enabling DEBUG (which would disable download threading).
    """
    logger = logging.getLogger("yfinance")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    prev_level = logger.level
    logger.addHandler(handler)
    try:
        if logger.level == logging.NOTSET or logger.level > logging.WARNING:
            logger.setLevel(logging.WARNING)
        data = yf.download(tickers, start=start, end=end,
                           auto_adjust=auto_adjust, progress=progress)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
    return stream.getvalue(), data


def _parse_failure_log(text):
    """Map ticker -> failure rationale string parsed from captured log text."""
    rationale = {}
    for sym in _NOTFOUND_RE.findall(text):
        rationale[sym] = "404 not found"
    for line in text.splitlines():
        m = _GROUP_RE.match(line.strip())
        if not m:
            continue
        reason = m.group("rationale").strip()
        for sym in (s.strip().strip("'\"") for s in m.group("syms").split(",")):
            if sym:
                rationale.setdefault(sym, reason)
    return rationale


def _is_hard_delisting(rationale):
    """True if the rationale indicates a genuine dead ticker (vs transient)."""
    r = rationale.lower()
    return ("404" in r) or ("not found" in r) or ("no timezone" in r)


def fetch_close_prices(stocks, start, end, *, auto_adjust=False, progress=True,
                       max_retries=4, base_delay=1.0):
    """Download closing prices, retrying transient (rate-limit) failures.

    Tickers that recover on any retry are returned in ``close``; tickers that
    never recover are classified as ``delisted`` (hard 404 / "no timezone found")
    or ``unavailable`` (transient / unknown). ``auto_adjust`` and ``progress`` are
    forwarded verbatim to every internal ``yf.download`` so per-call-site numeric
    behavior is preserved.
    """
    requested = [str(s) for s in ([stocks] if isinstance(stocks, str) else stocks)]
    pending = list(requested)
    acc = {}                 # ticker -> recovered Close Series
    last_rationale = {}

    for attempt in range(max_retries):
        if not pending:
            break
        log_text, data = _download_capturing_log(
            pending, start, end,
            auto_adjust=auto_adjust, progress=progress and attempt == 0)
        close = _extract_close(data, pending)
        for tkr in list(pending):
            if tkr in close.columns and close[tkr].notna().any():
                acc[tkr] = close[tkr]
                pending.remove(tkr)
        last_rationale.update(_parse_failure_log(log_text))
        if not pending:
            break
        if attempt < max_retries - 1:
            time.sleep(base_delay * (2 ** attempt))   # 1, 2, 4, 8, ... seconds

    # Reassemble in the originally requested order, then mirror the legacy
    # cleaning: drop all-NaN columns and columns whose latest price is NaN.
    ordered = [t for t in requested if t in acc]
    close_df = pd.DataFrame({t: acc[t] for t in ordered}) if ordered else pd.DataFrame()
    if not close_df.empty:
        close_df = close_df.dropna(axis=1, how="all")
        if len(close_df) > 0 and close_df.shape[1] > 0:
            close_df = close_df.loc[:, close_df.iloc[-1].notna()]

    valid = set(close_df.columns)
    delisted, unavailable = [], []
    for tkr in requested:
        if tkr in valid:
            continue
        if _is_hard_delisting(last_rationale.get(tkr, "")):
            delisted.append(tkr)
        else:
            unavailable.append(tkr)

    return FetchResult(close=close_df, requested=requested,
                       delisted=sorted(delisted), unavailable=sorted(unavailable))


class PriceCache:
    """Per-run, in-memory cache of cleaned per-ticker Close series.

    Lives for a single command invocation (instantiate it at the start of a run
    and let it go out of scope when the run ends — no disk, no cleanup). Each
    distinct ticker is downloaded at most once: ``get_close`` fetches only the
    tickers it has not seen, in one batched ``fetch_close_prices`` call, then
    reassembles the requested subset from cached columns.

    Caching raw per-ticker series is numerically identical to a per-subset
    ``fetch_close_prices(...).close`` within one run: a ticker's series is
    deterministic for a fixed (start, end, auto_adjust), and the per-column
    cleaning that ``fetch_close_prices`` applies (drop all-NaN / last-NaN
    columns) is per-ticker, so it does not depend on which other tickers shared
    the download. The subset-level ``dropna(how="any")`` still happens in the
    caller, so derived moments are unchanged.
    """

    def __init__(self):
        self._series = {}      # (ticker, start, end, auto_adjust) -> pd.Series
        self._attempted = set()  # keys we've tried (incl. failed downloads)

    def get_close(self, stocks, start, end, *, auto_adjust=False):
        requested = [str(s) for s in ([stocks] if isinstance(stocks, str) else stocks)]
        missing = [
            t for t in requested
            if (t, start, end, auto_adjust) not in self._series
            and (t, start, end, auto_adjust) not in self._attempted
        ]
        if missing:
            fetched = fetch_close_prices(
                missing, start=start, end=end,
                auto_adjust=auto_adjust, progress=False).close
            for tkr in fetched.columns:
                self._series[(str(tkr), start, end, auto_adjust)] = fetched[tkr]
            # Mark every attempted ticker so a perpetually-missing one (delisted /
            # unavailable) is not re-downloaded on every later problem.
            for t in missing:
                self._attempted.add((t, start, end, auto_adjust))

        cols = {t: self._series[(t, start, end, auto_adjust)]
                for t in requested if (t, start, end, auto_adjust) in self._series}
        if not cols:
            return pd.DataFrame()
        # Preserve requested order; align on the shared date index.
        return pd.DataFrame(cols)[[t for t in requested if t in cols]]
