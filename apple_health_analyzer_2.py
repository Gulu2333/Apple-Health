#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================
  Apple Health Sleep & Physiology Analyzer
  All-in-One Edition
============================================================
  A single script that reads your Apple Health export.xml
  and produces a complete set of academic-grade (APA 7th ed.)
  figures and a descriptive text report.

  How to use:
    1. Open a terminal / command prompt.
    2. Run:   python apple_health_analyzer.py
    3. When prompted, drag-and-drop your export.xml file
       into the terminal window and press Enter.
    4. Wait for the analysis to finish.
    5. Open the "output" folder next to this script to see
       all figures and the report.

  Requirements:
    pip install pandas matplotlib seaborn scipy numpy
============================================================
"""

# ================================================================
# IMPORTS
# ================================================================
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ================================================================
# CONFIGURATION
# ================================================================
OUTPUT_DIR = Path("output")
DPI = 300

SLEEP_GAP_HOURS = 3
MIN_SLEEP_HOURS = 3
HR_HRV_TOLERANCE_MIN = 5
BIN_SIZE_HOURS = 0.5
TAIL_CUTOFF_FRAC = 0.05
SMOOTH_WINDOW = 3
TRANSITION_WINDOW_MIN = 10

STAGE_MAP = {
    "HKCategoryValueSleepAnalysisAsleepCore":       "Core",
    "HKCategoryValueSleepAnalysisAsleepDeep":       "Deep",
    "HKCategoryValueSleepAnalysisAsleepREM":        "REM",
    "HKCategoryValueSleepAnalysisAsleepUnspecified": "Core",
    "HKCategoryValueSleepAnalysisAsleep":           "Core",
    "HKCategoryValueSleepAnalysisAwake":            "Awake",
}
SLEEP_ASLEEP_VALUES = set(STAGE_MAP.keys()) - {"HKCategoryValueSleepAnalysisAwake"}

STAGE_ORDER = ["Awake", "REM", "Core", "Deep"]
STAGE_COLORS = {
    "Awake": "#BFBFBF",
    "REM":   "#808080",
    "Core":  "#4D4D4D",
    "Deep":  "#1A1A1A",
}

# Global figure counter
_fig_counter = 0


# ================================================================
# HELPERS
# ================================================================
def safe_print(text):
    """Print with fallback for non-UTF-8 terminals (e.g. Windows GBK)."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", errors="replace").decode("ascii"))


def next_fig_number():
    global _fig_counter
    _fig_counter += 1
    return _fig_counter


def banner(msg):
    width = 60
    safe_print("")
    safe_print("=" * width)
    safe_print(f"  {msg}")
    safe_print("=" * width)


def step(num, total, msg):
    bar_len = 30
    filled = int(bar_len * num / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    pct = int(100 * num / total)
    safe_print(f"\n  [{bar}] Step {num}/{total} ({pct}%)")
    safe_print(f"  {msg}")
    safe_print("")


def substep(msg):
    safe_print(f"    ... {msg}")


def done(path):
    safe_print(f"    --> Saved: {path}")


def safe_input(prompt):
    """input() with GBK-safe prompt and graceful cancel."""
    try:
        safe_print(prompt)
        return input("  >> ").strip()
    except (EOFError, KeyboardInterrupt):
        safe_print("\n  Cancelled by user.")
        sys.exit(0)


# ================================================================
# APA STYLE
# ================================================================
def apply_apa_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
        "figure.dpi":        DPI,
        "savefig.dpi":       DPI,
        "savefig.bbox":      "tight",
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
    })


# ================================================================
# 1. USER INPUT
# ================================================================
def get_xml_path():
    banner("Apple Health Sleep & Physiology Analyzer")
    safe_print("")
    safe_print("  This program analyses your Apple Health data and")
    safe_print("  generates publication-ready figures and a report.")
    safe_print("")
    safe_print("  HOW TO PROVIDE YOUR FILE:")
    safe_print("    - Drag and drop your 'export.xml' file into this")
    safe_print("      window, then press Enter.")
    safe_print("    - Or type the full file path and press Enter.")
    safe_print("")

    while True:
        try:
            raw = input("  >> Path to export.xml: ").strip()
        except (EOFError, KeyboardInterrupt):
            safe_print("\n  Cancelled by user.")
            sys.exit(0)

        # Remove surrounding quotes (Windows drag-and-drop adds them)
        path = raw.strip("\"'")

        if not path:
            safe_print("  [!] No path entered. Please try again.")
            continue
        if not os.path.isfile(path):
            safe_print(f"  [!] File not found: {path}")
            safe_print("      Please check the path and try again.")
            continue
        if not path.lower().endswith(".xml"):
            safe_print("  [!] The file does not appear to be an XML file.")
            safe_print("      Please provide the 'export.xml' from Apple Health.")
            continue

        size_mb = os.path.getsize(path) / (1024 * 1024)
        safe_print(f"\n  File found: {os.path.basename(path)} ({size_mb:.1f} MB)")
        safe_print("  Starting analysis...\n")
        return path


# ================================================================
# 1b. USER OPTIONS (date range, hypnogram night)
# ================================================================
def ask_date_range(df_hr, df_hrv, df_spo2, df_stages):
    """Let the user optionally restrict the analysis to a date window.
    Returns (start_date, end_date) as tz-aware Timestamps, or (None, None)
    to keep everything."""
    # Find the overall date extent
    all_ts = []
    for df in (df_hr, df_hrv, df_spo2):
        if not df.empty:
            all_ts.extend([df["timestamp"].min(), df["timestamp"].max()])
    if not df_stages.empty:
        all_ts.extend([df_stages["start"].min(), df_stages["end"].max()])
    if not all_ts:
        return None, None

    earliest = min(all_ts)
    latest = max(all_ts)
    safe_print("")
    safe_print("  " + "-" * 56)
    safe_print("  DATE RANGE SELECTION")
    safe_print("  " + "-" * 56)
    safe_print(f"  Your data spans: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
    safe_print(f"  ({(latest - earliest).days} days)")
    safe_print("")
    safe_print("  Would you like to analyse only a specific date range?")
    safe_print("  Press Enter to use ALL data, or type a date range.")
    safe_print("  Format: YYYY-MM-DD YYYY-MM-DD  (start end)")
    safe_print("  Example: 2025-01-01 2025-06-30")
    safe_print("")

    raw = safe_input("  Date range (or Enter for all):")
    if not raw:
        safe_print("  -> Using all available data.")
        return None, None

    parts = raw.replace(",", " ").replace("/", "-").split()
    if len(parts) < 2:
        safe_print("  [!] Could not parse two dates. Using all data.")
        return None, None

    try:
        d_start = pd.Timestamp(parts[0], tz="UTC")
        d_end = pd.Timestamp(parts[1], tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    except Exception:
        safe_print("  [!] Invalid date format. Using all data.")
        return None, None

    safe_print(f"  -> Filtering to: {d_start.strftime('%Y-%m-%d')} through {d_end.strftime('%Y-%m-%d')}")
    return d_start, d_end


def apply_date_filter(df, col, d_start, d_end):
    """Filter a DataFrame to [d_start, d_end] on the given column."""
    if d_start is None:
        return df
    mask = (df[col] >= d_start) & (df[col] <= d_end)
    return df[mask].copy().reset_index(drop=True)


def ask_hypnogram_config(sessions_valid):
    """Let the user choose which night(s) to display in the hypnogram.
    Returns a dict with keys:
      'night_ids' : list of night_id values (or None for auto)
      'count'     : how many nights to draw
    """
    if sessions_valid.empty:
        return {"night_ids": None, "count": 1}

    safe_print("")
    safe_print("  " + "-" * 56)
    safe_print("  SINGLE-NIGHT HYPNOGRAM SETTINGS")
    safe_print("  " + "-" * 56)
    safe_print(f"  You have {len(sessions_valid)} valid sleep nights.")

    # Show a summary of available nights
    safe_print("")
    safe_print("  Available nights (showing first 20):")
    safe_print(f"    {'#':>4}  {'Date':>12}  {'Duration':>10}  Night ID")
    shown = 0
    for idx, row in sessions_valid.iterrows():
        if shown >= 20:
            safe_print(f"    ... and {len(sessions_valid) - 20} more.")
            break
        d_str = row["sleep_onset"].strftime("%Y-%m-%d")
        dur = f"{row['duration_hours']:.1f}h"
        safe_print(f"    {shown+1:>4}  {d_str:>12}  {dur:>10}  ({idx})")
        shown += 1

    safe_print("")
    safe_print("  How many nights to show in the hypnogram figure?")
    safe_print("  Press Enter for 1 (auto-pick the most detailed night).")
    safe_print("  Or type a number (e.g. 3) to show multiple nights.")
    safe_print("")

    raw_count = safe_input("  Number of nights (or Enter for 1):")
    count = 1
    if raw_count:
        try:
            count = max(1, min(int(raw_count), len(sessions_valid)))
        except ValueError:
            safe_print("  [!] Invalid number. Using 1.")
            count = 1

    safe_print("")
    safe_print("  Which night(s)? Options:")
    safe_print("    - Press Enter to auto-select (most data points)")
    safe_print("    - Type a date: 2025-06-15")
    safe_print("    - Type a date range: 2025-06-10 2025-06-20")
    safe_print("      (picks the best night(s) within that range)")
    safe_print("")

    raw_sel = safe_input("  Night selection (or Enter for auto):")
    if not raw_sel:
        safe_print(f"  -> Auto-selecting {count} night(s) with most data.")
        return {"night_ids": None, "count": count}

    # Try to parse as date or date range
    parts = raw_sel.replace(",", " ").replace("/", "-").split()
    try:
        sel_start = pd.Timestamp(parts[0], tz="UTC")
        if len(parts) >= 2:
            sel_end = pd.Timestamp(parts[1], tz="UTC") + pd.Timedelta(days=1)
        else:
            sel_end = sel_start + pd.Timedelta(days=1)
    except Exception:
        safe_print("  [!] Could not parse date. Using auto-select.")
        return {"night_ids": None, "count": count}

    # Find nights whose onset falls within the range
    mask = ((sessions_valid["sleep_onset"] >= sel_start) &
            (sessions_valid["sleep_onset"] < sel_end))
    candidates = sessions_valid[mask]
    if candidates.empty:
        safe_print("  [!] No nights found in that range. Using auto-select.")
        return {"night_ids": None, "count": count}

    selected_ids = candidates.index.tolist()[:count]
    safe_print(f"  -> Selected {len(selected_ids)} night(s) in range.")
    return {"night_ids": selected_ids, "count": count}


# ================================================================
# 2. STREAMING XML PARSER
# ================================================================
def parse_xml(xml_path):
    """Stream-parse Apple Health export.xml."""
    stages_raw = []
    hr_records = []
    hrv_records = []
    spo2_records = []

    date_fmt = "%Y-%m-%d %H:%M:%S %z"
    count = 0
    t0 = time.time()

    for event, elem in ET.iterparse(xml_path, events=("end",)):
        if elem.tag == "Record":
            rtype = elem.get("type", "")

            if rtype == "HKCategoryTypeIdentifierSleepAnalysis":
                cat = elem.get("value", "")
                stage = STAGE_MAP.get(cat)
                if stage is not None:
                    try:
                        s = datetime.strptime(elem.get("startDate"), date_fmt)
                        e = datetime.strptime(elem.get("endDate"), date_fmt)
                        stages_raw.append({"start": s, "end": e, "stage": stage})
                    except (TypeError, ValueError):
                        pass

            elif rtype == "HKQuantityTypeIdentifierHeartRate":
                try:
                    ts = datetime.strptime(elem.get("startDate"), date_fmt)
                    val = float(elem.get("value"))
                    hr_records.append({"timestamp": ts, "value": val})
                except (TypeError, ValueError):
                    pass

            elif rtype == "HKQuantityTypeIdentifierHeartRateVariabilitySDNN":
                try:
                    ts = datetime.strptime(elem.get("startDate"), date_fmt)
                    val = float(elem.get("value"))
                    hrv_records.append({"timestamp": ts, "value": val})
                except (TypeError, ValueError):
                    pass

            elif rtype == "HKQuantityTypeIdentifierOxygenSaturation":
                try:
                    ts = datetime.strptime(elem.get("startDate"), date_fmt)
                    val = float(elem.get("value")) * 100.0
                    spo2_records.append({"timestamp": ts, "value": val})
                except (TypeError, ValueError):
                    pass

            count += 1
            if count % 500_000 == 0:
                elapsed = time.time() - t0
                substep(f"Parsed {count:,} XML elements ({elapsed:.0f}s elapsed)")

        elem.clear()

    elapsed = time.time() - t0
    substep(f"Parsing complete in {elapsed:.0f} seconds.")
    substep(f"Heart Rate     : {len(hr_records):>9,} records")
    substep(f"HRV (SDNN)     : {len(hrv_records):>9,} records")
    substep(f"SpO2           : {len(spo2_records):>9,} records")
    substep(f"Sleep stages   : {len(stages_raw):>9,} records")

    return {
        "stages": stages_raw,
        "hr": hr_records,
        "hrv": hrv_records,
        "spo2": spo2_records,
    }


# ================================================================
# 3. BUILD DATAFRAMES
# ================================================================
def to_df(records):
    if not records:
        return pd.DataFrame(columns=["timestamp", "value"])
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def build_stage_df(stages_raw):
    df = pd.DataFrame(stages_raw)
    if df.empty:
        return df
    for col in ("start", "end"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df.sort_values("start", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["duration_min"] = (df["end"] - df["start"]).dt.total_seconds() / 60
    return df


# ================================================================
# 4. MERGE SLEEP NIGHTS
# ================================================================
def merge_sleep_nights(df_stages):
    if df_stages.empty:
        return df_stages, pd.DataFrame()

    gap = timedelta(hours=SLEEP_GAP_HOURS)
    night_id = 0
    ids = [0]

    for i in range(1, len(df_stages)):
        if df_stages.iloc[i]["start"] - df_stages.iloc[i-1]["end"] > gap:
            night_id += 1
        ids.append(night_id)

    df_stages = df_stages.copy()
    df_stages["night_id"] = ids

    onset_map = df_stages.groupby("night_id")["start"].min()
    wake_map = df_stages.groupby("night_id")["end"].max()
    df_stages["sleep_onset"] = df_stages["night_id"].map(onset_map)

    # Session summary
    sessions = pd.DataFrame({
        "sleep_onset": onset_map,
        "wake_time": wake_map,
    })
    sessions["duration_hours"] = \
        (sessions["wake_time"] - sessions["sleep_onset"]).dt.total_seconds() / 3600

    n_all = len(sessions)
    valid_ids = sessions[sessions["duration_hours"] >= MIN_SLEEP_HOURS].index
    sessions_valid = sessions.loc[valid_ids]
    df_stages = df_stages[df_stages["night_id"].isin(valid_ids)].copy()

    substep(f"{n_all} total nights found, "
            f"{len(sessions_valid)} valid (>= {MIN_SLEEP_HOURS}h), "
            f"{n_all - len(sessions_valid)} naps discarded.")

    return df_stages, sessions_valid


# ================================================================
# 5. MAP PHYSIOLOGY TO STAGES / SLEEP TIME
# ================================================================
def map_to_stages(df_physio, df_stages, col_name):
    if df_physio.empty or df_stages.empty:
        return pd.DataFrame()
    results = []
    for nid, grp in df_stages.groupby("night_id"):
        nstart, nend = grp["start"].min(), grp["end"].max()
        onset = grp["sleep_onset"].iloc[0]
        mask = (df_physio["timestamp"] >= nstart) & (df_physio["timestamp"] <= nend)
        chunk = df_physio.loc[mask]
        if chunk.empty:
            continue
        for _, frag in grp.iterrows():
            fm = (chunk["timestamp"] >= frag["start"]) & (chunk["timestamp"] < frag["end"])
            m = chunk.loc[fm].copy()
            if m.empty:
                continue
            m["stage"] = frag["stage"]
            m["night_id"] = nid
            m["sleep_onset"] = onset
            m["hours_since_onset"] = (m["timestamp"] - onset).dt.total_seconds() / 3600
            results.append(m[["timestamp", "value", "stage", "night_id",
                              "sleep_onset", "hours_since_onset"]])
    if not results:
        return pd.DataFrame()
    out = pd.concat(results, ignore_index=True)
    out.rename(columns={"value": col_name}, inplace=True)
    return out


def map_to_sleep_time(df_physio, sessions, col_name):
    if df_physio.empty or sessions.empty:
        return pd.DataFrame()
    results = []
    for _, sess in sessions.iterrows():
        onset, wake = sess["sleep_onset"], sess["wake_time"]
        mask = (df_physio["timestamp"] >= onset) & (df_physio["timestamp"] <= wake)
        chunk = df_physio.loc[mask].copy()
        if chunk.empty:
            continue
        chunk["hours_since_onset"] = (chunk["timestamp"] - onset).dt.total_seconds() / 3600
        results.append(chunk[["hours_since_onset", "value"]].rename(
            columns={"value": col_name}))
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def align_hr_hrv(df_hrv, df_hr):
    if df_hrv.empty or df_hr.empty:
        m = df_hrv.rename(columns={"value": "hrv"}).copy()
        m["hr"] = np.nan
        return m
    hrv = df_hrv.rename(columns={"value": "hrv"}).copy()
    hr = df_hr.rename(columns={"value": "hr"}).copy()
    merged = pd.merge_asof(hrv, hr, on="timestamp",
                           tolerance=pd.Timedelta(minutes=HR_HRV_TOLERANCE_MIN),
                           direction="nearest")
    return merged


# ================================================================
# 6. BINNING & SMOOTHING
# ================================================================
def bin_and_smooth(df, value_col, max_hours=8.0):
    if df.empty:
        return pd.DataFrame(columns=["bin_center", "mean", "sem"])
    df = df[(df["hours_since_onset"] >= 0) & (df["hours_since_onset"] <= max_hours)].copy()
    df["bin"] = (df["hours_since_onset"] // BIN_SIZE_HOURS) * BIN_SIZE_HOURS + BIN_SIZE_HOURS / 2
    agg = df.groupby("bin")[value_col].agg(["mean", "sem", "count"]).reset_index()
    agg.rename(columns={"bin": "bin_center"}, inplace=True)
    peak = agg["count"].max()
    agg = agg[agg["count"] >= peak * TAIL_CUTOFF_FRAC].copy()
    if len(agg) >= SMOOTH_WINDOW:
        agg["mean"] = agg["mean"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
        agg["sem"] = agg["sem"].rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
    return agg[["bin_center", "mean", "sem"]]


# ================================================================
# 7. TRANSITION EXTRACTION
# ================================================================
def extract_transitions(df_stages, df_hr, df_hrv):
    if df_stages.empty:
        return pd.DataFrame()
    window = timedelta(minutes=TRANSITION_WINDOW_MIN)
    records = []
    for nid, grp in df_stages.groupby("night_id"):
        grp = grp.sort_values("start")
        frags = grp.to_dict("records")
        for i in range(len(frags) - 1):
            curr, nxt = frags[i], frags[i+1]
            if curr["stage"] == nxt["stage"]:
                continue
            boundary = curr["end"]
            t0, t1 = boundary - window, boundary + window
            for df_src, metric in [(df_hr, "hr"), (df_hrv, "hrv")]:
                sl = df_src[(df_src["timestamp"] >= t0) & (df_src["timestamp"] <= t1)].copy()
                if sl.empty:
                    continue
                sl["delta_min"] = (sl["timestamp"] - boundary).dt.total_seconds() / 60
                sl["from_stage"] = curr["stage"]
                sl["to_stage"] = nxt["stage"]
                sl["transition"] = f"{curr['stage']} -> {nxt['stage']}"
                sl["metric"] = metric
                sl.rename(columns={"value": "metric_value"}, inplace=True)
                records.append(sl[["delta_min", "from_stage", "to_stage",
                                   "transition", "metric", "metric_value"]])
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# ================================================================
# FIGURES -- Part A: Data Quality (Figures 1-4)
# ================================================================
def fig_record_counts(raw):
    n = next_fig_number()
    labels = ["Heart Rate", "HRV (SDNN)", "SpO2", "Sleep Stages"]
    counts = [len(raw["hr"]), len(raw["hrv"]), len(raw["spo2"]), len(raw["stages"])]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.barh(labels, counts, color="dimgray", edgecolor="white", linewidth=0.5)
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.02, bar.get_y()+bar.get_height()/2,
                f"{c:,}", va="center", fontsize=10)
    ax.set_xlabel("Number of Records")
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    title = f"Figure {n}. Total extracted records by data type."
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fname = f"Figure_{n:02d}_Record_Counts.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname


def fig_temporal_coverage(df_hrv, df_hr, df_spo2):
    n = next_fig_number()
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for ax, (df, label) in zip(axes, [(df_hr, "Heart Rate"), (df_hrv, "HRV (SDNN)"),
                                       (df_spo2, "SpO2")]):
        if df.empty:
            ax.set_ylabel(f"{label}\n(records/day)")
            continue
        daily = df.set_index("timestamp").resample("D").size()
        ax.fill_between(daily.index, 0, daily.values, color="lightgray", linewidth=0)
        ax.plot(daily.index, daily.values, color="black", linewidth=0.8)
        ax.set_ylabel(f"{label}\n(records/day)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(False)
    axes[-1].set_xlabel("Date")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=30)
    title = f"Figure {n}. Daily record density over time, showing data collection continuity."
    axes[0].set_title(title, pad=12)
    fig.tight_layout()
    fname = f"Figure_{n:02d}_Temporal_Coverage.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname


def fig_distributions(df_hrv, df_hr, df_spo2):
    n = next_fig_number()
    datasets = [(df_hrv, "HRV (ms)", "HRV"), (df_hr, "HR (bpm)", "Heart Rate"),
                (df_spo2, "SpO2 (%)", "SpO2")]
    active = [(d,u,t) for d,u,t in datasets if not d.empty]
    if not active:
        return n, "", ""
    fig, axes = plt.subplots(1, len(active), figsize=(5*len(active), 5))
    if len(active) == 1: axes = [axes]
    outlier_info = {}
    for ax, (df, unit, tlabel) in zip(axes, active):
        vals = df["value"].dropna()
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1; lo, hi = q1-1.5*iqr, q3+1.5*iqr
        n_out = ((vals < lo) | (vals > hi)).sum()
        outlier_info[tlabel] = {"N": len(vals), "mean": vals.mean(), "sd": vals.std(),
                                "median": vals.median(), "Q1": q1, "Q3": q3,
                                "lo": lo, "hi": hi, "n_outliers": n_out,
                                "pct": n_out/len(vals)*100 if len(vals)>0 else 0}
        ax.hist(vals, bins=60, color="lightgray", edgecolor="white", linewidth=0.3)
        ax.axvline(vals.median(), color="dimgray", linewidth=1.5, label=f"Median={vals.median():.1f}")
        ax.axvline(lo, color="black", linestyle="--", linewidth=1, label=f"Fence={lo:.1f}")
        ax.axvline(hi, color="black", linestyle="-.", linewidth=1, label=f"Fence={hi:.1f}")
        ax.set_xlabel(unit); ax.set_ylabel("Frequency"); ax.set_title(tlabel)
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.grid(False)
    title = f"Figure {n}. Raw value distributions with IQR outlier fences."
    fig.suptitle(title, fontweight="bold", y=1.03)
    fig.tight_layout()
    fname = f"Figure_{n:02d}_Distributions.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname, outlier_info


def fig_sleep_merge(sessions_all, sessions_valid, n_raw_frags):
    n = next_fig_number()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    if not sessions_all.empty:
        ax.hist(sessions_all["duration_hours"], bins=40, color="lightgray",
                edgecolor="white", linewidth=0.3)
        ax.axvline(MIN_SLEEP_HOURS, color="black", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Session Duration (hours)")
        ax.set_ylabel("Frequency")
        ax.set_title("(a) Merged Session Duration")
        # Place stats text in upper-right, threshold label separately below it
        txt = (f"Raw fragments: {n_raw_frags:,}\n"
               f"Merged sessions: {len(sessions_all):,}\n"
               f"Valid (>={MIN_SLEEP_HOURS}h): {len(sessions_valid):,}\n"
               f"Threshold: {MIN_SLEEP_HOURS}h (dashed line)")
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, fontsize=8,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray",
                          alpha=0.95))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    ax = axes[1]
    if not sessions_valid.empty:
        hours = sessions_valid["sleep_onset"].dt.hour + sessions_valid["sleep_onset"].dt.minute/60
        display = hours.apply(lambda h: h - 24 if h > 18 else h)
        ax.hist(display, bins=30, color="lightgray", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("Sleep Onset Hour\n(centered on midnight)")
        ax.set_ylabel("Frequency")
        ax.set_title("(b) Sleep Onset Distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    title = f"Figure {n}. Sleep session merge diagnostics."
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fname = f"Figure_{n:02d}_Sleep_Merge.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname


# ================================================================
# FIGURES -- Part B: Sleep Regression (Figures 5-6)
# ================================================================
def fig_sleep_regression(b_hrv, b_hr, b_spo2):
    n = next_fig_number()
    panels = [(b_hrv, "Heart Rate Variability (ms)"),
              (b_hr, "Heart Rate (bpm)"),
              (b_spo2, "Blood Oxygen Saturation (%)")]
    active = [(b,y) for b,y in panels if not b.empty]
    if not active:
        return n, "", ""
    fig, axes = plt.subplots(len(active), 1, figsize=(8, 3*len(active)), sharex=True)
    if len(active)==1: axes=[axes]
    for ax, (binned, ylabel) in zip(axes, active):
        x, y, e = binned["bin_center"].values, binned["mean"].values, binned["sem"].values
        ax.fill_between(x, y-e, y+e, color="lightgray", alpha=0.5, linewidth=0, label="+/-1 SEM")
        ax.plot(x, y, color="black", linewidth=2, label="Mean")
        ax.set_ylabel(ylabel); ax.set_xlim(0, 8)
        ax.legend(loc="upper right", frameon=False, fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)
    axes[-1].set_xlabel("Hours Since Sleep Onset")
    axes[-1].xaxis.set_major_locator(mticker.MultipleLocator(1))
    title = f"Figure {n}. Multi-channel sleep physiology regression (Mean +/- SEM, 30-min bins)."
    axes[0].set_title(title, pad=12)
    fig.align_ylabels(axes); fig.tight_layout()
    fname = f"Figure_{n:02d}_Sleep_Regression.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname


def fig_hr_hrv_scatter(merged):
    n = next_fig_number()
    df = merged.dropna(subset=["hrv", "hr"])
    if len(df) < 10:
        return n, "", ""
    x, y = df["hr"].values, df["hrv"].values
    max_pts = 8000
    if len(df) > max_pts:
        idx = np.random.default_rng(42).choice(len(df), max_pts, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y
    r_val, p_val = sp_stats.pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(xs, ys, s=8, alpha=0.25, color="dimgray", edgecolors="none", rasterized=True)
    xline = np.linspace(x.min(), x.max(), 200)
    ax.plot(xline, slope*xline+intercept, color="black", linewidth=2, linestyle="--")
    p_str = "< .001" if p_val < 0.001 else f"= {p_val:.3f}"
    annot = f"$r$ = {r_val:.3f}\n$p$ {p_str}\n$N$ = {len(df):,}"
    ax.text(0.97, 0.97, annot, transform=ax.transAxes, fontsize=11,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="lightgray", alpha=0.9))
    ax.set_xlabel("Heart Rate (bpm)"); ax.set_ylabel("Heart Rate Variability (ms)")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)
    title = f"Figure {n}. HR vs HRV correlation with linear regression."
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fname = f"Figure_{n:02d}_HR_HRV_Correlation.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname, r_val, p_val, len(df)


# ================================================================
# FIGURES -- Part C: Circadian Rhythms (Figures 7-9)
# ================================================================
def _rhythm_bar(df, group_col, xlabel, tick_labels, ylabel, fig_title, fig_fname):
    n = next_fig_number()
    if df.empty:
        return n, "", ""
    agg = df.groupby(group_col)["value"].agg(["mean", "sem", "count"]).reset_index()
    nb = len(agg)
    fig_w = 10 if nb > 12 else 8
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    bw = 0.85 if nb > 12 else 0.7
    ax.bar(agg[group_col], agg["mean"], width=bw, yerr=agg["sem"], color="dimgray",
           edgecolor="white", linewidth=0.5, capsize=3,
           error_kw={"elinewidth": 1.2, "capthick": 1.2, "ecolor": "black"})
    if tick_labels:
        ax.set_xticks(agg[group_col])
        labs = tick_labels[:len(agg)]
        if nb > 12:
            ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=9)
        else:
            ax.set_xticklabels(labs)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    title = f"Figure {n}. {fig_title}"
    ax.set_title(title, pad=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)
    fig.tight_layout()
    fname = f"Figure_{n:02d}_{fig_fname}.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname


def fig_circadian(df_hrv_raw):
    if df_hrv_raw.empty:
        return []
    df = df_hrv_raw.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dow"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    results = []
    results.append(_rhythm_bar(df, "hour", "Hour of Day",
                               [f"{h:02d}:00" for h in range(24)],
                               "Heart Rate Variability (ms)",
                               "HRV circadian rhythm (24-hour profile, Mean +/- SEM).",
                               "HRV_24h_Rhythm"))
    results.append(_rhythm_bar(df, "dow", "Day of Week",
                               ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                               "Heart Rate Variability (ms)",
                               "HRV by day of week (Mean +/- SEM).",
                               "HRV_Weekly_Rhythm"))
    results.append(_rhythm_bar(df, "month", "Month",
                               ["Jan","Feb","Mar","Apr","May","Jun",
                                "Jul","Aug","Sep","Oct","Nov","Dec"],
                               "Heart Rate Variability (ms)",
                               "HRV by month of year (Mean +/- SEM).",
                               "HRV_Monthly_Rhythm"))
    return results


# ================================================================
# FIGURES -- Part D: Sleep Stage Physiology (Figures 10-15)
# ================================================================
def fig_stage_boxplot(hr_s, hrv_s):
    n = next_fig_number()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (df, col, ylabel) in zip(axes, [(hr_s,"hr","Heart Rate (bpm)"),
                                             (hrv_s,"hrv","HRV (ms)")]):
        if df.empty:
            ax.set_ylabel(ylabel); continue
        data, labs = [], []
        for s in STAGE_ORDER:
            v = df.loc[df["stage"]==s, col].dropna()
            if len(v)>0: data.append(v.values); labs.append(s)
        bp = ax.boxplot(data, labels=labs, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", linewidth=2),
                        flierprops=dict(marker=".", markersize=2, color="dimgray", alpha=0.3))
        for patch, lab in zip(bp["boxes"], labs):
            patch.set_facecolor(STAGE_COLORS[lab]); patch.set_edgecolor("black")
        ax.set_ylabel(ylabel); ax.set_xlabel("Sleep Stage")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)
    title = f"Figure {n}. HR and HRV distribution by sleep stage (Awake/REM/Core/Deep)."
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = f"Figure_{n:02d}_Stage_Boxplot.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname


def fig_stage_violin(hr_s, hrv_s):
    n = next_fig_number()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (df, col, ylabel) in zip(axes, [(hr_s,"hr","Heart Rate (bpm)"),
                                             (hrv_s,"hrv","HRV (ms)")]):
        if df.empty:
            ax.set_ylabel(ylabel); continue
        present = [s for s in STAGE_ORDER if s in df["stage"].unique()]
        parts = ax.violinplot([df.loc[df["stage"]==s, col].dropna().values for s in present],
                              positions=range(len(present)), showmeans=True,
                              showmedians=True, showextrema=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(STAGE_COLORS[present[i]]); body.set_edgecolor("black")
            body.set_alpha(0.75)
        parts["cmeans"].set_color("black"); parts["cmedians"].set_color("dimgray")
        parts["cmedians"].set_linestyle("--")
        ax.set_xticks(range(len(present))); ax.set_xticklabels(present)
        ax.set_ylabel(ylabel); ax.set_xlabel("Sleep Stage")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)
    title = f"Figure {n}. HR and HRV distribution shape by sleep stage (violin plot)."
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = f"Figure_{n:02d}_Stage_Violin.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname


def fig_transition_heatmap(df_trans):
    n = next_fig_number()
    if df_trans.empty:
        return n, "", ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, mtitle in zip(axes, ["hr","hrv"],
                                  ["Mean HR Change (bpm)", "Mean HRV Change (ms)"]):
        sub = df_trans[df_trans["metric"]==metric].copy()
        if sub.empty:
            ax.set_title(mtitle); continue
        sub["phase"] = np.where(sub["delta_min"]<0, "before", "after")
        pm = sub.groupby(["from_stage","to_stage","phase"])["metric_value"].mean().unstack("phase")
        if "before" not in pm.columns or "after" not in pm.columns:
            ax.set_title(mtitle); continue
        pm["delta"] = pm["after"] - pm["before"]
        d = pm["delta"].reset_index()
        matrix = d.pivot(index="from_stage", columns="to_stage", values="delta")
        present = [s for s in STAGE_ORDER if s in matrix.index or s in matrix.columns]
        matrix = matrix.reindex(index=present, columns=present)
        vmax = max(abs(np.nanmin(matrix.values)), abs(np.nanmax(matrix.values)), 1)
        im = ax.imshow(matrix.values, cmap="RdGy_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(present))); ax.set_xticklabels(present, rotation=45, ha="right")
        ax.set_yticks(range(len(present))); ax.set_yticklabels(present)
        ax.set_xlabel("To Stage"); ax.set_ylabel("From Stage"); ax.set_title(mtitle)
        for i in range(len(present)):
            for j in range(len(present)):
                val = matrix.iloc[i,j]
                if pd.notna(val):
                    c = "white" if abs(val) > vmax*0.6 else "black"
                    ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                            fontsize=10, color=c, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    title = f"Figure {n}. Mean physiological change at sleep stage transitions."
    fig.suptitle(title, fontweight="bold", y=1.04)
    fig.tight_layout()
    fname = f"Figure_{n:02d}_Transition_Heatmap.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname


def fig_transition_timecourse(df_trans):
    n = next_fig_number()
    if df_trans.empty:
        return n, "", ""
    top = df_trans.groupby("transition").size().sort_values(ascending=False).head(4).index.tolist()
    fig, axes = plt.subplots(len(top), 2, figsize=(12, 3.2*len(top)), squeeze=False)
    for row, tname in enumerate(top):
        sub = df_trans[df_trans["transition"]==tname]
        for col, metric, ylabel in [(0,"hr","Heart Rate (bpm)"), (1,"hrv","HRV (ms)")]:
            ax = axes[row, col]
            ms = sub[sub["metric"]==metric].copy()
            if ms.empty: continue
            ms["bin"] = ms["delta_min"].round(0)
            agg = ms.groupby("bin")["metric_value"].agg(["mean","sem"]).reset_index()
            agg = agg[(agg["bin"]>=-TRANSITION_WINDOW_MIN)&(agg["bin"]<=TRANSITION_WINDOW_MIN)]
            x,y,e = agg["bin"].values, agg["mean"].values, agg["sem"].fillna(0).values
            ax.fill_between(x, y-e, y+e, color="lightgray", alpha=0.6)
            ax.plot(x, y, color="black", linewidth=1.8)
            ax.axvline(0, color="dimgray", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_ylabel(ylabel)
            if row==len(top)-1: ax.set_xlabel("Minutes Relative to Transition")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)
        axes[row,0].annotate(tname, xy=(-0.25, 0.5), xycoords="axes fraction",
                             fontsize=10, fontweight="bold", ha="right", va="center")
    title = f"Figure {n}. HR and HRV trajectory around stage transitions (+/- SEM)."
    fig.suptitle(title, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = f"Figure_{n:02d}_Transition_Timecourse.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname


def _draw_one_hypnogram(df_stages, df_hr, df_hrv, night_id, fig_num_label):
    """Draw a single-night hypnogram panel set. Returns (fig, title_str, night_date_str)."""
    night = df_stages[df_stages["night_id"]==night_id].copy()
    onset = night["sleep_onset"].iloc[0]; wake = night["end"].max()
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True,
                              gridspec_kw={"height_ratios": [1, 1.2, 1.2]})
    stage_y = {"Deep": 0, "Core": 1, "REM": 2, "Awake": 3}
    ax = axes[0]
    for _, f in night.iterrows():
        xs = (f["start"]-onset).total_seconds()/3600
        xe = (f["end"]-onset).total_seconds()/3600
        ax.barh(stage_y.get(f["stage"],1), xe-xs, left=xs, height=0.8,
                color=STAGE_COLORS.get(f["stage"],"gray"), edgecolor="white", linewidth=0.3)
    ax.set_yticks(list(stage_y.values())); ax.set_yticklabels(list(stage_y.keys()))
    ax.set_ylabel("Stage"); ax.invert_yaxis()
    handles = [mpatches.Patch(fc=STAGE_COLORS[s], ec="black", lw=0.5, label=s) for s in STAGE_ORDER]
    ax.legend(handles=handles, loc="upper right", ncol=4, frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)

    for axi, (df_src, ylabel) in zip(axes[1:],
            [(df_hr, "Heart Rate (bpm)"), (df_hrv, "HRV (ms)")]):
        m = (df_src["timestamp"]>=onset)&(df_src["timestamp"]<=wake)
        chunk = df_src.loc[m].copy()
        if not chunk.empty:
            chunk["h"] = (chunk["timestamp"]-onset).dt.total_seconds()/3600
            axi.plot(chunk["h"], chunk["value"], color="black", linewidth=0.8, alpha=0.8)
            for _, f in night.iterrows():
                xs = (f["start"]-onset).total_seconds()/3600
                xe = (f["end"]-onset).total_seconds()/3600
                axi.axvspan(xs, xe, alpha=0.12, color=STAGE_COLORS.get(f["stage"],"gray"))
        axi.set_ylabel(ylabel)
        axi.spines["top"].set_visible(False); axi.spines["right"].set_visible(False); axi.grid(False)
    axes[-1].set_xlabel("Hours Since Sleep Onset")
    night_date = onset.strftime("%Y-%m-%d")
    title = f"Figure {fig_num_label}. Single-night hypnogram with HR and HRV overlay ({night_date})."
    axes[0].set_title(title, pad=12)
    fig.tight_layout()
    return fig, title, night_date


def fig_hypnogram(df_stages, df_hr, df_hrv, hypnogram_config=None):
    """Generate one or more hypnogram figures based on user config."""
    results = []
    if df_stages.empty:
        n = next_fig_number()
        return [(n, "", "")]

    cfg = hypnogram_config or {"night_ids": None, "count": 1}
    count = cfg.get("count", 1)
    chosen_ids = cfg.get("night_ids", None)

    if chosen_ids is None:
        # Auto-select: pick the night(s) with the most fragments
        night_sizes = df_stages.groupby("night_id").size().sort_values(ascending=False)
        chosen_ids = night_sizes.head(count).index.tolist()

    for nid in chosen_ids:
        if nid not in df_stages["night_id"].values:
            continue
        n = next_fig_number()
        fig, title, night_date = _draw_one_hypnogram(
            df_stages, df_hr, df_hrv, nid, n)
        fname = f"Figure_{n:02d}_Hypnogram_{night_date}.png"
        path = OUTPUT_DIR / fname
        fig.savefig(path); plt.close(fig)
        done(path)
        results.append((n, title, fname))

    return results if results else [(next_fig_number(), "", "")]


def fig_stage_proportion(df_stages):
    n = next_fig_number()
    if df_stages.empty:
        return n, "", ""
    rows = []
    for _, f in df_stages.iterrows():
        onset = f["sleep_onset"]; t = f["start"]
        while t < f["end"]:
            h = (t-onset).total_seconds()/3600
            if 0 <= h <= 10:
                rows.append({"hour": round(h*2)/2, "stage": f["stage"]})
            t += timedelta(minutes=1)
    if not rows:
        return n, "", ""
    df = pd.DataFrame(rows)
    ct = pd.crosstab(df["hour"], df["stage"])
    for s in STAGE_ORDER:
        if s not in ct.columns: ct[s] = 0
    ct = ct[STAGE_ORDER]
    ct_pct = ct.div(ct.sum(axis=1), axis=0)*100
    ct_pct = ct_pct[ct_pct.index<=8]
    rc = ct.sum(axis=1).reindex(ct_pct.index)
    ct_pct = ct_pct[rc >= rc.max()*0.05]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = ct_pct.index.values; bottoms = np.zeros(len(x))
    for s in STAGE_ORDER:
        v = ct_pct[s].values
        ax.fill_between(x, bottoms, bottoms+v, color=STAGE_COLORS[s], label=s,
                        linewidth=0.5, edgecolor="white")
        bottoms += v
    ax.set_xlim(0,8); ax.set_ylim(0,100)
    ax.set_xlabel("Hours Since Sleep Onset"); ax.set_ylabel("Stage Proportion (%)")
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=4)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.grid(False)
    title = f"Figure {n}. Sleep stage proportion across the night (stacked area)."
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fname = f"Figure_{n:02d}_Stage_Proportion.png"
    path = OUTPUT_DIR / fname
    fig.savefig(path); plt.close(fig)
    done(path)
    return n, title, fname


# ================================================================
# FULL TEXT REPORT
# ================================================================
def write_report(raw, df_hrv, df_hr, df_spo2, sessions_valid,
                 hr_staged, hrv_staged, df_trans, merged,
                 outlier_info, figure_log,
                 date_range_applied=None):
    lines = []
    lines.append("=" * 70)
    lines.append("  APPLE HEALTH SLEEP & PHYSIOLOGY ANALYSIS REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    # Section 1: Data Overview
    lines.append("\n\n--- SECTION 1: DATA OVERVIEW ---\n")
    if date_range_applied:
        lines.append(f"  ** User-selected date range: "
                     f"{date_range_applied[0]} to {date_range_applied[1]} **")
        lines.append("")
    lines.append(f"  Heart Rate records     : {len(raw['hr']):>10,}")
    lines.append(f"  HRV (SDNN) records     : {len(raw['hrv']):>10,}")
    lines.append(f"  SpO2 records           : {len(raw['spo2']):>10,}")
    lines.append(f"  Sleep stage fragments  : {len(raw['stages']):>10,}")
    if not df_hrv.empty:
        span_days = (df_hrv["timestamp"].max() - df_hrv["timestamp"].min()).days
        lines.append(f"\n  Date range  : {df_hrv['timestamp'].min().strftime('%Y-%m-%d')} "
                     f"to {df_hrv['timestamp'].max().strftime('%Y-%m-%d')}")
        lines.append(f"  Span        : {span_days} days ({span_days/365.25:.1f} years)")

    # Section 2: Descriptive Statistics
    lines.append("\n\n--- SECTION 2: DESCRIPTIVE STATISTICS ---\n")
    if outlier_info:
        for label, info in outlier_info.items():
            lines.append(f"  {label}:")
            lines.append(f"    N             = {info['N']:,}")
            lines.append(f"    Mean (SD)     = {info['mean']:.2f} ({info['sd']:.2f})")
            lines.append(f"    Median        = {info['median']:.2f}")
            lines.append(f"    IQR           = [{info['Q1']:.2f}, {info['Q3']:.2f}]")
            lines.append(f"    Outlier fence = [{info['lo']:.2f}, {info['hi']:.2f}]")
            lines.append(f"    Outliers      = {info['n_outliers']:,} ({info['pct']:.1f}%)")
            lines.append("")

    # Section 3: Sleep Sessions
    lines.append("\n--- SECTION 3: SLEEP SESSIONS ---\n")
    lines.append(f"  Valid nights           : {len(sessions_valid):,}")
    if not sessions_valid.empty:
        dur = sessions_valid["duration_hours"]
        lines.append(f"  Mean duration          : {dur.mean():.2f}h (SD = {dur.std():.2f})")
        lines.append(f"  Min / Max              : {dur.min():.2f}h / {dur.max():.2f}h")

    # Section 4: HR & HRV by Sleep Stage
    lines.append("\n\n--- SECTION 4: PHYSIOLOGY BY SLEEP STAGE ---\n")
    for label, df, col in [("Heart Rate (bpm)", hr_staged, "hr"),
                            ("HRV SDNN (ms)", hrv_staged, "hrv")]:
        if df.empty:
            continue
        lines.append(f"  {label}:")
        lines.append(f"    {'Stage':<8} {'N':>8} {'Mean':>8} {'SD':>8} {'Median':>8}")
        for s in STAGE_ORDER:
            v = df.loc[df["stage"]==s, col]
            if len(v)==0: continue
            lines.append(f"    {s:<8} {len(v):>8,} {v.mean():>8.1f} {v.std():>8.1f} {v.median():>8.1f}")
        # Kruskal-Wallis
        groups = [df.loc[df["stage"]==s, col].dropna().values
                  for s in STAGE_ORDER if s in df["stage"].unique()]
        groups = [g for g in groups if len(g)>0]
        if len(groups) >= 2:
            stat, pval = sp_stats.kruskal(*groups)
            p_str = "< .001" if pval < 0.001 else f"= {pval:.4f}"
            lines.append(f"    Kruskal-Wallis: H = {stat:.2f}, p {p_str}")
        lines.append("")

    # Section 5: Cohen's d
    lines.append("\n--- SECTION 5: PAIRWISE EFFECT SIZES (Cohen's d) ---\n")
    for label, df, col in [("HR", hr_staged, "hr"), ("HRV", hrv_staged, "hrv")]:
        if df.empty: continue
        lines.append(f"  {label}:")
        present = [s for s in STAGE_ORDER if s in df["stage"].unique()]
        for i in range(len(present)):
            for j in range(i+1, len(present)):
                v1 = df.loc[df["stage"]==present[i], col].dropna()
                v2 = df.loc[df["stage"]==present[j], col].dropna()
                if len(v1)<2 or len(v2)<2: continue
                psd = np.sqrt((v1.var()+v2.var())/2)
                if psd > 0:
                    d = (v1.mean()-v2.mean())/psd
                    lines.append(f"    {present[i]:>6} vs {present[j]:<6}: d = {d:+.3f}")
        lines.append("")

    # Section 6: Stage Transitions
    lines.append("\n--- SECTION 6: STAGE TRANSITION DYNAMICS ---\n")
    if not df_trans.empty:
        for metric, mlabel in [("hr","Heart Rate"), ("hrv","HRV")]:
            sub = df_trans[df_trans["metric"]==metric].copy()
            if sub.empty: continue
            lines.append(f"  {mlabel}:")
            sub["phase"] = np.where(sub["delta_min"]<0, "before", "after")
            pm = sub.groupby(["transition","phase"])["metric_value"].mean().unstack("phase")
            if "before" in pm.columns and "after" in pm.columns:
                pm["delta"] = pm["after"] - pm["before"]
                lines.append(f"    {'Transition':<20} {'Before':>8} {'After':>8} {'Delta':>8}")
                for t in pm.index:
                    lines.append(f"    {t:<20} {pm.loc[t,'before']:>8.1f} "
                                 f"{pm.loc[t,'after']:>8.1f} {pm.loc[t,'delta']:>+8.1f}")
            lines.append("")

    # Section 7: HR <-> HRV Correlation
    lines.append("\n--- SECTION 7: HR <-> HRV CORRELATION ---\n")
    df_corr = merged.dropna(subset=["hrv","hr"]) if not merged.empty else pd.DataFrame()
    if len(df_corr) >= 10:
        r, p = sp_stats.pearsonr(df_corr["hr"], df_corr["hrv"])
        p_str = "< .001" if p < 0.001 else f"= {p:.4f}"
        lines.append(f"  Pearson r = {r:.3f}, p {p_str}, N = {len(df_corr):,}")

    # Section 8: Figure Index
    lines.append("\n\n--- FIGURE INDEX ---\n")
    for entry in figure_log:
        if len(entry) >= 3 and entry[1] and entry[2]:
            lines.append(f"  {entry[1]}")
            lines.append(f"    File: {entry[2]}")
            lines.append("")

    # Section 9: Methods
    lines.append("\n--- PROCESSING METHODS ---\n")
    lines.append("  XML parsing        : Streaming (iterparse), memory-safe for large files")
    lines.append(f"  Sleep merge gap    : {SLEEP_GAP_HOURS} hours")
    lines.append(f"  Min sleep duration : {MIN_SLEEP_HOURS} hours")
    lines.append(f"  HR<->HRV alignment : pd.merge_asof, +/-{HR_HRV_TOLERANCE_MIN} min tolerance")
    lines.append(f"  Regression bins    : {BIN_SIZE_HOURS} hours (30 min)")
    lines.append(f"  Tail cutoff        : < {TAIL_CUTOFF_FRAC*100:.0f}% of peak bin count")
    lines.append(f"  Smoothing          : Rolling mean, window={SMOOTH_WINDOW}, centered")
    lines.append(f"  Transition window  : +/-{TRANSITION_WINDOW_MIN} min around boundary")
    lines.append(f"  APA compliance     : No top/right spines, no grid, greyscale only,")
    lines.append(f"                       Arial font, 300 DPI, shaded SEM bands")

    lines.append("\n" + "=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    text = "\n".join(lines)
    path = OUTPUT_DIR / "Analysis_Report.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    done(path)
    safe_print("")
    safe_print(text)


# ================================================================
# MAIN PIPELINE
# ================================================================
def main():
    apply_apa_style()

    # -- Step 0: Get file --
    xml_path = get_xml_path()

    # -- Create output directory next to the script --
    global OUTPUT_DIR
    OUTPUT_DIR = Path(os.path.dirname(os.path.abspath(xml_path))) / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    TOTAL_STEPS = 9
    figure_log = []
    outlier_info = {}

    # -- Step 1: Parse XML --
    step(1, TOTAL_STEPS, "Reading your Apple Health data (this may take a few minutes for large files)...")
    raw = parse_xml(xml_path)

    # -- Step 2: Build DataFrames --
    step(2, TOTAL_STEPS, "Organising and sorting your health records...")
    df_hr = to_df(raw["hr"])
    df_hrv = to_df(raw["hrv"])
    df_spo2 = to_df(raw["spo2"])
    df_stages = build_stage_df(raw["stages"])
    substep(f"Records ready: HR={len(df_hr):,}, HRV={len(df_hrv):,}, "
            f"SpO2={len(df_spo2):,}, Stages={len(df_stages):,}")

    # -- Step 2b: Optional date-range filter --
    d_start, d_end = ask_date_range(df_hr, df_hrv, df_spo2, df_stages)
    if d_start is not None:
        df_hr = apply_date_filter(df_hr, "timestamp", d_start, d_end)
        df_hrv = apply_date_filter(df_hrv, "timestamp", d_start, d_end)
        df_spo2 = apply_date_filter(df_spo2, "timestamp", d_start, d_end)
        if not df_stages.empty:
            df_stages = df_stages[
                (df_stages["start"] >= d_start) & (df_stages["end"] <= d_end)
            ].copy().reset_index(drop=True)
        # Also filter the raw counts for the report
        raw["hr"] = [r for r in raw["hr"]
                     if d_start <= pd.Timestamp(r["timestamp"]) <= d_end]
        raw["hrv"] = [r for r in raw["hrv"]
                      if d_start <= pd.Timestamp(r["timestamp"]) <= d_end]
        raw["spo2"] = [r for r in raw["spo2"]
                       if d_start <= pd.Timestamp(r["timestamp"]) <= d_end]
        raw["stages"] = [r for r in raw["stages"]
                         if d_start <= pd.Timestamp(r["start"]) <= d_end]
        substep(f"After date filter: HR={len(df_hr):,}, HRV={len(df_hrv):,}, "
                f"SpO2={len(df_spo2):,}, Stages={len(df_stages):,}")

    # -- Step 3: Merge sleep nights --
    step(3, TOTAL_STEPS, "Identifying your sleep nights (merging fragmented sleep data)...")
    df_stages, sessions_valid = merge_sleep_nights(df_stages)

    # Build sessions_all for the merge diagnostic figure
    sessions_all_tmp = build_stage_df(raw["stages"])
    if not sessions_all_tmp.empty:
        gap = timedelta(hours=SLEEP_GAP_HOURS)
        ids = [0]
        for i in range(1, len(sessions_all_tmp)):
            if sessions_all_tmp.iloc[i]["start"] - sessions_all_tmp.iloc[i-1]["end"] > gap:
                ids.append(ids[-1]+1)
            else:
                ids.append(ids[-1])
        sessions_all_tmp["nid"] = ids
        sessions_all = pd.DataFrame({
            "sleep_onset": sessions_all_tmp.groupby("nid")["start"].min(),
            "wake_time": sessions_all_tmp.groupby("nid")["end"].max(),
        })
        sessions_all["duration_hours"] = \
            (sessions_all["wake_time"]-sessions_all["sleep_onset"]).dt.total_seconds()/3600
    else:
        sessions_all = pd.DataFrame(columns=["sleep_onset","wake_time","duration_hours"])

    # -- Step 3b: Ask user about hypnogram night selection --
    hypnogram_config = ask_hypnogram_config(sessions_valid)

    # -- Step 4: Align physiology to sleep --
    step(4, TOTAL_STEPS, "Aligning your heart rate and HRV to sleep stages and sleep time...")
    hr_staged = map_to_stages(df_hr, df_stages, "hr")
    hrv_staged = map_to_stages(df_hrv, df_stages, "hrv")
    substep(f"HR matched to stages: {len(hr_staged):,}")
    substep(f"HRV matched to stages: {len(hrv_staged):,}")

    sleep_hrv = map_to_sleep_time(df_hrv, sessions_valid, "hrv")
    sleep_hr = map_to_sleep_time(df_hr, sessions_valid, "hr")
    sleep_spo2 = map_to_sleep_time(df_spo2, sessions_valid, "spo2")
    substep(f"Sleep-time aligned: HRV={len(sleep_hrv):,}, HR={len(sleep_hr):,}, SpO2={len(sleep_spo2):,}")

    merged = align_hr_hrv(df_hrv, df_hr)
    substep(f"HR<->HRV paired records: {merged['hr'].notna().sum():,}")

    # -- Step 5: Extract stage transitions --
    step(5, TOTAL_STEPS, "Analysing what happens when your sleep stages change...")
    df_trans = extract_transitions(df_stages, df_hr, df_hrv)
    substep(f"Transition data points: {len(df_trans):,}")

    # -- Step 6: Bin & smooth for regression --
    step(6, TOTAL_STEPS, "Computing sleep regression curves (30-min bins, smoothing)...")
    b_hrv = bin_and_smooth(sleep_hrv, "hrv")
    b_hr = bin_and_smooth(sleep_hr, "hr")
    b_spo2 = bin_and_smooth(sleep_spo2, "spo2")
    substep("Binning complete.")

    # -- Step 7: Generate all figures --
    step(7, TOTAL_STEPS, "Drawing all figures (this may take a moment)...")

    substep("Part A: Data quality figures...")
    figure_log.append(fig_record_counts(raw))

    figure_log.append(fig_temporal_coverage(df_hrv, df_hr, df_spo2))

    dist_result = fig_distributions(df_hrv, df_hr, df_spo2)
    if len(dist_result) == 4:
        figure_log.append(dist_result[:3])
        outlier_info = dist_result[3]
    else:
        figure_log.append(dist_result)

    figure_log.append(fig_sleep_merge(sessions_all, sessions_valid, len(raw["stages"])))

    substep("Part B: Sleep regression figures...")
    figure_log.append(fig_sleep_regression(b_hrv, b_hr, b_spo2))

    scatter_result = fig_hr_hrv_scatter(merged)
    figure_log.append(scatter_result[:3] if len(scatter_result) > 3 else scatter_result)

    substep("Part C: Circadian rhythm figures...")
    for r in fig_circadian(df_hrv):
        figure_log.append(r)

    substep("Part D: Sleep stage physiology figures...")
    figure_log.append(fig_stage_boxplot(hr_staged, hrv_staged))
    figure_log.append(fig_stage_violin(hr_staged, hrv_staged))
    figure_log.append(fig_transition_heatmap(df_trans))
    figure_log.append(fig_transition_timecourse(df_trans))
    for hyp_entry in fig_hypnogram(df_stages, df_hr, df_hrv, hypnogram_config):
        figure_log.append(hyp_entry)
    figure_log.append(fig_stage_proportion(df_stages))

    # -- Step 8: Write report --
    step(8, TOTAL_STEPS, "Writing the analysis report...")
    date_range_info = None
    if d_start is not None:
        date_range_info = (d_start.strftime("%Y-%m-%d"), d_end.strftime("%Y-%m-%d"))
    write_report(raw, df_hrv, df_hr, df_spo2, sessions_valid,
                 hr_staged, hrv_staged, df_trans, merged,
                 outlier_info, figure_log,
                 date_range_applied=date_range_info)

    # -- Step 9: Done --
    step(9, TOTAL_STEPS, "All done!")
    safe_print("")
    banner("Analysis Complete!")
    safe_print("")
    safe_print(f"  All results saved to:")
    safe_print(f"    {OUTPUT_DIR.resolve()}")
    safe_print("")
    safe_print(f"  Files generated:")
    for f in sorted(OUTPUT_DIR.glob("Figure_*.png")):
        safe_print(f"    {f.name}")
    safe_print(f"    Analysis_Report.txt")
    safe_print("")
    safe_print("  You can now open the 'output' folder to view your results.")
    safe_print("  Thank you for using the Apple Health Analyzer!")
    safe_print("")


if __name__ == "__main__":
    main()
