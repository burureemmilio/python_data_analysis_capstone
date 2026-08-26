"""
Kenya Road Accidents 2017 — A Data Story
A Streamlit app that walks through the road-accidents analysis as a guided
narrative: Introduction -> General Overview -> What -> When -> Where -> Why
-> Conclusion -> Recommendations.
"""

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Kenya Road Accidents 2017 — A Data Story",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSV_PATH = "road_accidents_clean.csv"

PERIOD_ORDER = ["MORNING", "AFTERNOON", "EVENING", "NIGHT"]
MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]
DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]
AGE_ORDER = ["0-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

PALETTE = px.colors.sequential.Reds
ACCENT = "#C0392B"
ACCENT2 = "#34495E"


# --------------------------------------------------------------------------
# Data loading & cleaning (mirrors the analysis notebook)
# --------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Focus on 2017, the most complete year in the dataset
    df = df[df["year"] == 2017].copy()

    text_cols = ["county", "road", "place", "victim_type", "gender",
                 "cause_group", "time_period", "month_name", "day_name"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.upper()
            df[c] = df[c].replace({"NAN": np.nan, "": np.nan, "NONE": np.nan})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df["month_name_title"] = df["month_name"].str.title()
    df["day_name_title"] = df["day_name"].str.title()

    df["age_group"] = df["age_group"].astype(str).str.strip().str.upper()
    age_map = {a.upper(): a for a in AGE_ORDER}
    df["age_group"] = df["age_group"].map(age_map).fillna(df["age_group"])

    df["location_signature"] = (
        df["county"].fillna("UNKNOWN") + " — " + df["place"].fillna("UNKNOWN")
    )
    df["county_road"] = (
        df["county"].fillna("UNKNOWN") + " | " + df["road"].fillna("UNKNOWN")
    )

    # Reconstruct distinct crash events for severity analysis
    crash_key_cols = [c for c in ["date", "time", "road", "place",
                                  "vehicles_involved"] if c in df.columns]
    key_df = df[crash_key_cols].astype(str).fillna("UNKNOWN")
    df["crash_key"] = key_df.agg("|".join, axis=1)

    return df


@st.cache_data
def compute_severity(df: pd.DataFrame) -> pd.Series:
    severity = df.groupby("crash_key")["number"].sum()

    def bucket(n):
        if n <= 1:
            return "1 victim"
        elif n == 2:
            return "2 victims"
        elif n == 3:
            return "3 victims"
        return "4+ victims (mass casualty)"

    order = ["1 victim", "2 victims",
             "3 victims", "4+ victims (mass casualty)"]
    return severity.apply(bucket).value_counts().reindex(order).fillna(0)


GEOJSON_URL = "https://raw.githubusercontent.com/mikelmaron/kenya-election-data/master/data/counties.geojson"
MANUAL_COUNTY_OVERRIDES = {
    "ELGEYOMARAKWET": "ELEGEYOMARAKWET",
}


def normalize_county_name(name: str) -> str:
    return re.sub(r"[^A-Z]", "", str(name).upper())


@st.cache_data(ttl=60 * 60 * 24, show_spinner=False)
def fetch_kenya_geojson():
    """Fetch Kenya county boundaries (constituency-level polygons tagged
    with their county). Returns the raw GeoJSON dict, or None if the
    fetch fails (e.g. no internet access in the deployment environment)."""
    try:
        resp = requests.get(GEOJSON_URL, timeout=15)
        resp.raise_for_status()
        gj = resp.json()
        # Tag every feature with a normalized county name so Plotly can
        # match it to the accident totals below. No dissolve step is
        # needed: same-county constituencies simply render in the same
        # color, which reads as one shape per county.
        for feat in gj["features"]:
            raw = feat["properties"].get("COUNTY_NAM", "")
            feat["properties"]["county_norm"] = normalize_county_name(raw)
        return gj
    except Exception:
        return None


@st.cache_data
def county_totals_for_map(df: pd.DataFrame) -> pd.DataFrame:
    totals = (
        df["county"].value_counts().rename_axis("county")
        .reset_index(name="accidents")
    )
    totals["county_norm"] = totals["county"].apply(normalize_county_name)
    totals["county_norm"] = totals["county_norm"].replace(
        MANUAL_COUNTY_OVERRIDES)
    return totals


df = load_data(CSV_PATH)
severity_buckets = compute_severity(df)

VEHICLE_COLS = [c for c in ["motorcycle", "matatu", "bus", "lorry_truck",
                            "pickup", "tractor", "car_suv", "trailer_tanker",
                            "van_minivan", "unknown_vehicle",
                            "unspecified_vehicle"] if c in df.columns]

# --------------------------------------------------------------------------
# Small pre-computed facts used to make the narrative text data-driven
# --------------------------------------------------------------------------
TOTAL_RECORDS = len(df)
N_COUNTIES = df["county"].nunique()
N_ROADS = df["road"].nunique()
TOP_COUNTY = df["county"].value_counts().idxmax()
TOP_COUNTY_N = df["county"].value_counts().max()
TOP_ROAD = df["road"].value_counts().idxmax()
TOP_CAUSE = df["cause_group"].value_counts().idxmax()
TOP_CAUSE_PCT = df["cause_group"].value_counts(normalize=True).max() * 100
TOP_VICTIM = df["victim_type"].value_counts().idxmax()
TOP_VICTIM_PCT = df["victim_type"].value_counts(normalize=True).max() * 100
TOP_VEHICLE = df[VEHICLE_COLS].sum().idxmax().replace("_", " ").title()
TOP_PERIOD = (df["time_period"].value_counts().reindex(PERIOD_ORDER)
              .idxmax())
risk_clock_tmp = pd.crosstab(df["day_name_title"], df["time_period"])
risk_clock_tmp = risk_clock_tmp.reindex(index=DAY_ORDER, columns=PERIOD_ORDER)
_flat = risk_clock_tmp.stack()
RISKIEST_DAY, RISKIEST_PERIOD = _flat.idxmax()
RISKIEST_N = int(_flat.max())
MASS_CASUALTY_PCT = (
    severity_buckets.get("4+ victims (mass casualty)", 0) /
    severity_buckets.sum() * 100
)
MALE_PCT = (
    df["gender"].value_counts(normalize=True).get("MALE", np.nan) * 100
)
_cause_counts_all = df["cause_group"].value_counts()
_cum_pct_all = _cause_counts_all.cumsum() / _cause_counts_all.sum() * 100
N_CAUSES_80 = int((_cum_pct_all <= 80).sum()) + 1


def section_header(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


def insight_box(text):
    st.info(text, icon="💡")


# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------
SECTIONS = [
    "Introduction",
    "General Overview",
    "Where",
    "Who",
    "When",
    "Why",
    "Conclusion",
    "Recommendations",
]

if "section_idx" not in st.session_state:
    st.session_state.section_idx = 0

st.sidebar.title("🚦 Road Safety Story")
st.sidebar.caption("Kenya road accidents, 2017")
chosen = st.sidebar.radio(
    "Navigate",
    SECTIONS,
    index=st.session_state.section_idx,
    key="nav_radio",
)
st.session_state.section_idx = SECTIONS.index(chosen)

st.sidebar.divider()
st.sidebar.markdown(
    f"**Dataset:** {TOTAL_RECORDS:,} victim records\n\n"
    f"**Year:** 2017\n\n"
    f"**Counties:** {N_COUNTIES}\n\n"
    f"**Data Source: [Kenyan Accidents Dataset from Kaggle](https://www.kaggle.com/datasets/waawerufidelis/accidents-kenya?resource=download)**"
)

section = chosen


def nav_buttons():
    st.write("")
    col1, _, col3 = st.columns([1, 4, 1])
    idx = st.session_state.section_idx
    with col1:
        if idx > 0:
            if st.button("← Previous", use_container_width=True):
                st.session_state.section_idx -= 1
                st.rerun()
    with col3:
        if idx < len(SECTIONS) - 1:
            if st.button("Next →", use_container_width=True):
                st.session_state.section_idx += 1
                st.rerun()


# ==========================================================================
# 1. INTRODUCTION
# ==========================================================================
if section == "Introduction":
    st.title("🚦 Kenya Road Accidents, 2017")
    st.markdown("### A data story from crash records to concrete action")

    st.markdown(
        f"""
Road traffic accidents remain one of Kenya's most persistent public-safety
challenges — costing lives, livelihoods, and public resources every year.
This report walks through **{TOTAL_RECORDS:,} recorded victim-level accident
records from 2017**, spanning **{N_COUNTIES} counties** and **{N_ROADS} named
roads**, to build a clear, evidence-based picture of the problem.

Rather than a static report, this is structured as a **guided story**. Use
the navigation on the left to move through it in order:
        """
    )

    st.markdown(
        """
1. **General Overview** — the shape of the dataset at a glance
2. **Where** — the counties, roads, and specific hotspots most affected
3. **Who** — who is affected, what vehicles are involved, how severe crashes are
4. **When** — the times, days, and months accidents cluster around
5. **Why** — the recorded causes behind the crashes
6. **Conclusion** — pulling the findings together
7. **Recommendations** — what should be done about it
        """
    )

    insight_box(
        "Each visualization below is followed by a short explanation of "
        "what it shows and why it matters — by the end, the goal is that "
        "the recommendations feel like the obvious next step, not a guess."
    )


# ==========================================================================
# 2. GENERAL OVERVIEW
# ==========================================================================
elif section == "General Overview":
    section_header(
        "General Overview",
        "The shape of the dataset before we zoom into specifics.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Victim records (2017)", f"{TOTAL_RECORDS:,}")
    c2.metric("Counties with records", N_COUNTIES)
    c3.metric("Roads recorded", N_ROADS)
    c4.metric("Mass-casualty crashes (4+ victims)",
              f"{MASS_CASUALTY_PCT:.1f}%")

    st.write("")
    monthly = (
        df["month_name_title"].value_counts().reindex(MONTH_ORDER).fillna(0)
    )
    fig = px.line(
        x=monthly.index, y=monthly.values, markers=True,
        labels={"x": "Month", "y": "Number of records"},
        title="Recorded Accidents Across 2017",
    )
    fig.update_traces(line_color=ACCENT, marker=dict(size=8))
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)

    peak_month = monthly.idxmax()
    insight_box(
        f"Accident records are spread across the whole year rather than "
        f"concentrated in one season, with **{peak_month}** recording the "
        f"most cases ({int(monthly.max())}). This sets the baseline — the "
        f"sections that follow break this total down by *who* is affected, "
        f"*when*, *where*, and *why* it happens."
    )


# ==========================================================================
# 3. WHERE
# ==========================================================================
elif section == "Where":
    section_header(
        "Where — Geographic Hotspots",
        "The counties, roads, and specific spots that account for the most incidents.",
    )

    kenya_geojson = fetch_kenya_geojson()
    if kenya_geojson is not None:
        county_totals = county_totals_for_map(df)
        map_kwargs = dict(
            data_frame=county_totals,
            geojson=kenya_geojson,
            locations="county_norm",
            featureidkey="properties.county_norm",
            color="accidents",
            color_continuous_scale="Reds",
            zoom=4.7,
            center={"lat": 0.4, "lon": 37.9},
            opacity=0.75,
            hover_name="county",
            labels={"accidents": "Recorded accidents"},
            title="Accident Intensity by County",
        )
        # Plotly >=5.24 renamed choropleth_mapbox -> choropleth_map (uses
        # free MapLibre tiles, no token needed). Support both so this keeps
        # working across plotly versions.
        if hasattr(px, "choropleth_map"):
            fig = px.choropleth_map(map_style="carto-positron", **map_kwargs)
        else:
            fig = px.choropleth_mapbox(
                mapbox_style="carto-positron", **map_kwargs)
        fig.update_layout(height=550, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)
        insight_box(
            f"The map makes the geographic concentration visible at a "
            f"glance — **{TOP_COUNTY.title()}** stands out darkest, and a "
            f"handful of counties around it carry most of the national "
            f"total, while large parts of the country record comparatively "
            f"few incidents."
        )
    else:
        st.warning(
            "County boundary data couldn't be fetched (this environment "
            "may not have internet access). The county ranking below still "
            "shows the same information without the map.",
            icon="⚠️",
        )

    st.write("")
    top_counties = df["county"].value_counts().head(15)
    fig = px.bar(
        x=top_counties.values, y=top_counties.index, orientation="h",
        labels={"x": "Number of records", "y": ""},
        title="Top 15 Counties by Recorded Accidents",
        color=top_counties.values, color_continuous_scale="Reds",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=480,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    insight_box(
        f"**{TOP_COUNTY.title()}** records the highest number of accident "
        f"victims ({TOP_COUNTY_N}), making it a natural starting point for "
        f"county-level road-safety investment."
    )

    st.write("")
    top_roads = df["road"].value_counts().head(15)
    fig = px.bar(
        x=top_roads.values, y=top_roads.index, orientation="h",
        labels={"x": "Number of records", "y": ""},
        title="Top 15 Roads by Recorded Accidents",
        color=top_roads.values, color_continuous_scale="Oranges",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=480,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    insight_box(
        f"The **{TOP_ROAD.title()}** road records the most incidents of "
        f"any single named route — a candidate for engineering review "
        f"(signage, speed bumps, lighting) rather than enforcement alone."
    )


# ==========================================================================
# 4. WHO
# ==========================================================================
elif section == "Who":
    section_header(
        "Who — Who Is Affected, and How Badly",
        "The people, vehicles, and severity behind the numbers.",
    )

    # Victim type
    victim_counts = df["victim_type"].value_counts().head(10)
    fig = px.bar(
        x=victim_counts.values, y=victim_counts.index, orientation="h",
        labels={"x": "Number of records", "y": ""},
        title="Most Affected Road User Types",
        color=victim_counts.values, color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    insight_box(
        f"**{TOP_VICTIM.title()}s** are the single most affected road-user "
        f"group, accounting for **{TOP_VICTIM_PCT:.0f}%** of all recorded "
        f"victims — a strong early signal for who road-safety "
        f"interventions should prioritise."
    )

    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        gender_counts = df["gender"].value_counts()
        fig = px.pie(
            values=gender_counts.values, names=gender_counts.index,
            title="Victims by Gender", hole=0.45,
            color_discrete_sequence=["#4C72B0", "#DD8452", "#8C8C8C"],
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        age_counts = df["age_group"].value_counts().reindex(AGE_ORDER).dropna()
        fig = px.bar(
            x=age_counts.index, y=age_counts.values,
            labels={"x": "Age group", "y": "Number of records"},
            title="Victims by Age Group",
            color=age_counts.values, color_continuous_scale="Blues",
        )
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    insight_box(
        f"Men make up roughly **{MALE_PCT:.0f}%** of recorded victims, and "
        f"working-age adults dominate the age distribution — consistent "
        f"with commuting and income-generating travel being the riskiest "
        f"activity on the road."
    )

    st.write("")
    vehicle_totals = df[VEHICLE_COLS].sum().sort_values(ascending=False)
    vehicle_totals.index = [v.replace("_", " ").title()
                            for v in vehicle_totals.index]
    fig = px.bar(
        x=vehicle_totals.values, y=vehicle_totals.index, orientation="h",
        labels={"x": "Number of records involving this vehicle", "y": ""},
        title="Vehicle Types Most Frequently Involved",
        color=vehicle_totals.values, color_continuous_scale="OrRd",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    insight_box(
        f"**{TOP_VEHICLE}s** appear in more accident records than any "
        f"other vehicle type — worth reading alongside the victim-type "
        f"chart above, since the two are closely linked."
    )

    st.write("")
    fig = px.bar(
        x=severity_buckets.index, y=severity_buckets.values,
        labels={"x": "Victims in the crash", "y": "Number of crash events"},
        title="Crash Severity: Victims per Distinct Crash Event",
        color=severity_buckets.index,
        color_discrete_sequence=px.colors.sequential.Reds[2:],
        text=severity_buckets.values.astype(int),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    insight_box(
        f"About **{MASS_CASUALTY_PCT:.1f}%** of distinct crash events "
        f"involve 4 or more victims. Most crashes are single-victim, but "
        f"the mass-casualty minority carries disproportionate cost and "
        f"deserves separate attention in emergency-response planning."
    )


# ==========================================================================
# 5. WHEN
# ==========================================================================
elif section == "When":
    section_header(
        "When — Timing Patterns",
        "The hours, days, and months where risk concentrates.",
    )

    risk_clock = pd.crosstab(df["day_name_title"], df["time_period"])
    risk_clock = risk_clock.reindex(index=DAY_ORDER, columns=PERIOD_ORDER)
    fig = px.imshow(
        risk_clock, text_auto=True, color_continuous_scale="YlOrRd",
        labels=dict(x="Time of day", y="Day of week", color="Records"),
        title="Risk Clock — Accidents by Day and Time of Day",
        aspect="auto",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
    insight_box(
        f"The single riskiest slot is **{RISKIEST_DAY} {RISKIEST_PERIOD.title()}**, "
        f"with {RISKIEST_N} recorded victims — a clear target window for "
        f"visible enforcement, checkpoints, or public-awareness pushes."
    )

    st.write("")
    hourly = df["hour"].dropna().astype(int).value_counts().sort_index()
    fig = px.line(
        x=hourly.index, y=hourly.values, markers=True,
        labels={"x": "Hour of day (24h)", "y": "Number of records"},
        title="Accidents by Hour of Day",
    )
    fig.update_traces(line_color=ACCENT2, marker=dict(size=6))
    fig.update_layout(height=400, xaxis=dict(dtick=1))
    st.plotly_chart(fig, use_container_width=True)
    peak_hour = int(hourly.idxmax())
    insight_box(
        f"Accidents rise sharply through the day and peak around "
        f"**{peak_hour}:00**, consistent with evening rush-hour traffic and "
        f"fading daylight rather than late-night driving alone."
    )

    st.write("")
    monthly = df["month_name_title"].value_counts().reindex(MONTH_ORDER)
    fig = px.bar(
        x=monthly.index, y=monthly.values,
        labels={"x": "Month", "y": "Number of records"},
        title="Seasonality — Accidents by Month",
        color=monthly.values, color_continuous_scale="Magma",
    )
    fig.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    insight_box(
        "Monthly totals fluctuate but don't show one dominant season — "
        "timing interventions around the weekly *risk clock* and daily "
        "peak hour above will likely have more impact than seasonal "
        "campaigns alone."
    )


# ==========================================================================
# 6. WHY
# ==========================================================================
elif section == "Why":
    section_header(
        "Why — Recorded Causes",
        "What's actually driving these crashes, and where effort pays off most.",
    )

    cause_counts = df["cause_group"].value_counts()
    fig = px.bar(
        x=cause_counts.values, y=cause_counts.index, orientation="h",
        labels={"x": "Number of records", "y": ""},
        title="Most Common Recorded Accident Causes",
        color=cause_counts.values, color_continuous_scale="Teal",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=450,
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    insight_box(
        f"**{TOP_CAUSE.title()}** is the leading recorded cause, "
        f"accounting for **{TOP_CAUSE_PCT:.0f}%** of all records — human "
        f"behaviour behind the wheel (or handlebars) outweighs mechanical "
        f"or road-condition factors in this dataset."
    )

    st.write("")
    top_n = 10
    counts = df["cause_group"].value_counts().head(top_n)
    cum_pct = counts.cumsum() / df["cause_group"].value_counts().sum() * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(x=counts.index, y=counts.values, name="Records",
                         marker_color=ACCENT))
    fig.add_trace(go.Scatter(x=counts.index, y=cum_pct.values, name="Cumulative %",
                             yaxis="y2", mode="lines+markers",
                             line=dict(color=ACCENT2, width=3)))
    fig.add_hline(y=80, line_dash="dash", line_color="grey", yref="y2")
    fig.update_layout(
        title="Pareto View — Causes Explaining 80% of Accidents",
        yaxis=dict(title="Number of records"),
        yaxis2=dict(title="Cumulative %", overlaying="y", side="right",
                    range=[0, 110]),
        height=450, legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    insight_box(
        f"Roughly the **top {N_CAUSES_80} cause categories** already "
        f"account for about 80% of all recorded accidents. This is the "
        f"clearest resource-allocation signal in the whole dataset: fixing "
        f"a handful of root causes goes a long way further than trying to "
        f"address every category equally."
    )

    st.write("")
    top10_counties = df["county"].value_counts().head(10).index
    cross = pd.crosstab(
        df[df["county"].isin(top10_counties)]["county"],
        df[df["county"].isin(top10_counties)]["cause_group"],
    )
    fig = px.imshow(
        cross, text_auto=True, color_continuous_scale="YlGnBu",
        labels=dict(x="Cause group", y="County", color="Records"),
        title="Which Causes Dominate in Which Top Counties",
        aspect="auto",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    insight_box(
        "Cause patterns aren't identical across counties — some places "
        "lean heavily toward one or two dominant causes, which means "
        "county-specific messaging will outperform one national campaign."
    )


# ==========================================================================
# 7. CONCLUSION
# ==========================================================================
elif section == "Conclusion":
    section_header(
        "Conclusion",
        "Pulling the story together.",
    )

    st.markdown(
        f"""
Across **{TOTAL_RECORDS:,} recorded victims** in 2017, a clear and
actionable pattern emerges rather than a random spread of incidents:

- **Who:** {TOP_VICTIM.title()}s are the most affected road-user group
  ({TOP_VICTIM_PCT:.0f}% of records), and men make up roughly
  {MALE_PCT:.0f}% of victims — largely working-age adults.
- **What:** {TOP_VEHICLE}s are the vehicle type most often involved, and
  about {MASS_CASUALTY_PCT:.1f}% of distinct crashes are mass-casualty
  events (4+ victims).
- **When:** Risk peaks around **{RISKIEST_DAY} {RISKIEST_PERIOD.title()}**,
  and daily accidents build toward an evening peak — this is a *when*
  problem as much as a *where* problem.
- **Where:** **{TOP_COUNTY.title()}** county and the **{TOP_ROAD.title()}**
  road stand out as the single largest contributors, with a small number of
  specific county–road corridors accounting for a disproportionate share of
  all incidents.
- **Why:** **{TOP_CAUSE.title()}** is the dominant recorded cause
  ({TOP_CAUSE_PCT:.0f}%), and a small set of top causes explains roughly
  80% of all accidents — a textbook case for focused, rather than broad,
  intervention.
        """
    )

    insight_box(
        "The story consistently points toward **concentration, not "
        "randomness**: a handful of user groups, locations, time windows, "
        "and causes explain most of the burden. That concentration is what "
        "makes the recommendations on the next page realistic to act on."
    )


# ==========================================================================
# 8. RECOMMENDATIONS
# ==========================================================================
elif section == "Recommendations":
    section_header(
        "Recommendations",
        "Where to focus limited road-safety resources for the greatest impact.",
    )

    st.markdown(
        f"""
**1. Target enforcement and awareness campaigns by time, not just place.**
Concentrate traffic police visibility, breathalyser checks, and public
messaging around **{RISKIEST_DAY} {RISKIEST_PERIOD.title()}**, and the
build-up toward the daily evening peak — this single window carries a
disproportionate share of recorded incidents.

**2. Prioritise {TOP_COUNTY.title()} county and the {TOP_ROAD.title()}
road corridor for infrastructure review.**
Engineering fixes (speed bumps, lighting, signage, pedestrian crossings)
along the top county–road hotspots identified in the *Where* section will
likely outperform county-wide measures, since incidents cluster tightly on
specific corridors rather than spreading evenly.

**3. Make {TOP_VICTIM.title()} safety a named priority, not a byproduct of
general road safety.**
Since {TOP_VICTIM.title()}s account for {TOP_VICTIM_PCT:.0f}% of victims,
interventions aimed specifically at this group — protective infrastructure,
targeted training, or vehicle-specific regulation for {TOP_VEHICLE.lower()}s
— should be evaluated on their own, not folded into generic campaigns.

**4. Attack the top {N_CAUSES_80} recorded causes first.**
Because a small number of cause categories (led by {TOP_CAUSE.title()})
explain roughly 80% of accidents, resources spent addressing that short
list — through enforcement, driver training, or road-condition fixes — will
outperform efforts spread evenly across every possible cause.

**5. Build a dedicated response plan for mass-casualty crashes.**
With roughly {MASS_CASUALTY_PCT:.1f}% of distinct crashes involving 4 or
more victims, emergency services should pre-position resources (ambulances,
trauma capacity) along the identified hotspot corridors during the
highest-risk time windows.

        """
    )

    st.success(
        "The End ,",

        icon="✅",
    )
