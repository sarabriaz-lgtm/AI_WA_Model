import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from groq import Groq


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dam Water Availability AI Assistant",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("💧 Dam Water Availability AI Assistant")

st.subheader(
    "AI-Assisted Rainfall–Runoff Assessment for Dam Sites"
)




# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_MODEL = "openai/gpt-oss-20b"

try:

    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

    groq_client = Groq(
        api_key=GROQ_API_KEY
    )

except Exception:

    GROQ_API_KEY = None
    groq_client = None


# ============================================================
# NASA POWER RAINFALL FUNCTION
# ============================================================

def get_daily_rainfall(
    latitude,
    longitude,
    start_date,
    end_date
):

    start_str = pd.to_datetime(
        start_date
    ).strftime("%Y%m%d")

    end_str = pd.to_datetime(
        end_date
    ).strftime("%Y%m%d")

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        "?parameters=PRECTOTCORR"
        "&community=AG"
        f"&longitude={longitude}"
        f"&latitude={latitude}"
        f"&start={start_str}"
        f"&end={end_str}"
        "&format=JSON"
    )

    response = requests.get(
        url,
        timeout=60
    )

    if response.status_code != 200:

        raise Exception(
            f"NASA POWER API error: "
            f"{response.status_code} - "
            f"{response.text[:500]}"
        )

    data = response.json()

    rainfall_data = (
        data["properties"]
        ["parameter"]
        ["PRECTOTCORR"]
    )

    rainfall_df = pd.DataFrame(
        list(rainfall_data.items()),
        columns=[
            "Date",
            "Rainfall_mm"
        ]
    )

    rainfall_df["Date"] = pd.to_datetime(
        rainfall_df["Date"]
    )

    rainfall_df["Rainfall_mm"] = pd.to_numeric(
        rainfall_df["Rainfall_mm"],
        errors="coerce"
    )

    rainfall_df = rainfall_df.dropna()

    rainfall_df = rainfall_df.sort_values(
        "Date"
    ).reset_index(drop=True)

    return rainfall_df


# ============================================================
# SCS CURVE NUMBER CALCULATION
# ============================================================

def calculate_scs_cn(
    rainfall_df,
    curve_number,
    catchment_area_km2
):

    CN = float(curve_number)

    if CN <= 0 or CN >= 100:

        raise ValueError(
            "Curve Number must be between 0 and 100."
        )

    if catchment_area_km2 <= 0:

        raise ValueError(
            "Catchment area must be greater than zero."
        )

    # Potential maximum retention
    S = (25400 / CN) - 254

    # Initial abstraction
    Ia = 0.2 * S

    df = rainfall_df.copy()

    # --------------------------------------------------------
    # SCS-CN DIRECT RUNOFF
    # --------------------------------------------------------

    def calculate_runoff(P):

        if P <= Ia:

            return 0.0

        return (
            (P - Ia) ** 2
            /
            (P - Ia + S)
        )

    df["Runoff_mm"] = (
        df["Rainfall_mm"]
        .apply(calculate_runoff)
    )

    # --------------------------------------------------------
    # RUNOFF VOLUME IN m3
    # --------------------------------------------------------

    # 1 mm over 1 km² = 1,000 m³

    df["Runoff_m3"] = (
        df["Runoff_mm"]
        * catchment_area_km2
        * 1000
    )

    # --------------------------------------------------------
    # RUNOFF VOLUME IN MCM
    # --------------------------------------------------------

    df["Runoff_MCM"] = (
        df["Runoff_m3"]
        / 1_000_000
    )

    # --------------------------------------------------------
    # RUNOFF VOLUME IN ACRE-FEET
    # --------------------------------------------------------

    # 1 acre-foot = 1,233.48184 m³

    df["Runoff_AcreFeet"] = (
        df["Runoff_m3"]
        / 1233.48184
    )

    return df, S, Ia


# ============================================================
# ANNUAL STATISTICS
# ============================================================

def calculate_annual_statistics(df):

    df = df.copy()

    df["Year"] = (
        df["Date"]
        .dt.year
    )

    annual = (
        df.groupby("Year")
        .agg(

            Rainfall_mm=(
                "Rainfall_mm",
                "sum"
            ),

            Runoff_mm=(
                "Runoff_mm",
                "sum"
            ),

            Runoff_MCM=(
                "Runoff_MCM",
                "sum"
            ),

            Runoff_AcreFeet=(
                "Runoff_AcreFeet",
                "sum"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # RUNOFF COEFFICIENT
    # --------------------------------------------------------

    annual["Runoff_Coefficient"] = np.where(

        annual["Rainfall_mm"] > 0,

        annual["Runoff_mm"]
        /
        annual["Rainfall_mm"],

        0
    )

    return annual


# ============================================================
# AI HYDROLOGICAL INTERPRETATION
# ============================================================

def generate_ai_interpretation(
    project_name,
    latitude,
    longitude,
    catchment_area_km2,
    curve_number,
    start_date,
    end_date,
    annual_results,
    total_runoff_mcm,
    total_runoff_acre_feet
):

    if groq_client is None:

        return (
            "⚠️ Groq API key is not configured. "
            "Please add GROQ_API_KEY in Streamlit Secrets."
        )

    prompt = f"""

You are an expert hydrologist specializing in rainfall-runoff
modelling, water resources engineering, and dam water availability.

PROJECT INFORMATION

Project Name:
{project_name}

Location:
Latitude = {latitude}
Longitude = {longitude}

Catchment Area:
{catchment_area_km2} km²

SCS Curve Number:
{curve_number}

Analysis Period:
{start_date} to {end_date}

TOTAL ESTIMATED DIRECT RUNOFF

{total_runoff_mcm:.3f} MCM

{total_runoff_acre_feet:.2f} Acre-feet

ANNUAL RESULTS

{annual_results.to_string(index=False)}

Provide a professional hydrological interpretation.

Discuss:

1. Rainfall characteristics
2. Estimated runoff response
3. Year-to-year variation
4. Relationship between rainfall and runoff
5. Effect of the selected Curve Number
6. Hydrological significance for dam water availability
7. Important limitations of the SCS-CN method

Important technical requirements:

- Do not invent missing information.
- Clearly state that SCS-CN estimates direct runoff.
- Do not claim that the calculated runoff is the final
  dependable reservoir yield.
- Mention evaporation, seepage, transmission losses,
  environmental flows, abstractions, reservoir storage,
  routing and operating rules where relevant.
- Keep the response professional and suitable for an
  engineering report.
"""

    try:

        response = groq_client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": (
                        "You are an expert hydrologist and "
                        "water resources engineer. Provide "
                        "technically sound and professional "
                        "hydrological interpretations."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,

            max_tokens=1200
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return (
            f"⚠️ AI interpretation error: {str(e)}"
        )


# ============================================================
# AI HYDROLOGY CHATBOT
# ============================================================

def ask_ai_chatbot(
    question,
    project_context
):

    if groq_client is None:

        return (
            "⚠️ Groq API key is not configured. "
            "Please add GROQ_API_KEY in Streamlit Secrets."
        )

    prompt = f"""

You are an AI hydrology assistant helping a civil engineer
and hydrologist with a dam water availability assessment.

PROJECT CONTEXT

{project_context}

USER QUESTION

{question}

Provide a technically sound answer using principles of:

- Hydrology
- Rainfall-runoff modelling
- SCS Curve Number
- Water resources engineering
- Dam water availability

Instructions:

- Do not invent data.
- Clearly state assumptions.
- If information is insufficient, say so.
- Keep the answer practical and technically understandable.
"""

    try:

        response = groq_client.chat.completions.create(

            model=GROQ_MODEL,

            messages=[

                {
                    "role": "system",
                    "content": (
                        "You are an expert hydrologist "
                        "specializing in rainfall-runoff "
                        "modelling, SCS-CN, dam water "
                        "availability and water resources "
                        "engineering."
                    )
                },

                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.3,

            max_tokens=800
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return (
            f"⚠️ Chatbot error: {str(e)}"
        )


# ============================================================
# SIDEBAR - PROJECT INPUTS
# ============================================================

st.sidebar.header("⚙️ Project Inputs")

project_name = st.sidebar.text_input(
    "Project Name",
    value="Enter Name of Project"
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=35.920000,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=74.300000,
    format="%.6f"
)

catchment_area_km2 = st.sidebar.number_input(
    "Catchment Area (km²)",
    min_value=0.01,
    value=100.0,
    step=1.0
)

curve_number = st.sidebar.number_input(
    "SCS Curve Number",
    min_value=1.0,
    max_value=99.0,
    value=72.0,
    step=1.0
)

st.sidebar.subheader(
    "📅 Rainfall Analysis Period"
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.Timestamp("2000-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    value=pd.Timestamp("2025-12-31")
)


# ============================================================
# CALCULATE BUTTON
# ============================================================

calculate_button = st.sidebar.button(
    "🚀 Calculate Water Availability",
    use_container_width=True
)


# ============================================================
# CALCULATION PROCESS
# ============================================================

if calculate_button:

    # --------------------------------------------------------
    # DATE VALIDATION
    # --------------------------------------------------------

    if start_date >= end_date:

        st.error(
            "❌ Start date must be earlier than end date."
        )

        st.stop()

    # --------------------------------------------------------
    # LATITUDE VALIDATION
    # --------------------------------------------------------

    if latitude < -90 or latitude > 90:

        st.error(
            "❌ Latitude must be between -90 and 90 degrees."
        )

        st.stop()

    # --------------------------------------------------------
    # LONGITUDE VALIDATION
    # --------------------------------------------------------

    if longitude < -180 or longitude > 180:

        st.error(
            "❌ Longitude must be between -180 and 180 degrees."
        )

        st.stop()

    # --------------------------------------------------------
    # RAINFALL DATA
    # --------------------------------------------------------

    with st.spinner(
        "🌧️ Retrieving daily rainfall from NASA POWER..."
    ):

        try:

            rainfall_df = get_daily_rainfall(

                latitude,

                longitude,

                start_date,

                end_date
            )

        except Exception as e:

            st.error(
                f"❌ Rainfall data retrieval failed: {e}"
            )

            st.stop()

    if rainfall_df.empty:

        st.error(
            "❌ No rainfall data were returned for "
            "the selected period."
        )

        st.stop()

    # --------------------------------------------------------
    # SCS-CN CALCULATION
    # --------------------------------------------------------

    with st.spinner(
        "🌊 Calculating SCS-CN direct runoff..."
    ):

        try:

            runoff_df, S, Ia = calculate_scs_cn(

                rainfall_df,

                curve_number,

                catchment_area_km2
            )

        except Exception as e:

            st.error(
                f"❌ Runoff calculation failed: {e}"
            )

            st.stop()

    # --------------------------------------------------------
    # ANNUAL STATISTICS
    # --------------------------------------------------------

    annual = calculate_annual_statistics(
        runoff_df
    )

    # --------------------------------------------------------
    # TOTAL RESULTS
    # --------------------------------------------------------

    total_rainfall_mm = (
        runoff_df["Rainfall_mm"]
        .sum()
    )

    total_runoff_mm = (
        runoff_df["Runoff_mm"]
        .sum()
    )

    total_runoff_m3 = (
        runoff_df["Runoff_m3"]
        .sum()
    )

    total_runoff_mcm = (
        runoff_df["Runoff_MCM"]
        .sum()
    )

    total_runoff_acre_feet = (
        runoff_df["Runoff_AcreFeet"]
        .sum()
    )

    # --------------------------------------------------------
    # RUNOFF COEFFICIENT
    # --------------------------------------------------------

    if total_rainfall_mm > 0:

        overall_runoff_coefficient = (
            total_runoff_mm
            /
            total_rainfall_mm
        )

    else:

        overall_runoff_coefficient = 0

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    st.session_state["runoff_df"] = runoff_df

    st.session_state["annual"] = annual

    st.session_state["project_name"] = project_name

    st.session_state["latitude"] = latitude

    st.session_state["longitude"] = longitude

    st.session_state[
        "catchment_area_km2"
    ] = catchment_area_km2

    st.session_state[
        "curve_number"
    ] = curve_number

    st.session_state[
        "start_date"
    ] = start_date

    st.session_state[
        "end_date"
    ] = end_date

    st.session_state[
        "total_runoff_mcm"
    ] = total_runoff_mcm

    st.session_state[
        "total_runoff_acre_feet"
    ] = total_runoff_acre_feet

    st.session_state[
        "overall_runoff_coefficient"
    ] = overall_runoff_coefficient

    st.session_state[
        "total_rainfall_mm"
    ] = total_rainfall_mm

    st.session_state[
        "total_runoff_mm"
    ] = total_runoff_mm

    st.success(
        "✅ Water availability calculation completed successfully."
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "runoff_df" in st.session_state:

    runoff_df = st.session_state[
        "runoff_df"
    ]

    annual = st.session_state[
        "annual"
    ]

    project_name = st.session_state[
        "project_name"
    ]

    latitude = st.session_state[
        "latitude"
    ]

    longitude = st.session_state[
        "longitude"
    ]

    catchment_area_km2 = st.session_state[
        "catchment_area_km2"
    ]

    curve_number = st.session_state[
        "curve_number"
    ]

    start_date = st.session_state[
        "start_date"
    ]

    end_date = st.session_state[
        "end_date"
    ]

    total_rainfall_mm = st.session_state[
        "total_rainfall_mm"
    ]

    total_runoff_mm = st.session_state[
        "total_runoff_mm"
    ]

    total_runoff_mcm = st.session_state[
        "total_runoff_mcm"
    ]

    total_runoff_acre_feet = st.session_state[
        "total_runoff_acre_feet"
    ]

    overall_runoff_coefficient = (
        st.session_state[
            "overall_runoff_coefficient"
        ]
    )


    # ========================================================
    # PROJECT SUMMARY
    # ========================================================

    st.header("📋 Project Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.write(
            f"**Project:** {project_name}"
        )

        st.write(
            f"**Latitude:** {latitude:.6f}"
        )

        st.write(
            f"**Longitude:** {longitude:.6f}"
        )

    with summary_col2:

        st.write(
            f"**Catchment Area:** "
            f"{catchment_area_km2:,.2f} km²"
        )

        st.write(
            f"**SCS Curve Number:** "
            f"{curve_number:.0f}"
        )

    with summary_col3:

        st.write(
            f"**Analysis Start:** "
            f"{start_date}"
        )

        st.write(
            f"**Analysis End:** "
            f"{end_date}"
        )


    # ========================================================
    # WATER AVAILABILITY RESULTS
    # ========================================================

    st.header("💧 Water Availability Results")

    # --------------------------------------------------------
    # FOUR MAIN METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Rainfall",
            f"{total_rainfall_mm:,.1f} mm"
        )

    with col2:

        st.metric(
            "Runoff Depth",
            f"{total_runoff_mm:,.1f} mm"
        )

    with col3:

        st.metric(
            "Runoff Volume (MCM)",
            f"{total_runoff_mcm:,.2f}"
        )

    with col4:

        st.metric(
            "Runoff Volume (AF)",
            f"{total_runoff_acre_feet:,.2f}"
        )

    # --------------------------------------------------------
    # RUNOFF COEFFICIENT
    # --------------------------------------------------------

    st.write("")

    coefficient_col1, coefficient_col2, coefficient_col3 = (
        st.columns(3)
    )

    with coefficient_col1:

        st.metric(
            "Runoff Coefficient",
            f"{overall_runoff_coefficient:.3f}"
        )

    with coefficient_col2:

        st.metric(
            "Catchment Area",
            f"{catchment_area_km2:,.2f} km²"
        )

    with coefficient_col3:

        st.metric(
            "Curve Number",
            f"{curve_number:.0f}"
        )


    # ========================================================
    # SCS PARAMETERS
    # ========================================================

    st.subheader(
        "SCS Curve Number Parameters"
    )

    S = (
        (25400 / curve_number)
        - 254
    )

    Ia = 0.2 * S

    scs_col1, scs_col2 = st.columns(2)

    with scs_col1:

        st.metric(
            "Potential Maximum Retention (S)",
            f"{S:.2f} mm"
        )

    with scs_col2:

        st.metric(
            "Initial Abstraction (Ia)",
            f"{Ia:.2f} mm"
        )


    # ========================================================
    # ENGINEERING NOTE
    # ========================================================

    st.info(
        """
        **Engineering Note:** The calculated runoff represents
        estimated **direct runoff using the SCS Curve Number method**.
        It should not automatically be interpreted as the final
        dependable reservoir yield. A detailed dam water availability
        assessment may also require evaporation, seepage, transmission
        losses, environmental flows, abstractions, reservoir storage,
        routing and operating rules.
        """
    )


    # ========================================================
    # ANNUAL RESULTS
    # ========================================================

    st.header(
        "📊 Annual Water Availability"
    )

    display_annual = annual.copy()

    display_annual["Rainfall_mm"] = (
        display_annual["Rainfall_mm"]
        .round(2)
    )

    display_annual["Runoff_mm"] = (
        display_annual["Runoff_mm"]
        .round(2)
    )

    display_annual["Runoff_MCM"] = (
        display_annual["Runoff_MCM"]
        .round(3)
    )

    display_annual["Runoff_AcreFeet"] = (
        display_annual["Runoff_AcreFeet"]
        .round(2)
    )

    display_annual["Runoff_Coefficient"] = (
        display_annual["Runoff_Coefficient"]
        .round(3)
    )

    st.dataframe(
        display_annual,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # RAINFALL GRAPH
    # ========================================================

    st.header(
        "🌧️ Rainfall Analysis"
    )

    fig1, ax1 = plt.subplots(
        figsize=(12, 5)
    )

    ax1.plot(
        runoff_df["Date"],
        runoff_df["Rainfall_mm"],
        linewidth=1
    )

    ax1.set_xlabel(
        "Date"
    )

    ax1.set_ylabel(
        "Daily Rainfall (mm)"
    )

    ax1.set_title(
        "Daily Rainfall"
    )

    ax1.grid(
        True,
        alpha=0.3
    )

    fig1.tight_layout()

    st.pyplot(fig1)

    plt.close(fig1)


    # ========================================================
    # DAILY RUNOFF GRAPH
    # ========================================================

    st.header(
        "🌊 Direct Runoff Analysis"
    )

    fig2, ax2 = plt.subplots(
        figsize=(12, 5)
    )

    ax2.plot(
        runoff_df["Date"],
        runoff_df["Runoff_mm"],
        linewidth=1
    )

    ax2.set_xlabel(
        "Date"
    )

    ax2.set_ylabel(
        "Daily Runoff (mm)"
    )

    ax2.set_title(
        "Daily SCS-CN Direct Runoff"
    )

    ax2.grid(
        True,
        alpha=0.3
    )

    fig2.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)


    # ========================================================
    # ANNUAL RUNOFF GRAPH
    # ========================================================

    st.header(
        "📈 Annual Runoff Volume"
    )

    fig3, ax3 = plt.subplots(
        figsize=(12, 5)
    )

    ax3.bar(
        annual["Year"].astype(str),
        annual["Runoff_MCM"]
    )

    ax3.set_xlabel(
        "Year"
    )

    ax3.set_ylabel(
        "Runoff Volume (MCM)"
    )

    ax3.set_title(
        "Annual Direct Runoff"
    )

    ax3.tick_params(
        axis="x",
        rotation=90
    )

    ax3.grid(
        axis="y",
        alpha=0.3
    )

    fig3.tight_layout()

    st.pyplot(fig3)

    plt.close(fig3)


    # ========================================================
    # DOWNLOAD RESULTS
    # ========================================================

    st.header(
        "📥 Download Results"
    )

    daily_csv = (
        runoff_df
        .to_csv(index=False)
        .encode("utf-8")
    )

    annual_csv = (
        annual
        .to_csv(index=False)
        .encode("utf-8")
    )

    download_col1, download_col2 = st.columns(2)

    with download_col1:

        st.download_button(

            label="⬇️ Download Daily Results",

            data=daily_csv,

            file_name=(
                "dam_water_availability_daily.csv"
            ),

            mime="text/csv",

            use_container_width=True
        )

    with download_col2:

        st.download_button(

            label="⬇️ Download Annual Results",

            data=annual_csv,

            file_name=(
                "dam_water_availability_annual.csv"
            ),

            mime="text/csv",

            use_container_width=True
        )


    # ========================================================
    # AI HYDROLOGICAL INTERPRETATION
    # ========================================================

    st.header(
        "🤖 AI Hydrological Interpretation"
    )

    st.write(
        """
        Use Generative AI to interpret the rainfall-runoff results
        from a hydrological and water-resources engineering perspective.
        """
    )

    if st.button(
        "🧠 Generate AI Interpretation",
        use_container_width=True
    ):

        with st.spinner(
            "🤖 AI is analyzing the hydrological results..."
        ):

            interpretation = (
                generate_ai_interpretation(

                    project_name,

                    latitude,

                    longitude,

                    catchment_area_km2,

                    curve_number,

                    start_date,

                    end_date,

                    annual,

                    total_runoff_mcm,

                    total_runoff_acre_feet
                )
            )

        st.markdown(
            interpretation
        )


    # ========================================================
    # AI HYDROLOGY CHATBOT
    # ========================================================

    st.header(
        "💬 Ask the Hydrology AI Assistant"
    )

    project_context = f"""

Project Name:
{project_name}

Location:
Latitude = {latitude}
Longitude = {longitude}

Catchment Area:
{catchment_area_km2} km²

SCS Curve Number:
{curve_number}

Analysis Period:
{start_date} to {end_date}

Total Rainfall:
{total_rainfall_mm:.2f} mm

Total Direct Runoff:
{total_runoff_mm:.2f} mm

Total Direct Runoff:
{total_runoff_mcm:.3f} MCM

Total Direct Runoff:
{total_runoff_acre_feet:.2f} Acre-feet

Overall Runoff Coefficient:
{overall_runoff_coefficient:.3f}
"""

    question = st.text_input(

        "Enter your hydrological question:",

        placeholder=(
            "Example: How does the Curve Number "
            "affect estimated runoff?"
        )
    )

    if st.button(
        "💬 Ask AI",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            with st.spinner(
                "🤖 AI is preparing the answer..."
            ):

                answer = ask_ai_chatbot(

                    question,

                    project_context
                )

            st.markdown(
                answer
            )


# ============================================================
# INITIAL SCREEN
# ============================================================

else:

    st.info(
        """
        👈 Enter the project information in the sidebar and
        click **Calculate Water Availability** to start the
        assessment.
        """
    )

    st.markdown(
        """
        ## 🔍 Application Workflow

        ### 1. Project Inputs
        Enter the dam coordinates, catchment area and SCS Curve Number.

        ### 2. Rainfall Retrieval
        Daily rainfall is automatically retrieved from NASA POWER.

        ### 3. Rainfall–Runoff Modelling
        The SCS Curve Number method is used to estimate direct runoff.

        ### 4. Water Availability
        Estimated runoff is converted into:

        - Cubic metres (m³)
        - Million Cubic Metres (MCM)
        - Acre-feet (AF)

        ### 5. Hydrological Analysis
        Annual rainfall, runoff volume and runoff coefficients are calculated.

        ### 6. Generative AI
        AI provides hydrological interpretation and answers technical questions.

        ### 7. Data Export
        Daily and annual results can be downloaded as CSV files.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Developed by HYDRO FLOW CODERS💯"
)
