import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Dam Water Availability AI",
    page_icon="💧",
    layout="wide"
)


# =========================================================
# GROQ CONFIGURATION
# =========================================================

GROQ_MODEL = "openai/gpt-oss-20b"

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)
else:
    groq_client = None


# =========================================================
# FUNCTIONS
# =========================================================

def get_daily_rainfall(latitude, longitude, start_date, end_date):

    start_str = pd.to_datetime(start_date).strftime("%Y%m%d")
    end_str = pd.to_datetime(end_date).strftime("%Y%m%d")

    url = (
        "https://power.larc.nasa.gov/api/temporal/daily/point"
        f"?parameters=PRECTOTCORR"
        f"&community=AG"
        f"&longitude={longitude}"
        f"&latitude={latitude}"
        f"&start={start_str}"
        f"&end={end_str}"
        f"&format=JSON"
    )

    response = requests.get(url, timeout=60)

    if response.status_code != 200:
        raise Exception(
            f"NASA POWER API error: "
            f"{response.status_code}"
        )

    data = response.json()

    rainfall_data = (
        data["properties"]
        ["parameter"]
        ["PRECTOTCORR"]
    )

    df = pd.DataFrame(
        list(rainfall_data.items()),
        columns=["Date", "Rainfall_mm"]
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df["Rainfall_mm"] = pd.to_numeric(
        df["Rainfall_mm"],
        errors="coerce"
    )

    df = df.dropna()

    df = df.sort_values("Date").reset_index(drop=True)

    return df


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

    S = (25400 / CN) - 254

    Ia = 0.2 * S

    df = rainfall_df.copy()

    def calculate_runoff(P):

        if P <= Ia:
            return 0.0

        return (
            (P - Ia) ** 2
        ) / (
            P - Ia + S
        )

    df["Runoff_mm"] = (
        df["Rainfall_mm"]
        .apply(calculate_runoff)
    )

    df["Runoff_m3"] = (
        df["Runoff_mm"]
        * catchment_area_km2
        * 1000
    )

    df["Runoff_MCM"] = (
        df["Runoff_m3"] / 1_000_000
    )

    return df, S, Ia


def calculate_annual_statistics(df):

    df = df.copy()

    df["Year"] = df["Date"].dt.year

    annual = (
        df.groupby("Year")
        .agg(
            Rainfall_mm=("Rainfall_mm", "sum"),
            Runoff_mm=("Runoff_mm", "sum"),
            Runoff_MCM=("Runoff_MCM", "sum")
        )
        .reset_index()
    )

    annual["Runoff_Coefficient"] = np.where(
        annual["Rainfall_mm"] > 0,
        annual["Runoff_mm"]
        / annual["Rainfall_mm"],
        0
    )

    return annual


def generate_ai_interpretation(
    project_name,
    latitude,
    longitude,
    catchment_area_km2,
    curve_number,
    start_date,
    end_date,
    annual_results,
    total_runoff_mcm
):

    if groq_client is None:
        return (
            "Groq API key is not configured. "
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )

    prompt = f"""
You are an expert hydrologist specializing in rainfall-runoff
modelling and dam water availability.

Project Name:
{project_name}

Latitude:
{latitude}

Longitude:
{longitude}

Catchment Area:
{catchment_area_km2} km²

Curve Number:
{curve_number}

Analysis Period:
{start_date} to {end_date}

Total Estimated Direct Runoff:
{total_runoff_mcm:.3f} MCM

Annual Results:
{annual_results.to_string(index=False)}

Provide a professional hydrological interpretation covering:

1. Rainfall characteristics
2. Runoff response
3. Year-to-year variation
4. Rainfall-runoff relationship
5. Significance for dam water availability
6. Limitations of the SCS-CN approach

Do not invent data.

Clearly explain that SCS-CN estimates direct runoff and does not
represent final reservoir yield. Mention that reservoir routing,
evaporation, seepage, transmission losses, environmental flows,
abstractions and operating rules may need to be considered.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert hydrologist "
                    "and water resources engineer."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=1000
    )

    return response.choices[0].message.content


def ask_ai_chatbot(question, project_context):

    if groq_client is None:
        return (
            "Groq API key is not configured. "
            "Please add GROQ_API_KEY to Streamlit Secrets."
        )

    prompt = f"""
You are an AI hydrology assistant.

Project information:

{project_context}

User question:

{question}

Answer using sound hydrological and water-resources
engineering principles.

Do not invent missing information.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert hydrologist specializing "
                    "in rainfall-runoff modelling and dam "
                    "water availability."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=700
    )

    return response.choices[0].message.content


# =========================================================
# USER INTERFACE
# =========================================================

st.title("💧 Dam Water Availability AI")

st.write(
    "Estimate rainfall-runoff and potential direct runoff "
    "using NASA POWER rainfall data and the SCS Curve Number method."
)


# =========================================================
# INPUTS
# =========================================================

st.sidebar.header("Project Information")

project_name = st.sidebar.text_input(
    "Project Name",
    value="Example Dam Project"
)

latitude = st.sidebar.number_input(
    "Latitude",
    value=35.9200,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    value=74.3000,
    format="%.6f"
)

catchment_area_km2 = st.sidebar.number_input(
    "Catchment Area (km²)",
    min_value=0.1,
    value=250.0
)

curve_number = st.sidebar.number_input(
    "SCS Curve Number",
    min_value=1.0,
    max_value=99.0,
    value=72.0
)

start_date = st.sidebar.date_input(
    "Analysis Start Date",
    value=pd.Timestamp("2000-01-01")
)

end_date = st.sidebar.date_input(
    "Analysis End Date",
    value=pd.Timestamp("2020-12-31")
)


# =========================================================
# CALCULATE
# =========================================================

if st.sidebar.button(
    "Calculate Water Availability",
    type="primary"
):

    if start_date >= end_date:

        st.error(
            "End date must be later than start date."
        )

    else:

        with st.spinner(
            "Downloading rainfall and calculating runoff..."
        ):

            try:

                rainfall_df = get_daily_rainfall(
                    latitude,
                    longitude,
                    start_date,
                    end_date
                )

                runoff_df, S, Ia = calculate_scs_cn(
                    rainfall_df,
                    curve_number,
                    catchment_area_km2
                )

                annual_results = (
                    calculate_annual_statistics(
                        runoff_df
                    )
                )

                total_rainfall_mm = (
                    runoff_df["Rainfall_mm"].sum()
                )

                total_runoff_mm = (
                    runoff_df["Runoff_mm"].sum()
                )

                total_runoff_mcm = (
                    runoff_df["Runoff_MCM"].sum()
                )

                st.session_state["rainfall_df"] = (
                    rainfall_df
                )

                st.session_state["runoff_df"] = (
                    runoff_df
                )

                st.session_state["annual_results"] = (
                    annual_results
                )

                st.session_state["S"] = S

                st.session_state["Ia"] = Ia

                st.session_state[
                    "total_rainfall_mm"
                ] = total_rainfall_mm

                st.session_state[
                    "total_runoff_mm"
                ] = total_runoff_mm

                st.session_state[
                    "total_runoff_mcm"
                ] = total_runoff_mcm

                st.session_state[
                    "project_name"
                ] = project_name

                st.success(
                    "Water availability calculation completed."
                )

            except Exception as e:

                st.error(
                    f"Calculation error: {e}"
                )


# =========================================================
# RESULTS
# =========================================================

if "runoff_df" in st.session_state:

    runoff_df = st.session_state["runoff_df"]

    annual_results = (
        st.session_state["annual_results"]
    )

    S = st.session_state["S"]

    Ia = st.session_state["Ia"]

    total_rainfall_mm = (
        st.session_state["total_rainfall_mm"]
    )

    total_runoff_mm = (
        st.session_state["total_runoff_mm"]
    )

    total_runoff_mcm = (
        st.session_state["total_runoff_mcm"]
    )


    # =====================================================
    # METRICS
    # =====================================================

    st.subheader("Water Availability Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Rainfall",
        f"{total_rainfall_mm:,.1f} mm"
    )

    col2.metric(
        "Runoff Depth",
        f"{total_runoff_mm:,.1f} mm"
    )

    col3.metric(
        "Estimated Runoff",
        f"{total_runoff_mcm:,.2f} MCM"
    )

    col4.metric(
        "Curve Number",
        f"{curve_number:.0f}"
    )


    # =====================================================
    # SCS PARAMETERS
    # =====================================================

    st.subheader("SCS-CN Parameters")

    col1, col2 = st.columns(2)

    col1.metric(
        "Potential Maximum Retention (S)",
        f"{S:.2f} mm"
    )

    col2.metric(
        "Initial Abstraction (Ia)",
        f"{Ia:.2f} mm"
    )


    # =====================================================
    # ANNUAL RESULTS
    # =====================================================

    st.subheader("Annual Water Availability")

    st.dataframe(
        annual_results,
        use_container_width=True
    )


    # =====================================================
    # RAINFALL GRAPH
    # =====================================================

    st.subheader("Daily Rainfall")

    fig1, ax1 = plt.subplots(
        figsize=(12, 4)
    )

    ax1.plot(
        runoff_df["Date"],
        runoff_df["Rainfall_mm"]
    )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Rainfall (mm)")
    ax1.grid(True)

    st.pyplot(fig1)


    # =====================================================
    # RUNOFF GRAPH
    # =====================================================

    st.subheader("Daily Estimated Runoff")

    fig2, ax2 = plt.subplots(
        figsize=(12, 4)
    )

    ax2.plot(
        runoff_df["Date"],
        runoff_df["Runoff_mm"]
    )

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Runoff (mm)")
    ax2.grid(True)

    st.pyplot(fig2)


    # =====================================================
    # DOWNLOAD DATA
    # =====================================================

    st.subheader("Download Results")

    csv_data = runoff_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Daily Results CSV",
        data=csv_data,
        file_name="dam_water_availability_daily.csv",
        mime="text/csv"
    )

    annual_csv = annual_results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download Annual Results CSV",
        data=annual_csv,
        file_name="dam_water_availability_annual.csv",
        mime="text/csv"
    )


    # =====================================================
    # AI INTERPRETATION
    # =====================================================

    st.subheader("🤖 AI Hydrological Interpretation")

    if st.button("Generate AI Interpretation"):

        with st.spinner(
            "AI is analyzing the hydrological results..."
        ):

            try:

                interpretation = (
                    generate_ai_interpretation(
                        project_name,
                        latitude,
                        longitude,
                        catchment_area_km2,
                        curve_number,
                        start_date,
                        end_date,
                        annual_results,
                        total_runoff_mcm
                    )
                )

                st.session_state[
                    "interpretation"
                ] = interpretation

            except Exception as e:

                st.error(
                    f"AI error: {e}"
                )


    if "interpretation" in st.session_state:

        st.markdown(
            st.session_state["interpretation"]
        )


    # =====================================================
    # AI CHATBOT
    # =====================================================

    st.subheader("💬 Ask the Hydrology AI")

    project_context = f"""
Project Name: {project_name}

Latitude: {latitude}
Longitude: {longitude}

Catchment Area:
{catchment_area_km2} km²

SCS Curve Number:
{curve_number}

Analysis Period:
{start_date} to {end_date}

Total Rainfall:
{total_rainfall_mm:.2f} mm

Total Runoff Depth:
{total_runoff_mm:.2f} mm

Estimated Direct Runoff:
{total_runoff_mcm:.3f} MCM

S:
{S:.2f} mm

Initial Abstraction:
{Ia:.2f} mm
"""


    if "chat_history" not in st.session_state:

        st.session_state["chat_history"] = []


    for message in st.session_state["chat_history"]:

        with st.chat_message(message["role"]):

            st.markdown(
                message["content"]
            )


    user_question = st.chat_input(
        "Ask a question about your dam water availability..."
    )


    if user_question:

        st.session_state["chat_history"].append(
            {
                "role": "user",
                "content": user_question
            }
        )

        with st.chat_message("user"):

            st.markdown(user_question)


        with st.chat_message("assistant"):

            with st.spinner("Analyzing..."):

                try:

                    answer = ask_ai_chatbot(
                        user_question,
                        project_context
                    )

                    st.markdown(answer)

                    st.session_state[
                        "chat_history"
                    ].append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:

                    st.error(
                        f"AI error: {e}"
                    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Dam Water Availability AI | "
    "NASA POWER + SCS Curve Number + Generative AI"
)

