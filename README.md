# 💧 Dam Water Availability AI Assistant

An AI-assisted web application for **preliminary dam-site water availability assessment** using automated rainfall data retrieval, the **SCS Curve Number (SCS-CN)** method, runoff-volume estimation, data visualization, and AI-based hydrological interpretation.

---

## 🚀 Project Overview

The **Dam Water Availability AI Assistant** is designed to simplify the early-stage assessment of water availability at a proposed dam or reservoir site.

Instead of manually collecting rainfall data, processing spreadsheets, calculating runoff, and interpreting results, the application provides an integrated workflow:

**Project Inputs → NASA POWER Rainfall → SCS-CN Runoff → Water Volume → Annual Analysis → Visualization → AI Interpretation**

The tool is intended for **preliminary screening and educational/engineering analysis**, not as a replacement for a detailed hydrological feasibility study.

---

## 🎯 Problem Statement

Preliminary dam-site assessment requires rainfall analysis and runoff estimation to understand the potential availability of water within a catchment.

Traditional workflows can involve:

- Manual rainfall-data collection
- Spreadsheet-based calculations
- Repetitive runoff calculations
- Manual graph preparation
- Difficulty interpreting hydrological results

This project provides a simple web-based solution that automates these steps and adds an AI assistant for easier interpretation.

---

## 💡 Solution

The application allows users to enter basic catchment and project information and automatically performs the preliminary analysis.

### User provides:

- Project Name
- Latitude
- Longitude
- Catchment Area (km²)
- SCS Curve Number
- Rainfall Start Date
- Rainfall End Date

### The application then:

1. Retrieves daily rainfall data.
2. Calculates daily direct runoff.
3. Converts runoff depth into water volume.
4. Generates annual statistics.
5. Creates rainfall and runoff graphs.
6. Provides AI-assisted hydrological interpretation.
7. Allows users to ask questions about the results.
8. Provides downloadable CSV data.

---

## 🏗️ System Workflow

```text
             USER INPUT
                 │
                 ▼
       ┌─────────────────────┐
       │ Project Information │
       │ Lat / Long          │
       │ Catchment Area      │
       │ Curve Number        │
       │ Analysis Period     │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │   NASA POWER API    │
       │   Daily Rainfall    │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │    SCS-CN Method    │
       │  Daily Runoff (Q)   │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Runoff Volume       │
       │ m³ / MCM / Acre-ft  │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ Annual Statistics   │
       │ Tables & Graphs     │
       └──────────┬──────────┘
                  │
                  ▼
       ┌─────────────────────┐
       │ AI Interpretation   │
       │ & AI Q&A Assistant  │
       └─────────────────────┘
