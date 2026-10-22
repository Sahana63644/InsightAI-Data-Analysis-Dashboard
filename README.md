# InsightAI – AI Data Analysis Dashboard

An interactive data analysis dashboard built with Python and Streamlit to help users explore datasets, clean data, visualize patterns, and generate useful insights.

## Features

- **Multiple file formats:** Upload CSV, Excel (`.xlsx`, `.xls`), and JSON files.
- **Dataset overview:** View row counts, column counts, data types, and descriptive statistics.
- **Data cleaning:** Identify and handle missing values and duplicate records.
- **Interactive filters:** Filter datasets using selected column values.
- **Data visualization:** Create interactive charts, including histograms, bar charts, scatter plots, line charts, box plots, pie charts, area charts, treemaps, and correlation heatmaps.
- **Automated insights:** Generate rule-based summaries and observations from the uploaded data.
- **Report downloads:** Export cleaned datasets and analysis reports.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- OpenPyXL
- xlrd

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sahana63644/InsightAI-Data-Analysis-Dashboard.git
cd InsightAI-Data-Analysis-Dashboard
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## How to Use

1. Launch the dashboard.
2. Upload a CSV, Excel, or JSON dataset.
3. Review the dataset overview and column information.
4. Clean missing values and duplicate records as needed.
5. Apply filters to focus on specific data.
6. Explore visualizations and automatically generated insights.
7. Download the cleaned dataset or analysis report.

## Project Objective

The objective of InsightAI is to simplify exploratory data analysis by combining data cleaning, interactive visualization, filtering, and automated insights in one user-friendly interface.

## Future Improvements

- Integrate a large language model for natural-language data questions.
- Add more advanced anomaly detection and statistical analysis.
- Support scheduled reports and additional export formats.
- Add automated machine learning recommendations.

## Author

**Sahana B.**

GitHub: https://github.com/Sahana63644

## Disclaimer

Automated insights are intended to support exploratory analysis. Verify important conclusions against the underlying data before making decisions.
