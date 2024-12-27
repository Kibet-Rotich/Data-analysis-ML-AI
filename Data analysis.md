# World Bank Data Analysis - Kenya

This repository contains two Jupyter notebooks: `worldbank_data.ipynb` and `kenya.ipynb`. These notebooks analyze World Bank data with a specific focus on Kenya, covering data extraction, cleaning, transformation, and analysis.

## Project Objectives
- Extract and clean raw World Bank data.
- Perform data aggregation and transformation for exploratory analysis.
- Analyze key economic, military, and demographic trends for Kenya.
- Identify correlations and lagged effects to uncover potential relationships between metrics.

---

## Notebook 1: `worldbank_data.ipynb`

### **Overview**
This notebook processes World Bank datasets to filter and analyze global data, with a focus on Kenya.

### **Steps**

1. **Data Extraction:**
   - Loaded two datasets: `data1.csv` and `data2.csv`.
   - Extracted key series and unique indicators for further analysis.
   - Isolated `"Military expenditure (% of GDP)"` for all countries and saved it as `military_expenditure_data.xlsx`.

2. **Data Cleaning:**
   - Removed unnecessary rows and columns.
   - Cleaned indicator names and converted year columns to numerical data.

3. **Data Aggregation:**
   - Filtered data for Kenya.
   - Transformed year columns into rows (tidy data format).
   - Generated a correlation heatmap for Kenyan indicators.

4. **Analysis:**
   - Visualized Kenya's military expenditure trends (2000-2015) with line graphs.
   - Created a heatmap to analyze the relationships between indicators for Kenya.

---

## Notebook 2: `kenya.ipynb`

### **Overview**
Building on the first notebook, this analysis focuses exclusively on Kenya's data for a deeper exploration.

### **Steps**

1. **Data Cleaning and Refinement:**
   - Loaded `kenya_data1.xlsx` and replaced missing values (`..`) with `pd.NA`.
   - Dropped columns with excessive missing values.
   - Forward-filled and zero-filled remaining missing values.
   - Renamed `"parsed year"` to `"Year"` for consistency.

2. **Data Transformation:**
   - Calculated correlation matrices for Kenya's indicators.
   - Filtered correlations greater than `0.5` for visualization in heatmaps.
   - Extracted highly correlated columns (`≥ 0.85`) and created a structured DataFrame.

3. **Trend Visualization:**
   - Plotted key indicators such as population growth, GNI, life expectancy, and fertility rates over time.
   - Generated detailed trend visualizations for specific metrics.

4. **Lagged Correlation Analysis:**
   - Analyzed lagged relationships between GDP growth (5 years lag) and `"Military expenditure (% of GDP)"`.
   - Visualized the correlation and relationship using scatter plots.

---

## Key Findings
- **Military Expenditure Trends:** Visualized Kenya's military spending and its relationship to GDP growth.
- **Economic Indicators:** Highlighted correlations between GNI, life expectancy, and fertility rates over time.
- **Lagged Effects:** Discovered potential delayed effects of GDP growth on military expenditure.

---

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/worldbank-kenya-analysis.git
2. Open the notebooks in Jupyter or Google Colab:
    - [`worldbank_data.ipynb`](worldbank_data.ipynb)
    - [`kenya.ipynb`](kenya.ipynb)
3. Explore the cleaned datasets:
    - `kenya_cleaned.xlsx`
    - `highly_correlated_df_cleaned.xlsx`

---

## Dependencies

- **Python**: 3.8+
- **Required Libraries**:
    - `pandas`
    - `numpy`
    - `matplotlib`
    - `seaborn`
    - `openpyxl`

### Install Dependencies
Use the following command to install the required libraries:
```bash
pip install pandas numpy matplotlib seaborn openpyxl
