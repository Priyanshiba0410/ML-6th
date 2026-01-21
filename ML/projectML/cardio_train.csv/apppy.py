# import streamlit as st

# st.write("Hello World")
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np


import pandas as pd
import plotly.express as px

# Load your dataset
df = pd.read_csv(r"C:\Users\priya\Downloads\ML\projectML\cardio_train.csv")

# If your CSV has age in days, convert to years (optional)
df['age_years'] = df['age'] / 365

# Plot histogram
fig = px.histogram(df, x='age_years')
fig.show()

# --- Configuration ---
st.set_page_config(
    page_title="Cardiovascular Disease Dataset Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Loads, cleans, and transforms the cardiovascular training data."""
    # Assuming the file is in the same directory and is semicolon-delimited
    df = pd.read_csv("cardio_train.csv", sep=';')

    # Convert age from days to years for better interpretability
    df['age_years'] = (df['age'] / 365.25).round().astype(int)

    # Clean up and map categorical features for plotting
    # Gender (assuming 1: Female, 2: Male based on typical dataset conventions)
    df['gender_mapped'] = df['gender'].map({1: 'Female', 2: 'Male'})

    # Cholesterol and Glucose
    map_level = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
    df['cholesterol_mapped'] = df['cholesterol'].map(map_level)
    df['gluc_mapped'] = df['gluc'].map(map_level)

    # Binary features (0/1 to Yes/No)
    map_binary = {0: 'No', 1: 'Yes'}
    for col in ['smoke', 'alco', 'active', 'cardio']:
        df[f'{col}_mapped'] = df[col].map(map_binary)

    # Basic cleaning for blood pressure (removing extreme outliers/errors)
    # A quick filter to remove ap_hi < ap_lo or extremely high/low values
    df = df[(df['ap_hi'] < 300) & (df['ap_hi'] > 50)]
    df = df[(df['ap_lo'] < 200) & (df['ap_lo'] > 30)]
    df = df[df['ap_hi'] >= df['ap_lo']]


    return df

# Load the data
df = load_data()


# --- Sidebar for Filtering and Controls ---
st.sidebar.header("Explore & Filter")
st.sidebar.markdown("Use the controls below to filter the dataset.")

# Target Variable Filter
cardio_filter = st.sidebar.multiselect(
    "Filter by Cardiovascular Disease (Cardio)",
    options=df['cardio_mapped'].unique(),
    default=df['cardio_mapped'].unique()
)

# Age Slider
age_range = st.sidebar.slider(
    "Age Range (Years)",
    min_value=int(df['age_years'].min()),
    max_value=int(df['age_years'].max()),
    value=(int(df['age_years'].min()), int(df['age_years'].max()))
)

# Apply Filters
df_filtered = df[
    (df['cardio_mapped'].isin(cardio_filter)) &
    (df['age_years'] >= age_range[0]) &
    (df['age_years'] <= age_range[1])
]

# --- Main Dashboard Content ---
st.title("🔬 Cardiovascular Disease Dataset Explorer")
st.markdown("""
This interactive dashboard allows you to explore the distribution of features
in the `cardio_train.csv` dataset, filtered by **Cardio status** and **Age**.
""")
st.write(f"**Displayed Data Rows:** {len(df_filtered)}")
st.divider()

# --- Section 1: Data Overview ---
st.header("1. Data Overview")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Raw Data Sample")
    st.dataframe(df_filtered.head())
with col2:
    st.subheader("Data Shape and Columns")
    st.write(f"**Total Rows (Original):** {len(df)}")
    st.write(f"**Displayed Rows (Filtered):** {len(df_filtered)}")
    st.write(f"**Total Columns:** {df.shape[1]}")
    # Display column information in a more readable format
    st.markdown("""
    | Feature | Type | Description |
    | :--- | :--- | :--- |
    | `age_years` | Numeric | Age in years |
    | `height` | Numeric | Height in cm |
    | `weight` | Numeric | Weight in kg |
    | `ap_hi` | Numeric | Systolic BP |
    | `ap_lo` | Numeric | Diastolic BP |
    | `cholesterol` | Cat. | 1:Normal, 3:Well Above Normal |
    | `cardio` | Binary | Target: 1 (Present), 0 (Absent) |
    """)

st.divider()

# --- Section 2: Distribution of Categorical Features ---
st.header("2. Categorical Feature Distributions")
st.markdown("Distribution of features, separated by **Cardio Status**.")

categorical_cols = ['gender_mapped', 'cholesterol_mapped', 'gluc_mapped', 'smoke_mapped', 'alco_mapped', 'active_mapped']

# Create a container for the plots
plot_container = st.container()

# Plotting the categorical features
num_cols = 3
rows = np.ceil(len(categorical_cols) / num_cols).astype(int)

with plot_container:
    for i in range(rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            index = i * num_cols + j
            if index < len(categorical_cols):
                col_name = categorical_cols[index]
                with cols[j]:
                    fig = px.histogram(
                        df_filtered,
                        x=col_name,
                        color='cardio_mapped',
                        barmode='group',
                        title=f'Distribution of {col_name.replace("_mapped", "").title()}',
                        height=400,
                        color_discrete_map={'No': '#1f77b4', 'Yes': '#d62728'} # Blue for No, Red for Yes
                    )
                    fig.update_layout(xaxis={'categoryorder':'category ascending'})
                    st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- Section 3: Distribution of Continuous Features ---
# --- Section 3: Distribution of Continuous Features (Corrected) ---
st.header("3. Continuous Feature Distributions")
st.markdown("Histograms of continuous features, comparing **Cardio Status**.")

continuous_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo']

# Plotting the continuous features
plot_container_cont = st.container()

with plot_container_cont:
    # Use a maximum of 3 plots per row for better visibility
    num_cols = 3
    rows = np.ceil(len(continuous_cols) / num_cols).astype(int)

    for i in range(rows): 
        cols = st.columns(num_cols)
        for j in range(num_cols):
            index = i * num_cols + j
            if index < len(continuous_cols):
                col_name = continuous_cols[index]
                with cols[j]:
                    fig = px.histogram(
                        df_filtered,
                        x=col_name,
                        color='cardio_mapped',
                        marginal='box', # Add a box plot for better visualization
                        title=f'Distribution of {col_name.title()}',
                        height=400,
                        histnorm='percent', # Show percentages on the y-axis
                        color_discrete_map={'No': '#1f77b4', 'Yes': '#d62728'}
                    )
                    # The problematic fig.update_traces line has been removed.
                    st.plotly_chart(fig, use_container_width=True)