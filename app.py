import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="World Bank Global Development Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown('<h1 class="main-header">🌍 World Bank Global Development Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sample data generation function (replace with actual World Bank data loading)
@st.cache_data
def load_sample_data():
    """Generate sample World Bank-style data for demonstration"""
    countries = ['United States', 'China', 'India', 'Germany', 'Brazil', 'Japan', 'United Kingdom', 
                'France', 'Italy', 'Canada', 'Russia', 'South Korea', 'Australia', 'Spain', 'Mexico',
                'Indonesia', 'Netherlands', 'Saudi Arabia', 'Turkey', 'Taiwan', 'Nigeria', 'Egypt',
                'South Africa', 'Argentina', 'Bangladesh', 'Vietnam', 'Chile', 'Finland', 'Malaysia',
                'Thailand', 'Philippines', 'Ireland', 'Israel', 'UAE', 'Norway', 'Austria', 'Belgium']
    
    years = list(range(2000, 2024))
    
    data = []
    for country in countries:
        base_gdp = np.random.uniform(1000, 70000)
        base_life_exp = np.random.uniform(65, 85)
        base_pop_growth = np.random.uniform(-0.5, 3.0)
        base_electricity = np.random.uniform(70, 100)
        base_co2 = np.random.uniform(0.5, 20)
        base_internet = np.random.uniform(20, 95)
        base_unemployment = np.random.uniform(2, 15)
        base_literacy = np.random.uniform(75, 99)
        
        for year in years:
            # Add some trend and randomness
            year_factor = (year - 2000) / 23
            gdp_growth = np.random.normal(0.03, 0.05)
            
            data.append({
                'Country': country,
                'Year': year,
                'GDP_per_capita': base_gdp * (1 + gdp_growth) ** (year - 2000) + np.random.normal(0, base_gdp * 0.1),
                'Life_expectancy': base_life_exp + year_factor * 2 + np.random.normal(0, 1),
                'Population_growth': base_pop_growth + np.random.normal(0, 0.5),
                'Access_to_electricity': min(100, base_electricity + year_factor * 10 + np.random.normal(0, 2)),
                'CO2_emissions': base_co2 + np.random.normal(0, 1),
                'Internet_users': min(100, base_internet + year_factor * 30 + np.random.normal(0, 5)),
                'Unemployment_rate': max(0, base_unemployment + np.random.normal(0, 2)),
                'Literacy_rate': min(100, base_literacy + year_factor * 5 + np.random.normal(0, 1))
            })
    
    return pd.DataFrame(data)

# Load data
df = load_sample_data()

# Sidebar for filters and controls
st.sidebar.header("🎛️ Dashboard Controls")

# Year range selector
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    value=(2010, 2023),
    step=1
)

# Country selector
countries = sorted(df['Country'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    countries,
    default=['United States', 'China', 'India', 'Germany', 'Brazil']
)

# Indicator selector
indicators = {
    'GDP_per_capita': 'GDP per Capita (USD)',
    'Life_expectancy': 'Life Expectancy (years)',
    'Population_growth': 'Population Growth (%)',
    'Access_to_electricity': 'Access to Electricity (%)',
    'CO2_emissions': 'CO2 Emissions (tons per capita)',
    'Internet_users': 'Internet Users (%)',
    'Unemployment_rate': 'Unemployment Rate (%)',
    'Literacy_rate': 'Literacy Rate (%)'
}

selected_indicator = st.sidebar.selectbox(
    "Select Primary Indicator",
    list(indicators.keys()),
    format_func=lambda x: indicators[x]
)

# Analysis mode selector
analysis_mode = st.sidebar.radio(
    "Analysis Mode",
    ["Overview", "Time Series", "Country Comparison", "Correlation Analysis", "Clustering Analysis"]
)

# Filter data based on selections
filtered_df = df[
    (df['Year'] >= year_range[0]) & 
    (df['Year'] <= year_range[1])
]

if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

# Main dashboard content
if analysis_mode == "Overview":
    st.header("📊 Global Development Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    latest_year = filtered_df['Year'].max()
    latest_data = filtered_df[filtered_df['Year'] == latest_year]
    
    with col1:
        avg_gdp = latest_data['GDP_per_capita'].mean()
        st.metric("Avg GDP per Capita", f"${avg_gdp:,.0f}")
    
    with col2:
        avg_life_exp = latest_data['Life_expectancy'].mean()
        st.metric("Avg Life Expectancy", f"{avg_life_exp:.1f} years")
    
    with col3:
        avg_internet = latest_data['Internet_users'].mean()
        st.metric("Avg Internet Users", f"{avg_internet:.1f}%")
    
    with col4:
        avg_co2 = latest_data['CO2_emissions'].mean()
        st.metric("Avg CO2 Emissions", f"{avg_co2:.1f} tons/capita")
    
    st.markdown("---")
    
    # World map visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🗺️ Global Distribution")
        fig_map = px.choropleth(
            latest_data,
            locations="Country",
            locationmode="country names",
            color=selected_indicator,
            hover_name="Country",
            color_continuous_scale="Viridis",
            title=f"{indicators[selected_indicator]} by Country ({latest_year})"
        )
        fig_map.update_layout(height=400)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.subheader("📈 Top 10 Countries")
        top_countries = latest_data.nlargest(10, selected_indicator)
        fig_bar = px.bar(
            top_countries,
            x=selected_indicator,
            y='Country',
            orientation='h',
            color=selected_indicator,
            color_continuous_scale="Blues",
            title=f"Top 10 - {indicators[selected_indicator]}"
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

elif analysis_mode == "Time Series":
    st.header("📈 Time Series Analysis")
    
    if selected_countries:
        # Multi-country time series
        fig_ts = px.line(
            filtered_df,
            x='Year',
            y=selected_indicator,
            color='Country',
            title=f"{indicators[selected_indicator]} Over Time",
            markers=True
        )
        fig_ts.update_layout(height=500)
        st.plotly_chart(fig_ts, use_container_width=True)
        
        # Growth rate analysis
        st.subheader("📊 Growth Rate Analysis")
        growth_data = []
        for country in selected_countries:
            country_data = filtered_df[filtered_df['Country'] == country].sort_values('Year')
            if len(country_data) > 1:
                growth_rate = ((country_data[selected_indicator].iloc[-1] / country_data[selected_indicator].iloc[0]) ** (1/(len(country_data)-1)) - 1) * 100
                growth_data.append({'Country': country, 'Annual Growth Rate (%)': growth_rate})
        
        if growth_data:
            growth_df = pd.DataFrame(growth_data)
            fig_growth = px.bar(
                growth_df,
                x='Country',
                y='Annual Growth Rate (%)',
                color='Annual Growth Rate (%)',
                color_continuous_scale="RdYlGn",
                title=f"Average Annual Growth Rate - {indicators[selected_indicator]}"
            )
            st.plotly_chart(fig_growth, use_container_width=True)

elif analysis_mode == "Country Comparison":
    st.header("🔄 Country Comparison")
    
    if len(selected_countries) >= 2:
        # Radar chart comparison
        latest_data = filtered_df[filtered_df['Year'] == filtered_df['Year'].max()]
        comparison_indicators = ['GDP_per_capita', 'Life_expectancy', 'Internet_users', 'Access_to_electricity']
        
        fig_radar = go.Figure()
        
        for country in selected_countries[:5]:  # Limit to 5 countries for readability
            country_data = latest_data[latest_data['Country'] == country]
            if not country_data.empty:
                values = []
                for indicator in comparison_indicators:
                    # Normalize values to 0-100 scale for radar chart
                    max_val = filtered_df[indicator].max()
                    min_val = filtered_df[indicator].min()
                    normalized = ((country_data[indicator].iloc[0] - min_val) / (max_val - min_val)) * 100
                    values.append(normalized)
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=[indicators[ind] for ind in comparison_indicators],
                    fill='toself',
                    name=country
                ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100])
            ),
            showlegend=True,
            title="Country Comparison - Multi-Indicator Radar Chart"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Side-by-side comparison table
        st.subheader("📋 Detailed Comparison Table")
        comparison_df = latest_data[latest_data['Country'].isin(selected_countries)][
            ['Country'] + list(indicators.keys())
        ].round(2)
        st.dataframe(comparison_df, use_container_width=True)

elif analysis_mode == "Correlation Analysis":
    st.header("🔗 Correlation Analysis")
    
    # Correlation heatmap
    numeric_cols = [col for col in indicators.keys()]
    correlation_data = filtered_df[numeric_cols].corr()
    
    fig_heatmap, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Development Indicators')
    st.pyplot(fig_heatmap)
    
    # Scatter plot analysis
    col1, col2 = st.columns(2)
    
    with col1:
        indicator_x = st.selectbox("X-axis Indicator", list(indicators.keys()), 
                                  format_func=lambda x: indicators[x])
    
    with col2:
        indicator_y = st.selectbox("Y-axis Indicator", 
                                  [k for k in indicators.keys() if k != indicator_x],
                                  format_func=lambda x: indicators[x])
    
    latest_data = filtered_df[filtered_df['Year'] == filtered_df['Year'].max()]
    fig_scatter = px.scatter(
        latest_data,
        x=indicator_x,
        y=indicator_y,
        color='Country',
        size='GDP_per_capita',
        hover_name='Country',
        title=f"{indicators[indicator_x]} vs {indicators[indicator_y]}",
        trendline="ols"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

elif analysis_mode == "Clustering Analysis":
    st.header("🎯 Country Clustering Analysis")
    
    # Prepare data for clustering
    latest_data = filtered_df[filtered_df['Year'] == filtered_df['Year'].max()]
    clustering_features = ['GDP_per_capita', 'Life_expectancy', 'Internet_users', 'Access_to_electricity']
    
    # Remove rows with missing values
    cluster_data = latest_data[['Country'] + clustering_features].dropna()
    
    if len(cluster_data) > 0:
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(cluster_data[clustering_features])
        
        # Number of clusters selector
        n_clusters = st.slider("Number of Clusters", 2, 8, 4)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Add cluster labels to data
        cluster_data['Cluster'] = cluster_labels
        
        # PCA for visualization
        pca = PCA(n_components=2)
        pca_features = pca.fit_transform(features_scaled)
        
        cluster_data['PCA1'] = pca_features[:, 0]
        cluster_data['PCA2'] = pca_features[:, 1]
        
        # Plot clusters
        fig_cluster = px.scatter(
            cluster_data,
            x='PCA1',
            y='PCA2',
            color='Cluster',
            hover_name='Country',
            title=f"Country Clusters (K={n_clusters}) - PCA Visualization",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        
        # Cluster summary
        st.subheader("📊 Cluster Summary")
        cluster_summary = cluster_data.groupby('Cluster')[clustering_features].mean().round(2)
        st.dataframe(cluster_summary, use_container_width=True)
        
        # Countries by cluster
        st.subheader("🏛️ Countries by Cluster")
        for cluster_id in sorted(cluster_data['Cluster'].unique()):
            countries_in_cluster = cluster_data[cluster_data['Cluster'] == cluster_id]['Country'].tolist()
            st.write(f"**Cluster {cluster_id}:** {', '.join(countries_in_cluster)}")

# Data export section
st.sidebar.markdown("---")
st.sidebar.header("💾 Data Export")

if st.sidebar.button("Download Filtered Data"):
    csv = filtered_df.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"world_bank_data_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    📊 World Bank Global Development Dashboard | 
    Built with Streamlit & Plotly | 
    Data: World Bank Open Data
</div>
""", unsafe_allow_html=True)

# Instructions for running
st.sidebar.markdown("---")
st.sidebar.info("""
**How to run this dashboard:**

1. Save this code as `app.py`
2. Install requirements:
   ```
   pip install streamlit pandas plotly seaborn scikit-learn
   ```
3. Run the dashboard:
   ```
   streamlit run app.py
   ```
4. Replace sample data with real World Bank data using their API or CSV files
""")