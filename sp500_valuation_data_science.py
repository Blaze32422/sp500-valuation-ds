import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from jinja2 import Template
from io import BytesIO

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("deep")

# Simulated stock data
fake_stock_info = {
    "AAPL": {
        "trailingPE": 28.5,
        "pegRatio": 1.4,
        "priceToBook": 12.3,
        "returnOnEquity": 0.28,
        "trailingEps": 6.05,
        "marketCap": 2500000000000,
        "dividendYield": 0.006
    },
    "MSFT": {
        "trailingPE": 35.2,
        "pegRatio": 2.0,
        "priceToBook": 13.2,
        "returnOnEquity": 0.35,
        "trailingEps": 9.21,
        "marketCap": 2800000000000,
        "dividendYield": 0.009
    },
    "GOOGL": {
        "trailingPE": 23.8,
        "pegRatio": 1.1,
        "priceToBook": 6.5,
        "returnOnEquity": 0.24,
        "trailingEps": 5.91,
        "marketCap": 1800000000000,
        "dividendYield": 0.0
    }
}

# Load and clean data
def load_data(uploaded_file=None):
    try:
        if uploaded_file:
            df = pd.read_csv(uploaded_file, thousands=',', na_values=['', 'nan'])
            required_columns = ['Symbol', 'Shortname', 'Sector', 'Currentprice', 'Marketcap', 'Ebitda', 'Revenuegrowth']
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                st.error(f"Uploaded CSV is missing columns: {missing_cols}")
                return pd.DataFrame()
            # Map CSV columns to required metrics
            df['P/E'] = df['Currentprice'] / df['Ebitda']  # Simplified P/E calculation
            df['EPS'] = df['Ebitda'] / 1000000  # Simplified EPS
            df['MarketCap'] = df['Marketcap']
            df['ROE'] = df['Revenuegrowth'] * 100  # Simplified ROE
        else:
            # Convert fake_stock_info to DataFrame
            data = []
            for ticker, info in fake_stock_info.items():
                data.append({
                    'Symbol': ticker,
                    'Shortname': ticker,
                    'Sector': 'Technology',
                    'P/E': info.get("trailingPE"),
                    'PEG': info.get("pegRatio"),
                    'P/B': info.get("priceToBook"),
                    'ROE': info.get("returnOnEquity") * 100 if info.get("returnOnEquity") else None,
                    'EPS': info.get("trailingEps"),
                    'MarketCap': info.get("marketCap"),
                    'DividendYield': info.get("dividendYield") * 100 if info.get("dividendYield") else None
                })
            df = pd.DataFrame(data)
        
        # Debug: Display raw DataFrame and columns
        st.write("**Debug: Raw DataFrame**", df)
        st.write("**Debug: DataFrame Columns**", df.columns.tolist())
        
        # Check for required columns
        required_metrics = ['P/E', 'EPS', 'MarketCap', 'ROE']
        missing_metrics = [col for col in required_metrics if col not in df.columns]
        if missing_metrics:
            st.error(f"DataFrame is missing required columns: {missing_metrics}")
            return pd.DataFrame()
        
        # Clean data
        df = df.dropna(subset=required_metrics)
        for col in required_metrics:
            try:
                df[col] = df[col].astype(float)
            except Exception as e:
                st.error(f"Error converting column {col} to float: {e}")
                return pd.DataFrame()
        
        # Debug: Display cleaned DataFrame
        st.write("**Debug: Cleaned DataFrame**", df)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Calculate valuation metrics
def calculate_valuation_metrics(df):
    try:
        df['ValueScore'] = df.apply(
            lambda row: (1 / row['P/E']) * (row['ROE'] / 100) * (row['MarketCap'] / 1e9)
            if row['EPS'] > 0 else 0, axis=1)
        return df
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return df

# Filter data
def filter_data(df):
    try:
        return df[
            (df['P/E'] < 30) &
            (df['ROE'] > 15) &
            (df['EPS'] > 0)
        ].sort_values(by='ValueScore', ascending=False).reset_index(drop=True)
    except Exception as e:
        st.error(f"Error filtering data: {e}")
        return pd.DataFrame()

# Perform EDA
def perform_eda(df):
    st.subheader("Exploratory Data Analysis")
    st.write("**Summary Statistics**")
    st.dataframe(df[['P/E', 'ROE', 'EPS', 'MarketCap']].describe().round(2))
    
    st.write("**Sector Distribution**")
    sector_counts = df['Sector'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=sector_counts.values, y=sector_counts.index, ax=ax)
    ax.set_title('Distribution of Companies by Sector')
    ax.set_xlabel('Number of Companies')
    ax.set_ylabel('Sector')
    plt.tight_layout()
    st.pyplot(fig)

# Plotting functions
def plot_value_scores(top10):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='ValueScore', y='Symbol', data=top10, ax=ax)
    ax.set_title('Top 10 Companies by Value Score')
    ax.set_xlabel('Value Score')
    ax.set_ylabel('Ticker')
    plt.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    plt.close(fig)
    return buffer.getvalue()

def plot_eps_trend(top20):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='P/E', y='EPS', size='MarketCap', hue='Sector', data=top20, ax=ax)
    for i, row in top20.iterrows():
        ax.text(row['P/E'], row['EPS'], row['Symbol'], fontsize=8)
    ax.set_title('P/E vs EPS (Top 20)')
    ax.set_xlabel('P/E Ratio')
    ax.set_ylabel('EPS (USD)')
    plt.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300)
    plt.close(fig)
    return buffer.getvalue()

# Convert plot to base64
def plot_to_base64(plot_bytes):
    return base64.b64encode(plot_bytes).decode('utf-8')

# Generate HTML report
def generate_html_report(top10, top20, sector_counts, bar_plot_bytes, scatter_plot_bytes):
    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>S&P 500 Valuation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f4f4f4; color: #333; }
            h1 { color: #2c3e50; }
            h2 { color: #34495e; }
            table { width: 100%; border-collapse: collapse; margin-bottom: 20px; background-color: white; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #3498db; color: white; }
            img { max-width: 100%; height: auto; margin-bottom: 20px; }
            .section { margin-bottom: 40px; }
        </style>
    </head>
    <body>
        <h1>S&P 500 Valuation Report: Best Value per EPS</h1>
        <div class="section">
            <h2>Summary</h2>
            <p>This report ranks companies by value per EPS, using a score based on P/E ratio, ROE, and market cap. EDA reveals sector trends, with Technology dominating ({{ sector_counts.get('Technology', 0) }} of top 10 companies).</p>
        </div>
        <div class="section">
            <h2>Top 10 Value Stocks</h2>
            <table>
                <tr>
                    <th>Ticker</th>
                    <th>Company</th>
                    <th>Sector</th>
                    <th>P/E</th>
                    <th>ROE (%)</th>
                    <th>EPS ($)</th>
                    <th>Market Cap ($B)</th>
                    <th>Value Score</th>
                </tr>
                {% for company in top10 %}
                <tr>
                    <td>{{ company['Symbol'] }}</td>
                    <td>{{ company['Shortname'] }}</td>
                    <td>{{ company['Sector'] }}</td>
                    <td>{{ company['P/E']|round(2) }}</td>
                    <td>{{ company['ROE']|round(2) }}</td>
                    <td>{{ company['EPS']|round(2) }}</td>
                    <td>{{ (company['MarketCap']/1e9)|round(2) }}</td>
                    <td>{{ company['ValueScore']|round(4) }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        <div class="section">
            <h2>Value Scores of Top 10 Companies</h2>
            <img src="data:image/png;base64,{{ bar_plot_base64 }}" alt="Value Scores Bar Chart">
        </div>
        <div class="section">
            <h2>P/E vs EPS</h2>
            <img src="data:image/png;base64,{{ scatter_plot_base64 }}" alt="Scatter Plot">
        </div>
        <div class="section">
            <h2>Conclusion</h2>
            <p>The analysis identifies undervalued stocks with strong earnings potential, with Technology stocks leading due to high ROE and EPS.</p>
        </div>
    </body>
    </html>
    """
    bar_plot_base64 = plot_to_base64(bar_plot_bytes)
    scatter_plot_base64 = plot_to_base64(scatter_plot_bytes)
    with open('sp500_valuation_report.html', 'w') as f:
        f.write(Template(template_str).render(
            top10=top10.to_dict('records'),
            sector_counts=sector_counts,
            bar_plot_base64=bar_plot_base64,
            scatter_plot_base64=scatter_plot_base64
        ))

# Main Streamlit app
def main():
    st.markdown("""
        <style>
        .main { background-color: #f0f2f6; }
        .stButton>button { background-color: #3498db; color: white; }
        h1 { color: #2c3e50; }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("S&P 500 Valuation Analysis for Data Science")
    st.write("""
    This tool ranks S&P 500 companies by value per EPS, using a score based on P/E ratio, ROE, and market cap. Upload a CSV file (columns: Symbol, Shortname, Sector, Currentprice, Marketcap, Ebitda, Revenuegrowth) or use default data.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload sp500_companies.csv", type="csv")
    df = load_data(uploaded_file=uploaded_file)

    if not df.empty:
        # Perform EDA
        perform_eda(df)

        # Process and filter data
        df = calculate_valuation_metrics(df)
        filtered_df = filter_data(df)

        # Display top 10
        st.subheader("Top 10 Value Stocks")
        top10 = filtered_df.head(10)
        st.dataframe(
            top10[['Symbol', 'Shortname', 'Sector', 'P/E', 'ROE', 'EPS', 'MarketCap', 'ValueScore']]
            .round(2)
        )

        # Sector counts
        sector_counts = top10['Sector'].value_counts().to_dict()
        st.write(f"**Insight**: Technology sector dominates with {sector_counts.get('Technology', 0)} of the top 10 companies.")

        # Visualizations
        st.subheader("Value Scores of Top 10 Companies")
        bar_plot_bytes = plot_value_scores(top10)
        st.image(bar_plot_bytes, use_column_width=True)

        st.subheader("P/E vs EPS")
        top20 = filtered_df.head(20)
        scatter_plot_bytes = plot_eps_trend(top20)
        st.image(scatter_plot_bytes, use_column_width=True)

        # Download HTML report
        st.subheader("Download Report")
        generate_html_report(top10, top20, sector_counts, bar_plot_bytes, scatter_plot_bytes)
        with open('sp500_valuation_report.html', 'rb') as f:
            st.download_button(
                label="Download HTML Report",
                data=f,
                file_name="sp500_valuation_report.html",
                mime="text/html"
            )

if __name__ == "__main__":
    main()
