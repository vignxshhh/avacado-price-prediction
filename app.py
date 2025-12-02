import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="🥑 Avocado Price Predictor & EDA", layout="wide")

# --- Load Model & Encoders ---
@st.cache_resource
def load_model():
    return joblib.load("models/avocado_price_model.pkl")

@st.cache_resource
def load_encoders():
    le_region = joblib.load("models/region_encoder.pkl")
    le_type = joblib.load("models/type_encoder.pkl")
    return le_region, le_type

@st.cache_data
def load_data():
    df = pd.read_csv("data/avocado.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    
    # Custom Season Classification
    def classify_season(month):
        if month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Fall"
        else:
            return "Winter"
    
    df["Season"] = df["Month"].apply(classify_season)
    return df

model = load_model()
le_region, le_type = load_encoders()
df = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio("Go to:", ["Prediction", "EDA"])

# -------------------------------
# 🔮 PREDICTION SECTION
# -------------------------------
if section == "Prediction":
    st.title("🥑 Hass Avocado Price Predictor")

    selected_date = st.date_input("Select Date", datetime.today())
    region = st.selectbox("Select Region", le_region.classes_)
    avocado_type = st.selectbox("Select Type", le_type.classes_)

    year = selected_date.year
    month = selected_date.month
    day = selected_date.day
    day_of_week = selected_date.weekday()

    region_encoded = le_region.transform([region])[0]
    type_encoded = le_type.transform([avocado_type])[0]

    input_data = pd.DataFrame([{
        "Total Volume": 0,
        "4046": 0,
        "4225": 0,
        "4770": 0,
        "Total Bags": 0,
        "Small Bags": 0,
        "Large Bags": 0,
        "XLarge Bags": 0,
        "Year": year,
        "region_encoded": region_encoded,
        "type_encoded": type_encoded,
        "Month": month,
        "Day": day,
        "DayOfWeek": day_of_week
    }])

    if st.button("Predict Avocado Price"):
        predicted_price = model.predict(input_data)[0]
        st.success(f"💰 Predicted Average Price: ${predicted_price:.2f}")
        st.info(f"📍 Region: {region} | Type: {avocado_type} | Date: {selected_date.strftime('%Y-%m-%d')}")

# -------------------------------
# 📊 EDA SECTION
# -------------------------------
elif section == "EDA":
    st.title("📊 Avocado Price EDA Dashboard")

    # 1. Average Price Over Time
    st.subheader("📈 1. Average Price Over Time")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x="Date", y="AveragePrice", ax=ax1, ci=None)
    ax1.set_title("Average Price Over Time")
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # 2. Sales Volume by Type
    st.subheader("🥑 2. Sales Volume by Type")
    sales_by_type = df.groupby("type")["Total Volume"].sum()
    fig2, ax2 = plt.subplots()
    ax2.pie(sales_by_type, labels=sales_by_type.index, autopct="%1.1f%%", startangle=140)
    ax2.set_title("Sales Distribution by Type")
    st.pyplot(fig2)

    # 3. Average Price by Region
    st.subheader("📊 3. Average Price by Region")
    avg_price_region = df.groupby("region")["AveragePrice"].mean().sort_values(ascending=False)
    fig3, ax3 = plt.subplots(figsize=(10, 12))
    sns.barplot(x=avg_price_region.values, y=avg_price_region.index, palette="viridis", ax=ax3)
    ax3.set_title("Average Price by Region")
    st.pyplot(fig3)

    # 4. Average Price by Type
    st.subheader("🥑 4. Average Price by Type")
    fig4, ax4 = plt.subplots()
    sns.boxplot(data=df, x="type", y="AveragePrice", palette="Set2", ax=ax4)
    ax4.set_title("Price Distribution by Type")
    st.pyplot(fig4)

    # 5. Correlation Heatmap
    st.subheader("🔗 5. Correlation Heatmap")
    corr = df[["AveragePrice", "Total Volume", "4046", "4225", "4770",
               "Total Bags", "Small Bags", "Large Bags", "XLarge Bags"]].corr()
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)

    # 6. Monthly Average Price Trend
    st.subheader("📅 6. Monthly Average Price Trend")
    monthly_avg = df.groupby("Month")["AveragePrice"].mean()
    fig6, ax6 = plt.subplots()
    sns.lineplot(x=monthly_avg.index, y=monthly_avg.values, marker="o", ax=ax6)
    ax6.set_title("Monthly Avg Price Trend")
    ax6.set_xlabel("Month")
    ax6.set_ylabel("Average Price")
    st.pyplot(fig6)

    # 7. Seasonal Sales Volume
    st.subheader("🍂 7. Total Sales by Season")
    season_sales = df.groupby("Season")["Total Volume"].sum().sort_values(ascending=False)
    fig7, ax7 = plt.subplots()
    sns.barplot(x=season_sales.index, y=season_sales.values, palette="coolwarm", ax=ax7)
    ax7.set_title("Total Avocado Sales by Season")
    st.pyplot(fig7)

    # 8. Yearly Average Price Trend
    st.subheader("📆 8. Yearly Average Price Trend")
    yearly_avg = df.groupby("Year")["AveragePrice"].mean()
    fig8, ax8 = plt.subplots()
    sns.lineplot(x=yearly_avg.index, y=yearly_avg.values, marker="o", ax=ax8)
    ax8.set_title("Yearly Avg Price Trend")
    st.pyplot(fig8)

    # 9. Bag Size Contribution
    st.subheader("📦 9. Contribution of Bag Sizes")
    bag_means = df[["Small Bags", "Large Bags", "XLarge Bags"]].mean()
    fig9, ax9 = plt.subplots()
    ax9.pie(bag_means, labels=bag_means.index, autopct="%1.1f%%", startangle=140)
    ax9.set_title("Average Distribution of Bag Sizes")
    st.pyplot(fig9)

    # 10. Region Comparison (User Selects 2)
    st.subheader("🌍 10. Compare Prices Between Two Regions")
    region_list = df["region"].unique()
    col1, col2 = st.columns(2)
    with col1:
        region1 = st.selectbox("Select First Region", region_list, index=0, key="region1")
    with col2:
        region2 = st.selectbox("Select Second Region", region_list, index=1, key="region2")

    df_comp = df[df["region"].isin([region1, region2])]
    fig10, ax10 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df_comp, x="Date", y="AveragePrice", hue="region", ax=ax10, ci=None)
    ax10.set_title(f"Price Trends: {region1} vs {region2}")
    ax10.set_xlabel("Date")
    ax10.set_ylabel("Avg Price ($)")
    ax10.xaxis.set_major_locator(mdates.YearLocator())
    ax10.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    st.pyplot(fig10)

    # 11. Top Regions by Sales
    st.subheader("🏆 11. Top 10 Regions by Total Sales")
    top_regions = df.groupby("region")["Total Volume"].sum().nlargest(10)
    fig11, ax11 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_regions.values, y=top_regions.index, palette="mako", ax=ax11)
    ax11.set_title("Top 10 Regions by Sales")
    st.pyplot(fig11)

        # 12. Bag Sales Over Time
    st.subheader("👜 12. Bag Sales Over Time")
    fig12, ax12 = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x="Date", y="Small Bags", label="Small Bags", ax=ax12)
    sns.lineplot(data=df, x="Date", y="Large Bags", label="Large Bags", ax=ax12)
    sns.lineplot(data=df, x="Date", y="XLarge Bags", label="XLarge Bags", ax=ax12)
    ax12.set_title("Bag Sales Over Time")
    ax12.set_xlabel("Date")
    ax12.set_ylabel("Sales Volume")
    ax12.legend()
    ax12.xaxis.set_major_locator(mdates.YearLocator())
    ax12.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    st.pyplot(fig12)

    # 13. Bag Type Share by Year
    st.subheader("📦 13. Bag Size Share by Year")
    yearly_bags = df.groupby("Year")[["Small Bags", "Large Bags", "XLarge Bags"]].sum()
    fig13, ax13 = plt.subplots(figsize=(10, 6))
    yearly_bags.plot(kind="bar", stacked=True, ax=ax13, colormap="tab20")
    ax13.set_title("Yearly Bag Sales Distribution")
    ax13.set_xlabel("Year")
    ax13.set_ylabel("Total Bags Sold")
    st.pyplot(fig13)

    # 14. Bag Size Contribution by Region (User Selects Region)
    st.subheader("🌎 14. Bag Size Contribution by Region")
    selected_region = st.selectbox("Select Region for Bag Analysis", df["region"].unique())
    region_bag = df[df["region"] == selected_region][["Small Bags", "Large Bags", "XLarge Bags"]].sum()
    fig14, ax14 = plt.subplots()
    ax14.pie(region_bag, labels=region_bag.index, autopct="%1.1f%%", startangle=140)
    ax14.set_title(f"Bag Size Distribution in {selected_region}")
    st.pyplot(fig14)

    # 15. Correlation of Bag Sizes with Price
    st.subheader("🔗 15. Correlation of Bag Sizes with Average Price")
    fig15, ax15 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="Small Bags", y="AveragePrice", label="Small Bags", ax=ax15)
    sns.scatterplot(data=df, x="Large Bags", y="AveragePrice", label="Large Bags", ax=ax15)
    sns.scatterplot(data=df, x="XLarge Bags", y="AveragePrice", label="XLarge Bags", ax=ax15)
    ax15.set_title("Correlation Between Bag Sizes & Average Price")
    ax15.set_xlabel("Bag Sales Volume")
    ax15.set_ylabel("Average Price")
    ax15.legend()
    st.pyplot(fig15)
