[analysisNSW Government Service Delivery Intelligence.py](https://github.com/user-attachments/files/32696656/analysisNSW.Government.Service.Delivery.Intelligence.py)
# NSW Government Service Delivery Intelligence

**Status:** 🛠️ Planned / In Development  
**Python • SQL • Power BI • DAX • ETL • Forecasting**

A decision-support platform for analysing public-service demand,
processing times, service accessibility and regional performance.
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\oluwa\Downloads\nsw-government-service-delivery-intelligence"
)

DB = ROOT / "data" / "nsw_service_delivery.db"
OUT = ROOT / "outputs" / "graphs"

# Create graph output directory if it does not exist
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK DATABASE
# ============================================================

if not DB.exists():
    raise FileNotFoundError(
        f"\nDatabase not found:\n{DB}\n\n"
        "Make sure nsw_service_delivery.db is inside the data folder."
    )

print("Database found:")
print(DB)


# ============================================================
# LOAD DATA FROM SQLITE DATABASE
# ============================================================

with sqlite3.connect(str(DB)) as con:

    yearly = pd.read_sql_query(
        """
        SELECT
            year,
            COUNT(*) AS interactions,
            AVG(satisfaction_score) AS satisfaction,
            100.0 * AVG(first_contact_resolved) AS fcr
        FROM service_interactions
        GROUP BY year
        ORDER BY year
        """,
        con
    )

    regional = pd.read_sql_query(
        """
        SELECT
            region,
            AVG(satisfaction_score) AS satisfaction
        FROM service_interactions
        GROUP BY region
        ORDER BY satisfaction
        """,
        con
    )

    channels = pd.read_sql_query(
        """
        SELECT
            channel,
            AVG(wait_minutes) AS wait_minutes
        FROM service_interactions
        GROUP BY channel
        ORDER BY wait_minutes DESC
        """,
        con
    )

    services = pd.read_sql_query(
        """
        SELECT
            service,
            100.0 * AVG(complaint) AS complaint_rate
        FROM service_interactions
        GROUP BY service
        ORDER BY complaint_rate
        """,
        con
    )

    scatter = pd.read_sql_query(
        """
        SELECT
            wait_minutes,
            satisfaction_score
        FROM service_interactions
        WHERE interaction_id % 250 = 0
        """,
        con
    )

    prediction_data = pd.read_sql_query(
        """
        SELECT
            wait_minutes,
            satisfaction_score
        FROM service_interactions
        WHERE wait_minutes IS NOT NULL
          AND satisfaction_score IS NOT NULL
        """,
        con
    )


# ============================================================
# CHECK DATA
# ============================================================

if prediction_data.empty:
    raise ValueError(
        "No prediction data found in the service_interactions table."
    )

print("\nData loaded successfully.")
print("Prediction records:", len(prediction_data))


# ============================================================
# FUNCTION TO SAVE GRAPHS
# ============================================================

def save_graph(filename):
    plt.tight_layout()

    plt.savefig(
        OUT / filename,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {filename}")


# ============================================================
# GRAPH 1
# AVERAGE SATISFACTION BY YEAR
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    yearly["year"],
    yearly["satisfaction"],
    marker="o",
    linewidth=2
)

plt.title("Average Satisfaction by Year")
plt.xlabel("Year")
plt.ylabel("Satisfaction Score (1-5)")


save_graph("01_satisfaction_trend.png")


# ============================================================
# GRAPH 2
# AVERAGE SATISFACTION BY NSW REGION
# ============================================================

plt.figure(figsize=(9, 6))

plt.barh(
    regional["region"],
    regional["satisfaction"]
)

plt.title("Average Satisfaction by NSW Region")
plt.xlabel("Average Satisfaction Score (1-5)")
plt.ylabel("NSW Region")

save_graph("02_regional_satisfaction.png")


# ============================================================
# GRAPH 3
# AVERAGE WAIT TIME BY CHANNEL
# ============================================================

plt.figure(figsize=(8, 5))

plt.bar(
    channels["channel"],
    channels["wait_minutes"]
)

plt.title("Average Wait Time by Service Channel")
plt.xlabel("Service Channel")
plt.ylabel("Average Wait Time (Minutes)")
plt.xticks(rotation=20)

save_graph("03_channel_wait_time.png")


# ============================================================
# GRAPH 4
# COMPLAINT RATE BY SERVICE
# ============================================================

plt.figure(figsize=(9, 6))

plt.barh(
    services["service"],
    services["complaint_rate"]
)

plt.title("Complaint Rate by NSW Government Service")
plt.xlabel("Complaint Rate (%)")
plt.ylabel("Service")

save_graph("04_service_complaints.png")


# ============================================================
# GRAPH 5
# WAIT TIME VS SATISFACTION
# ============================================================

plt.figure(figsize=(8, 5))

plt.scatter(
    scatter["wait_minutes"],
    scatter["satisfaction_score"],
    alpha=0.35
)

plt.title("Wait Time vs Citizen Satisfaction")
plt.xlabel("Wait Time (Minutes)")
plt.ylabel("Satisfaction Score (1-5)")

save_graph("05_wait_vs_satisfaction.png")


# ============================================================
# PREDICTIVE ANALYTICS
# ============================================================

print("\nBuilding predictive model...")


# Independent variable
X = prediction_data[["wait_minutes"]]

# Target variable
y = prediction_data["satisfaction_score"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# TRAIN LINEAR REGRESSION MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)


# ============================================================
# MAKE PREDICTIONS
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


print("\n======================================")
print("PREDICTIVE MODEL PERFORMANCE")
print("======================================")

print(f"Mean Absolute Error: {mae:.3f}")
print(f"Root Mean Squared Error: {rmse:.3f}")
print(f"R-squared Score: {r2:.3f}")

print("======================================")


# ============================================================
# GRAPH 6
# PREDICTED SATISFACTION BY WAIT TIME
# ============================================================

wait_range = pd.DataFrame(
    {
        "wait_minutes": np.linspace(
            prediction_data["wait_minutes"].min(),
            prediction_data["wait_minutes"].max(),
            100
        )
    }
)

predicted_satisfaction = model.predict(wait_range)


plt.figure(figsize=(9, 6))

plt.scatter(
    prediction_data["wait_minutes"],
    prediction_data["satisfaction_score"],
    alpha=0.12,
    label="Actual Interactions"
)

plt.plot(
    wait_range["wait_minutes"],
    predicted_satisfaction,
    linewidth=3,
    label="Prediction"
)

plt.title(
    "Predicted Citizen Satisfaction Based on Wait Time"
)

plt.xlabel("Wait Time (Minutes)")
plt.ylabel("Predicted Satisfaction Score")

plt.legend()


save_graph(
    "06_predicted_satisfaction_wait_time.png"
)


# ============================================================
# GRAPH 7
# ACTUAL VS PREDICTED SATISFACTION
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.35
)

minimum = min(
    y_test.min(),
    predictions.min()
)

maximum = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    linewidth=2
)

plt.title(
    "Actual vs Predicted Citizen Satisfaction"
)

plt.xlabel(
    "Actual Satisfaction Score"
)

plt.ylabel(
    "Predicted Satisfaction Score"
)

save_graph(
    "07_actual_vs_predicted_satisfaction.png"
)


# ============================================================
# GRAPH 8
# FUTURE SATISFACTION FORECAST
# ============================================================

if len(yearly) >= 2:

    year_model = LinearRegression()

    X_year = yearly[["year"]]
    y_year = yearly["satisfaction"]

    year_model.fit(
        X_year,
        y_year
    )

    last_year = int(
        yearly["year"].max()
    )

    future_years = np.arange(
        last_year + 1,
        last_year + 6
    )

    future_year_df = pd.DataFrame(
        {
            "year": future_years
        }
    )

    future_satisfaction = year_model.predict(
        future_year_df
    )

    # Keep forecast within valid 1-5 satisfaction scale
    future_satisfaction = np.clip(
        future_satisfaction,
        1,
        5
    )

    forecast = pd.DataFrame(
        {
            "year": future_years,
            "predicted_satisfaction":
                future_satisfaction
        }
    )

    print("\n======================================")
    print("FIVE-YEAR SATISFACTION FORECAST")
    print("======================================")

    print(
        forecast.to_string(
            index=False
        )
    )

    print("======================================")


    plt.figure(figsize=(10, 6))

    plt.plot(
        yearly["year"],
        yearly["satisfaction"],
        marker="o",
        linewidth=2,
        label="Historical Satisfaction"
    )

    plt.plot(
        forecast["year"],
        forecast["predicted_satisfaction"],
        marker="o",
        linestyle="--",
        linewidth=2,
        label="Predicted Satisfaction"
    )

    plt.axvline(
        x=last_year,
        linestyle=":",
        alpha=0.7
    )

    plt.title(
        "NSW Government Service Satisfaction Forecast"
    )

    plt.xlabel("Year")

    plt.ylabel(
        "Average Satisfaction Score (1-5)"
    )

    plt.legend()
    
    save_graph(
        "08_satisfaction_forecast.png"
    )

else:

    print(
        "\nNot enough yearly data "
        "to create a future forecast."
    )


# ============================================================
# GRAPH 9
# PREDICTION ERROR DISTRIBUTION
# ============================================================

errors = y_test - predictions

plt.figure(figsize=(8, 5))

plt.hist(
    errors,
    bins=30
)

plt.axvline(
    x=0,
    linestyle="--",
    linewidth=2
)

plt.title(
    "Satisfaction Prediction Error Distribution"
)

plt.xlabel(
    "Prediction Error (Actual - Predicted)"
)

plt.ylabel(
    "Number of Service Interactions"
)

save_graph(
    "09_prediction_errors.png"
)


# ============================================================
# SAVE PREDICTION RESULTS TO CSV
# ============================================================

prediction_results = pd.DataFrame(
    {
        "wait_minutes": X_test["wait_minutes"].values,
        "actual_satisfaction": y_test.values,
        "predicted_satisfaction": predictions
    }
)

prediction_results.to_csv(
    OUT.parent / "prediction_results.csv",
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n======================================")
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("======================================")

print("\n9 analytical graphs generated.")

print(
    "\nGraphs saved in:"
)

print(OUT)

print(
    "\nPrediction results saved in:"
)

print(
    OUT.parent / "prediction_results.csv"
)
