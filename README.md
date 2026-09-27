[analysis.py](https://github.com/user-attachments/files/32696619/analysis.py)
# NSW Government Service Delivery Intelligence

**Status:** 🛠️ Planned / In Development  
**Python • SQL • Power BI • DAX • ETL • Forecasting**

A decision-support platform for analysing public-service demand,
processing times, service accessibility and regional performance.
from pathlib import Path
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "nsw_service_delivery.db"
OUT = ROOT / "outputs" / "graphs"
OUT.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(DB) as con:
    yearly = pd.read_sql_query("""SELECT year, COUNT(*) interactions,
        AVG(satisfaction_score) satisfaction,
        100.0*AVG(first_contact_resolved) fcr
        FROM service_interactions GROUP BY year ORDER BY year""", con)
    regional = pd.read_sql_query("""SELECT region, AVG(satisfaction_score) satisfaction
        FROM service_interactions GROUP BY region ORDER BY satisfaction""", con)
    channels = pd.read_sql_query("""SELECT channel, AVG(wait_minutes) wait_minutes
        FROM service_interactions GROUP BY channel ORDER BY wait_minutes DESC""", con)
    services = pd.read_sql_query("""SELECT service, 100.0*AVG(complaint) complaint_rate
        FROM service_interactions GROUP BY service ORDER BY complaint_rate""", con)
    scatter = pd.read_sql_query("""SELECT wait_minutes, satisfaction_score
        FROM service_interactions WHERE interaction_id % 250 = 0""", con)

def save(name):
    plt.tight_layout()
    plt.savefig(OUT/name, dpi=180, bbox_inches="tight")
    plt.close()

plt.figure(figsize=(9,5)); plt.plot(yearly.year, yearly.satisfaction, marker="o")
plt.title("Average satisfaction by year"); plt.xlabel("Year"); plt.ylabel("Satisfaction (1–5)"); save("01_satisfaction_trend.png")

plt.figure(figsize=(9,6)); plt.barh(regional.region, regional.satisfaction)
plt.title("Average satisfaction by NSW region"); plt.xlabel("Satisfaction (1–5)"); save("02_regional_satisfaction.png")

plt.figure(figsize=(8,5)); plt.bar(channels.channel, channels.wait_minutes)
plt.title("Average wait time by channel"); plt.ylabel("Minutes"); save("03_channel_wait_time.png")

plt.figure(figsize=(9,6)); plt.barh(services.service, services.complaint_rate)
plt.title("Complaint rate by service"); plt.xlabel("Complaint rate (%)"); save("04_service_complaints.png")

plt.figure(figsize=(8,5)); plt.scatter(scatter.wait_minutes, scatter.satisfaction_score, alpha=.35)
plt.title("Wait time vs satisfaction"); plt.xlabel("Wait time (minutes)"); plt.ylabel("Satisfaction (1–5)"); save("05_wait_vs_satisfaction.png")
print("Five graphs generated in", OUT)
