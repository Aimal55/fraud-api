import pandas as pd
import numpy as np
import os
import os



script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

transactions_dir = os.path.join(parent_dir, "dataset", "transactions")
os.makedirs(transactions_dir, exist_ok=True)

np.random.seed(42)
n_legit = 8000                   
n_fraud = 10                 #class-imbalance ≈ 1 : 800

##LEGIT TRANSACTION
#Amount Range is Between 1 to 500

legit_amounts = np.random.uniform(1, 500, n_legit)
legit_bins = np.random.choice([123456, 654321, 111111], n_legit)
legit_device_ids = [f"device_{i}" for i in range(n_legit)]
legit_geo_lat = np.random.normal(25.0, 0.05, n_legit)
legit_geo_lon = np.random.normal(67.0, 0.05, n_legit)
legit_labels = np.zeros(n_legit)

##FRAUD TRANSACTION
#Amount Higher Between 300 to 1000
#Creating unique fraud devices id  
#Fraud Geo Location Offset from the Legit 

fraud_amounts = np.random.uniform(300, 1000, n_fraud)
fraud_bins = np.random.choice([123456, 654321, 111111], n_fraud)
fraud_device_ids = [f"fraud_device_{i%7}" for i in range(n_fraud)]

fraud_geo_lat = np.random.normal(25.2, 0.05, n_fraud)  # shifted north
fraud_geo_lon = np.random.normal(67.2, 0.05, n_fraud)  # shifted east
fraud_labels = np.ones(n_fraud)
##Combine longtitude and latitude of geo as tuple
geo_tuples_legit = [(float(lat), float(lon)) for lat, lon in zip(legit_geo_lat, legit_geo_lon)]
geo_tuples_fraud = [(float(lat), float(lon)) for lat, lon in zip(fraud_geo_lat, fraud_geo_lon)]

df = pd.DataFrame({
    "amount": np.concatenate([legit_amounts, fraud_amounts]),
    "bin": np.concatenate([legit_bins, fraud_bins]),
    "device_id": legit_device_ids + fraud_device_ids,
    "geo": geo_tuples_legit + geo_tuples_fraud,
    "label": np.concatenate([legit_labels, fraud_labels])
})

# Shuffle rows
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
csv_path = os.path.join(transactions_dir, "transactions.csv")
df.to_csv(csv_path, index=False)
print("Synthetic Data of Transactions Generated!!")
