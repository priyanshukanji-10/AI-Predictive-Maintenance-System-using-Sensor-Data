from kafka import KafkaProducer
import pandas as pd
import json
import time
import random

df = pd.read_csv(r"D:\Predictive Maintenance\ai 2020.csv")

producer = KafkaProducer(
    bootstrap_servers='127.0.0.1:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
    api_version=(0, 10, 1)  # prevents version check error
)

while True:
    row = df.sample(1).iloc[0]

    data = {
        "temperature": float(row["Air temperature [K]"]),
        "process_temp": float(row["Process temperature [K]"]),
        "rpm": int(row["Rotational speed [rpm]"]),
        "torque": float(row["Torque [Nm]"]),
        "tool_wear": int(row["Tool wear [min]"])
    }

    # simulate anomaly
    if random.random() < 0.1:
        data["temperature"] += random.uniform(20, 50)

    producer.send('sensor-data', data)
    print("Sent:", data)

    time.sleep(1)