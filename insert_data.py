# insert_data.py
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Connect to database
conn = psycopg2.connect(
    dbname="bank_reviews",
    user="postgres",
    password="Sharon08#2939",  
    host="localhost",
    port="5432"
)

cur = conn.cursor()

# Clear existing data (optional)
cur.execute("DELETE FROM reviews")  # Delete reviews first to avoid FK constraint
cur.execute("DELETE FROM banks")

# Insert banks and capture their actual bank_id values
banks = [
    ('Commercial Bank of Ethiopia', 'CBE Birr'),
    ('Bank of Abyssinia', 'BOA Mobile'), 
    ('Dashen Bank', 'Dashen Mobile')
]

bank_ids = []  # Store real bank_id values returned by PostgreSQL
for bank in banks:
    cur.execute(
        "INSERT INTO banks (bank_name, app_name) VALUES (%s, %s) RETURNING bank_id",
        bank
    )
    bank_id = cur.fetchone()[0]
    bank_ids.append(bank_id)
    print(f"Inserted {bank[0]} with ID: {bank_id}")

# Insert sample reviews using correct bank_id
np.random.seed(42)
bank_names = ['CBE', 'BOA', 'Dashen']
total_reviews = 0

for bank_idx in range(len(banks)):
    bank_id = bank_ids[bank_idx]
    bank_short = bank_names[bank_idx]
    
    for i in range(400):  # 400 reviews per bank
        # Generate realistic rating per bank
        if bank_idx == 0:      # CBE
            rating = np.random.normal(3.8, 0.8)
        elif bank_idx == 1:    # BOA
            rating = np.random.normal(4.1, 0.7)
        else:                  # Dashen
            rating = np.random.normal(3.5, 0.9)
        
        rating = max(1.0, min(5.0, round(rating, 1)))
        
        # Determine sentiment based on rating
        if rating >= 4:
            sentiment = 'positive'
            sentiment_score = np.random.uniform(0.3, 1.0)
            review_text = "Great app! Very user friendly and reliable."
        elif rating <= 2:
            sentiment = 'negative'
            sentiment_score = np.random.uniform(-1.0, -0.3)
            review_text = "App crashes frequently and very slow."
        else:
            sentiment = 'neutral'
            sentiment_score = np.random.uniform(-0.2, 0.2)
            review_text = "Average app, does the job."
        
        # Add bank-specific suffix
        review_text += f" - {bank_short} mobile banking"
        
        # Random date in the last year
        days_ago = np.random.randint(0, 365)
        review_date = datetime.now().date() - timedelta(days=days_ago)
        
        # Insert review
        cur.execute("""
            INSERT INTO reviews 
            (bank_id, review_text, rating, review_date, sentiment_label, sentiment_score, source)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (bank_id, review_text, rating, review_date, sentiment, sentiment_score, 'Google Play Store'))
        
        total_reviews += 1

conn.commit()
print(f"✅ Successfully inserted {total_reviews} reviews and {len(banks)} banks.")
cur.close()
conn.close()