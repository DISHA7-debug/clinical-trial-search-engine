import sqlite3
import pandas as pd

conn = sqlite3.connect(
    "data/metadata.db"
)

query = """
SELECT
    title,
    condition,
    phase,
    status
FROM trials_metadata
WHERE condition LIKE '%Lung%'
LIMIT 10
"""

df = pd.read_sql_query(
    query,
    conn
)

print(df)

conn.close()