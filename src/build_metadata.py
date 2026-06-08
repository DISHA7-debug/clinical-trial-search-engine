import os
import sqlite3

import pandas as pd


DB_PATH = "data/metadata.db"
CSV_PATH = "data/trials.csv"


def create_table(cursor):

    cursor.execute("""
        DROP TABLE IF EXISTS trials_metadata
    """)

    cursor.execute("""
        CREATE TABLE trials_metadata (
            nct_id TEXT PRIMARY KEY,
            title TEXT,
            condition TEXT,
            phase TEXT,
            status TEXT
        )
    """)


def clean_value(value):

    if pd.isna(value):
        return "UNKNOWN"

    return str(value)


def prepare_records(df):

    records = []

    for _, row in df.iterrows():

        records.append(
            (
                clean_value(row["nct_id"]),
                clean_value(row["title"]),
                clean_value(row["condition"]),
                clean_value(row["phase"]),
                clean_value(row["status"])
            )
        )

    return records


def main():

    print("Loading trials...")

    df = pd.read_csv(
        CSV_PATH
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    conn = sqlite3.connect(
        DB_PATH
    )

    cursor = conn.cursor()

    create_table(
        cursor
    )

    records = prepare_records(
        df
    )

    cursor.executemany(
        """
        INSERT INTO trials_metadata
        VALUES (?, ?, ?, ?, ?)
        """,
        records
    )

    conn.commit()

    cursor.execute(
        "SELECT COUNT(*) FROM trials_metadata"
    )

    count = cursor.fetchone()[0]

    print(
        f"Rows inserted: {count}"
    )

    conn.close()

    print(
        f"\nSaved: {DB_PATH}"
    )


if __name__ == "__main__":
    main()