import requests
import pandas as pd
import os
import time

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

PAGE_SIZE = 100
MAX_PAGES = 5
MAX_RETRIES = 3

all_records = []

page_token = None
page_number = 1

while True:

    params = {
        "query.cond": "cancer",
        "pageSize": PAGE_SIZE
    }

    if page_token:
        params["pageToken"] = page_token

    print(f"\nFetching Page {page_number}...")

    success = False

    for attempt in range(MAX_RETRIES):

        try:

            response = requests.get(
                BASE_URL,
                params=params,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            success = True

            break

        except Exception as e:

            print(
                f"Attempt {attempt + 1}/{MAX_RETRIES} failed"
            )

            print(e)

            if attempt < MAX_RETRIES - 1:
                print("Retrying in 5 seconds...\n")
                time.sleep(5)

    if not success:

        print(
            f"Skipping page {page_number}"
        )

        break

    studies = data.get(
        "studies",
        []
    )

    print(
        f"Retrieved {len(studies)} studies"
    )

    for study in studies:

        protocol = study.get(
            "protocolSection",
            {}
        )

        identification = protocol.get(
            "identificationModule",
            {}
        )

        status = protocol.get(
            "statusModule",
            {}
        )

        conditions = protocol.get(
            "conditionsModule",
            {}
        )

        description = protocol.get(
            "descriptionModule",
            {}
        )

        design = protocol.get(
            "designModule",
            {}
        )

        record = {

            "nct_id":
                identification.get(
                    "nctId"
                ),

            "title":
                identification.get(
                    "briefTitle"
                ),

            "condition":
                ", ".join(
                    conditions.get(
                        "conditions",
                        []
                    )
                ),

            "summary":
                description.get(
                    "briefSummary",
                    ""
                ),

            "phase":
                ", ".join(
                    design.get(
                        "phases",
                        []
                    )
                ),

            "status":
                status.get(
                    "overallStatus",
                    ""
                )
        }

        all_records.append(
            record
        )

    page_token = data.get(
        "nextPageToken"
    )

    if not page_token:

        print(
            "\nNo more pages available."
        )

        break

    if page_number >= MAX_PAGES:

        print(
            f"\nReached MAX_PAGES={MAX_PAGES}"
        )

        break

    page_number += 1

print(
    f"\nTotal Raw Records: {len(all_records)}"
)

df = pd.DataFrame(
    all_records
)

df = df.drop_duplicates(
    subset="nct_id"
)

df["search_text"] = (
    df["title"].fillna("")
    + " "
    + df["condition"].fillna("")
    + " "
    + df["summary"].fillna("")
)

os.makedirs(
    "data",
    exist_ok=True
)

df.to_csv(
    "data/trials.csv",
    index=False
)

print(
    f"\nFinal Unique Trials: {len(df)}"
)

print(
    f"Columns: {len(df.columns)}"
)

print(
    "\nSample Search Text:\n"
)

print(
    df["search_text"]
    .iloc[0][:500]
)

print(
    "\nSaved: data/trials.csv"
)