import os
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd
from azure.cosmos import CosmosClient, PartitionKey

# Load .env values from the project root or current working directory
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

# Cosmos DB details from environment
COSMOS_URI = os.getenv("COSMOS_URI")
COSMOS_KEY = os.getenv("COSMOS_KEY")
if not COSMOS_URI or not COSMOS_KEY:
    raise RuntimeError("Missing COSMOS_URI or COSMOS_KEY in .env")

DATABASE_NAME = "guardian-db"
CONTAINER_NAME = "insurance-data"

# Excel file
EXCEL_FILE = "Gradhack_Insure_Data_CLEANED.xlsx"


# Connect to Cosmos DB
client = CosmosClient(COSMOS_URI, credential=COSMOS_KEY)

database = client.create_database_if_not_exists(
    id=DATABASE_NAME
)

container = database.create_container_if_not_exists(
    id=CONTAINER_NAME,
    partition_key=PartitionKey(path="/id")
)

print("Connected to Cosmos DB")


# Read Excel
df = pd.read_excel(EXCEL_FILE)

print(f"Loaded {len(df)} rows from Excel")


# Upload rows
for index, row in df.iterrows():

    item = row.to_dict()

    # Convert Excel dates into strings
    for key, value in item.items():
        if pd.isna(value):
            item[key] = None
        elif isinstance(value, pd.Timestamp):
            item[key] = value.isoformat()

    # Cosmos requires an id field
    item["id"] = str(index)

    container.upsert_item(item)

    print(f"Uploaded row {index + 1}/{len(df)}")


print("Migration complete!")