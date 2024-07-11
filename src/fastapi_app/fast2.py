import argparse
import asyncio
import logging
import os
import time

import numpy as np
import pandas as pd
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv
from sqlalchemy import inspect, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker
from tqdm import tqdm

from fastapi_app.openai_clients import create_openai_embed_client
from fastapi_app.postgres_engine import create_postgres_engine_from_args, create_postgres_engine_from_env
from fastapi_app.postgres_models import Item

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragapp")

EMBEDDING_FIELDS = [
    "package_name",
    "package_picture",
    "url",
    "installment_month",
    "installment_limit",
    "shop_name",
    "category",
    "category_tags",
    "preview_1_10",
    "selling_point",
    "meta_keywords",
    "brand",
    "min_max_age",
    "locations",
    "meta_description",
    "price_details",
    "package_details",
    "important_info",
    "payment_booking_info",
    "general_info",
    "early_signs_for_diagnosis",
    "how_to_diagnose",
    "hdcare_summary",
    "common_question",
    "know_this_disease",
    "courses_of_action",
    "signals_to_proceed_surgery",
    "get_to_know_this_surgery",
    "comparisons",
    "getting_ready",
    "recovery",
    "side_effects",
    "review_4_5_stars",
    "brand_option_in_thai_name",
    "faq",
]


def get_to_str_method(item, field):
    method_name = f"to_str_for_embedding_{field}"
    return getattr(item, method_name, None)


def convert_to_int(value):
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def convert_to_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def convert_to_str(value):
    if value is None:
        return None
    return str(value)


async def verify_database_connection(engine):
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text("SELECT 1"))
            logger.info(f"Database connection test result: {result.scalar()}")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


async def verify_table_exists(engine, table_name):
    try:
        async with engine.begin() as conn:

            def check_table(connection):
                inspector = inspect(connection)
                return table_name in inspector.get_table_names()

            exists = await conn.run_sync(check_table)
            if exists:
                logger.info(f"Table '{table_name}' exists in the database.")
            else:
                logger.error(f"Table '{table_name}' does not exist in the database.")
                raise ValueError(f"Table '{table_name}' not found")
    except Exception as e:
        logger.error(f"Error verifying table existence: {e}")
        raise


async def count_records(engine, table_name):
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            count = result.scalar()
            logger.info(f"Total records in {table_name} after processing: {count}")
    except Exception as e:
        logger.error(f"Error counting records: {e}")


async def fetch_existing_records(session, batch_size=1000):
    offset = 0
    existing_records = {}
    while True:
        query = select(Item).offset(offset).limit(batch_size)
        result = await session.execute(query)
        items = result.scalars().all()
        if not items:
            break
        for item in items:
            existing_records[item.url] = item
        offset += batch_size
        logger.info(f"Fetched {len(items)} records, offset now {offset}")
    return existing_records


async def seed_and_update_embeddings(engine):
    start_time = time.time()

    # Verify database connection
    await verify_database_connection(engine)

    # Verify table exists
    table_name = "packages_all_staging"  # Replace with your actual table name
    await verify_table_exists(engine, table_name)

    async_session = async_sessionmaker(engine, expire_on_commit=False)

    current_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path = os.path.join(current_dir, "packages.csv")

    try:
        df = pd.read_csv(csv_path, delimiter=",", quotechar='"', escapechar="\\", on_bad_lines="skip", encoding="utf-8")
        logger.info(f"Read {len(df)} rows from CSV file")
        logger.info(f"First few rows: {df.head().to_dict()}")
        if len(df) == 0:
            logger.error("CSV file is empty. No data to process.")
            return
    except pd.errors.ParserError as e:
        logger.error(f"Error reading CSV file: {e}")
        return
    except FileNotFoundError:
        logger.error(f"CSV file not found at path: {csv_path}")
        return

    logger.info("CSV file read successfully. Processing data...")

    str_columns = df.select_dtypes(include=[object]).columns
    df[str_columns] = df[str_columns].replace({np.nan: None})

    num_columns = df.select_dtypes(include=([np.number])).columns
    df[num_columns] = df[num_columns].replace({np.nan: None})

    records = df.to_dict(orient="records")
    new_records = {record["url"]: record for record in records}

    logger.info(f"Processed {len(new_records)} records from CSV")

    logger.info("Fetching existing records from the database...")

    async with async_session() as session:
        existing_records = await fetch_existing_records(session)

    logger.info(f"Fetched {len(existing_records)} existing records.")

    azure_credential = DefaultAzureCredential()
    openai_embed_client, openai_embed_model, openai_embed_dimensions = await create_openai_embed_client(
        azure_credential
    )

    logger.info("Starting to insert, update, or delete records in the database...")
    logger.info(f"Processing {len(new_records)} new records")

    async with async_session() as session:
        # Test insertion
        try:
            test_item = Item(url="test_url", package_name="Test Package", price=100.0)
            session.add(test_item)
            await session.commit()
            logger.info("Test insertion successful")
        except Exception as e:
            logger.error(f"Test insertion failed: {e}")
            return

        insertion_count = 0
        update_count = 0
        error_count = 0

        for url, record in tqdm(new_records.items(), desc="Processing new records"):
            try:
                record["price"] = convert_to_float(record.get("price"))
                if record["price"] is None:
                    logger.warning(f"Skipping record with invalid price: {record}")
                    continue

                existing_item = existing_records.get(url)

                if existing_item:
                    if existing_item.price != record["price"]:
                        existing_item.price = record["price"]
                        session.merge(existing_item)
                        update_count += 1
                        logger.info(f"Updated record with URL {url}")
                else:
                    logger.info(f"Attempting to insert new record with URL {url}")
                    item_data = {key: value for key, value in record.items() if key in Item.__table__.columns}
                    item = Item(**item_data)
                    session.add(item)
                    insertion_count += 1
                    logger.info(f"New record added for URL {url}")

                if (insertion_count + update_count) % 100 == 0:
                    await session.commit()
                    logger.info(f"Committed batch: {insertion_count} insertions, {update_count} updates")

            except Exception as e:
                logger.error(f"Error processing record with URL {url}: {e}")
                error_count += 1
                continue

        try:
            await session.commit()
            logger.info(f"Final commit: {insertion_count} insertions, {update_count} updates, {error_count} errors")
        except Exception as e:
            logger.error(f"Error during final commit: {e}")

    # Count records at the end
    await count_records(engine, table_name)

    logger.info("All records processed.")
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info(f"Total time taken: {elapsed_time:.2f} seconds")

    # Close the Azure credential client
    await azure_credential.close()

    # Close the OpenAI client if needed
    await openai_embed_client.close()


async def main():
    parser = argparse.ArgumentParser(description="Seed database with CSV data")
    parser.add_argument("--host", type=str, help="Postgres host")
    parser.add_argument("--username", type=str, help="Postgres username")
    parser.add_argument("--password", type=str, help="Postgres password")
    parser.add_argument("--database", type=str, help="Postgres database")
    parser.add_argument("--sslmode", type=str, help="Postgres sslmode")

    args = parser.parse_args()
    if args.host is None:
        engine = await create_postgres_engine_from_env()
    else:
        engine = await create_postgres_engine_from_args(args)

    # Log the connection parameters (be careful not to log sensitive information)
    logger.info(f"Connecting to database: {engine.url.database} on host: {engine.url.host}")

    await seed_and_update_embeddings(engine)
    await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    load_dotenv(override=True)
    asyncio.run(main())
