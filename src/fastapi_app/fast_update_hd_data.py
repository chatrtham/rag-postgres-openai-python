import argparse
import asyncio
import logging
import os
import time

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from sqlalchemy import select, text, delete
from sqlalchemy.ext.asyncio import async_sessionmaker
from tqdm import tqdm
from azure.identity.aio import DefaultAzureCredential

from fastapi_app.embeddings import compute_text_embedding
from fastapi_app.openai_clients import create_openai_embed_client
from fastapi_app.postgres_engine import (
    create_postgres_engine_from_args,
    create_postgres_engine_from_env,
)
from fastapi_app.postgres_models import Item

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragapp")

EMBEDDING_FIELDS = [
    'package_name', 'package_picture', 'url', 'installment_month', 'installment_limit',
    'shop_name', 'category', 'category_tags','preview_1_10', 'selling_point','meta_keywords',
    'brand', 'min_max_age','locations', 'meta_description', 'price_details','package_details',
    'important_info', 'payment_booking_info', 'general_info', 'early_signs_for_diagnosis',
    'how_to_diagnose', 'hdcare_summary', 'common_question', 'know_this_disease',
    'courses_of_action', 'signals_to_proceed_surgery', 'get_to_know_this_surgery',
    'comparisons', 'getting_ready', 'recovery', 'side_effects', 'review_4_5_stars',
    'brand_option_in_thai_name', 'faq'
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
    logger.info("Checking if the packages table exists...")
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                """
                SELECT EXISTS 
                (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'packages')
                """
            )
        )
        if not result.scalar():
            logger.error("Packages table does not exist. Please run the database setup script first.")
            return

    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        csv_path = os.path.join(current_dir, "packages.csv")

        try:
            df = pd.read_csv(csv_path, delimiter=',', quotechar='"', escapechar='\\', on_bad_lines='skip', encoding='utf-8')
        except pd.errors.ParserError as e:
            logger.error(f"Error reading CSV file: {e}")
            return

        logger.info("CSV file read successfully. Processing data...")

        str_columns = df.select_dtypes(include=[object]).columns
        df[str_columns] = df[str_columns].replace({np.nan: None})

        num_columns = df.select_dtypes(include=([np.number])).columns
        df[num_columns] = df[num_columns].replace({np.nan: None})

        records = df.to_dict(orient='records')
        new_records = {record["url"]: record for record in records}

        logger.info("Fetching existing records from the database...")

        existing_records = await fetch_existing_records(session)

        logger.info(f"Fetched {len(existing_records)} existing records.")

        azure_credential = DefaultAzureCredential()
        openai_embed_client, openai_embed_model, openai_embed_dimensions = await create_openai_embed_client(azure_credential)

        logger.info("Starting to insert, update, or delete records in the database...")

        for url, record in tqdm(new_records.items(), desc="Processing new records"):
            try:
                record["price"] = convert_to_float(record.get("price"))
                record["cash_discount"] = convert_to_float(record.get("cash_discount"))
                record["price_after_cash_discount"] = convert_to_float(record.get("price_after_cash_discount"))
                record["price_to_reserve_for_this_package"] = convert_to_float(record.get("price_to_reserve_for_this_package"))
                record["brand_ranking_position"] = convert_to_int(record.get("brand_ranking_position"))

                if record["price"] is None:
                    logger.error(f"Skipping record with invalid price: {record}")
                    continue

                existing_item = existing_records.get(url)

                if existing_item:
                    # Update only the price if there is a change
                    if existing_item.price != record["price"]:
                        existing_item.price = record["price"]
                        session.merge(existing_item)
                        await session.commit()
                        logger.info(f"Updated price for existing record with URL {url}")
                else:
                    # Insert new item
                    item_data = {key: value for key, value in record.items() if key in Item.__table__.columns}
                    for field in EMBEDDING_FIELDS:
                        item_data[f'embedding_{field}'] = None

                    for key, value in item_data.items():
                        if key not in ["price", "cash_discount", "price_after_cash_discount", "price_to_reserve_for_this_package", "brand_ranking_position"]:
                            item_data[key] = convert_to_str(value)

                    item = Item(**item_data)

                    # Generate embeddings for the new item
                    for field in EMBEDDING_FIELDS:
                        to_str_method = get_to_str_method(item, field)
                        if to_str_method:
                            field_value = to_str_method()
                            if field_value:
                                try:
                                    embedding = await compute_text_embedding(
                                        field_value,
                                        openai_client=openai_embed_client,
                                        embed_model=openai_embed_model,
                                        embedding_dimensions=openai_embed_dimensions,
                                    )
                                    setattr(item, f'embedding_{field}', embedding)
                                    logger.info(f"Updated embedding for {field} of item {item.url}")
                                except Exception as e:
                                    logger.error(f"Error updating embedding for {field} of item {item.url}: {e}")

                    session.merge(item)
                    await session.commit()
                    logger.info(f"Inserted new record with URL {url}")

            except Exception as e:
                logger.error(f"Error processing record with URL {url}: {e}")
                await session.rollback()
                continue

        # Delete rows that are not in the new CSV
        for url in tqdm(existing_records.keys() - new_records.keys(), desc="Deleting outdated records"):
            try:
                await session.execute(delete(Item).where(Item.url == url))
                await session.commit()
                logger.info(f"Deleted outdated record with URL {url}")
            except Exception as e:
                logger.error(f"Error deleting record with URL {url}: {e}")
                await session.rollback()

        logger.info("All records processed successfully.")
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Total time taken: {elapsed_time:.2f} seconds")

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

    await seed_and_update_embeddings(engine)
    await engine.dispose()

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    load_dotenv(override=True)
    asyncio.run(main())
