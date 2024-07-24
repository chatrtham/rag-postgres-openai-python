import argparse
import asyncio
import logging
import os

import numpy as np
import pandas as pd
import sqlalchemy.exc
from dotenv import load_dotenv
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import async_sessionmaker
from tqdm import tqdm

from fastapi_app.postgres_engine import (
    create_postgres_engine_from_args,
    create_postgres_engine_from_env,
)
from fastapi_app.postgres_models import Package

logger = logging.getLogger("ragapp")


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


async def seed_data(engine):
    logger.info("Checking if the packages_all_staging table exists...")
    async with engine.begin() as conn:
        result = await conn.execute(
            text(
                """
                SELECT EXISTS 
                (SELECT 1 FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'packages_all_staging')
                """  # noqa
            )
        )
        if not result.scalar():
            logger.error("Packages table does not exist. Please run the database setup script first.")
            return

    async with async_sessionmaker(engine, expire_on_commit=False)() as session:
        current_dir = os.path.dirname(os.path.realpath(__file__))
        csv_path = os.path.join(current_dir, "packages.csv")

        try:
            df = pd.read_csv(
                csv_path, delimiter=",", quotechar='"', escapechar="\\", on_bad_lines="skip", encoding="utf-8"
            )
        except pd.errors.ParserError as e:
            logger.error(f"Error reading CSV file: {e}")
            return

        str_columns = df.select_dtypes(include=[object]).columns
        df[str_columns] = df[str_columns].replace({np.nan: None})

        num_columns = df.select_dtypes(include=[np.number]).columns
        df[num_columns] = df[num_columns].replace({np.nan: None})

        records = df.to_dict(orient="records")

        logger.info("Starting to insert records into the database...")
        for record in tqdm(records, desc="Inserting records"):
            try:
                record["url"] = convert_to_str(record["url"])
                if record["url"] is None:
                    logger.error(f"Skipping record with invalid url: {record}")
                    continue

                if "price" in record:
                    record["price"] = convert_to_float(record["price"])
                if "cash_discount" in record:
                    record["cash_discount"] = convert_to_float(record["cash_discount"])
                if "price_after_cash_discount" in record:
                    record["price_after_cash_discount"] = convert_to_float(record["price_after_cash_discount"])
                if "price_to_reserve_for_this_package" in record:
                    record["price_to_reserve_for_this_package"] = convert_to_float(
                        record["price_to_reserve_for_this_package"]
                    )
                if "brand_ranking_position" in record:
                    record["brand_ranking_position"] = convert_to_int(record["brand_ranking_position"])

                package = await session.execute(select(Package).filter(Package.url == record["url"]))
                if package.scalars().first():
                    continue

                item_data = {key: value for key, value in record.items() if key in Package.__table__.columns}


                for key, value in item_data.items():
                    if key not in [
                        "price",
                        "price_to_reserve_for_this_package",
                        "cash_discount",
                        "price_after_cash_discount",
                    ]:
                        item_data[key] = convert_to_str(value)

                package = Package(**item_data)
                session.add(package)

            except Exception as e:
                logger.error(f"Error inserting record with url {record['url']}: {e}")
                await session.rollback()
                continue

        try:
            await session.commit()
            logger.info("All records inserted successfully.")
        except sqlalchemy.exc.IntegrityError as e:
            logger.error(f"Integrity error during commit: {e}")


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

    await seed_data(engine)
    await engine.dispose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)
    load_dotenv(override=True)
    asyncio.run(main())
