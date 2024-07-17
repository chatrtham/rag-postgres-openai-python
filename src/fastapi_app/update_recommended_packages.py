import asyncio
import logging
import pandas as pd
from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
from tqdm.asyncio import tqdm_asyncio

from embeddings import compute_text_embedding
from openai_clients import create_openai_embed_client
from postgres_engine import create_postgres_engine_from_env
from postgres_models import Item

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_FIELDS = [
    'package_name', 'locations', 'category', 'category_tags', 'meta_keywords', 
    'brand', 'meta_description', 'price_details', 'package_details', 
    'important_info', 'payment_booking_info', 'general_info', 
    'early_signs_for_diagnosis', 'how_to_diagnose', 'hdcare_summary', 
    'common_question', 'know_this_disease', 'courses_of_action', 
    'signals_to_proceed_surgery', 'get_to_know_this_surgery', 'comparisons', 
    'getting_ready', 'recovery', 'side_effects', 'review_4_5_stars', 
    'brand_option_in_thai_name', 'faq'
]

def get_to_str_method(item, field):
    method_name = f"to_str_for_embedding_{field}"
    return getattr(item, method_name, None)

async def fetch_items(session, urls, offset, batch_size):
    return (
        await session.scalars(
            select(Item)
            .where(Item.url.in_(urls))
            .where(Item.is_recommended.is_(False))
            .offset(offset)
            .limit(batch_size)
        )
    ).all()

async def process_batch(async_session_maker, items, openai_embed_client, openai_embed_model, openai_embed_dimensions):
    async with async_session_maker() as session:
        async with session.begin():
            for item in items:
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
                item.is_recommended = True
                session.add(item)
            await session.commit()

async def update_embeddings(batch_size=100):
    engine = await create_postgres_engine_from_env()
    azure_credential = DefaultAzureCredential()
    openai_embed_client, openai_embed_model, openai_embed_dimensions = await create_openai_embed_client(azure_credential)

    async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    # Read CSV file
    df = pd.read_csv('recommended_packages.csv')
    urls = df['package url'].tolist()
    logger.info(f"Total URLs in CSV: {len(urls)}")
    unique_urls = list(set(urls))
    logger.info(f"Unique URLs: {len(unique_urls)}")

    async with async_session_maker() as fetch_session:
        total_items = await fetch_session.scalar(
            select(func.count(Item.url))
            .where(Item.url.in_(unique_urls))
            .where(Item.is_recommended.is_(False))
        )
        logger.info(f"Found {total_items} items to process in database.")

        offset = 0
        with tqdm_asyncio(total=total_items, desc="Updating embeddings") as pbar:
            while offset < total_items:
                async with async_session_maker() as fetch_session:
                    items = await fetch_items(fetch_session, unique_urls, offset, batch_size)
                    if not items:
                        break

                    # Expunge items from the original session
                    for item in items:
                        fetch_session.expunge(item)

                    logger.info(f"Processing batch of {len(items)} items (offset {offset})")
                    await process_batch(async_session_maker, items, openai_embed_client, openai_embed_model, openai_embed_dimensions)
                    offset += batch_size
                    pbar.update(len(items))

    await engine.dispose()
    await azure_credential.close()  # Ensure the Azure credential client session is closed

if __name__ == "__main__":
    asyncio.run(update_embeddings())
