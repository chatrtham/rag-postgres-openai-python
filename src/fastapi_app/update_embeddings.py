import asyncio
import logging

from azure.identity.aio import DefaultAzureCredential
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from tqdm import tqdm

from fastapi_app.embeddings import compute_text_embedding
from fastapi_app.openai_clients import create_openai_embed_client
from fastapi_app.postgres_engine import create_postgres_engine_from_env
from fastapi_app.postgres_models import Item

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_FIELDS = [
    'package_name', 'package_picture', 'url', 'installment_month', 'installment_limit',
    'shop_name', 'category', 'category_tags',
    'preview_1_10', 'selling_point', 'meta_keywords', 'brand', 'min_max_age',
    'locations', 'meta_description','price_details', 'package_details', 'important_info',
    'payment_booking_info', 'general_info', 'early_signs_for_diagnosis', 'how_to_diagnose',
    'hdcare_summary', 'common_question', 'know_this_disease', 'courses_of_action',
    'signals_to_proceed_surgery', 'get_to_know_this_surgery', 'comparisons', 'getting_ready',
    'recovery', 'side_effects', 'review_4_5_stars', 'brand_option_in_thai_name', 'faq'
]

def get_to_str_method(item, field):
    method_name = f"to_str_for_embedding_{field}"
    return getattr(item, method_name, None)

async def update_embeddings():
    engine = await create_postgres_engine_from_env()
    azure_credential = DefaultAzureCredential()
    openai_embed_client, openai_embed_model, openai_embed_dimensions = await create_openai_embed_client(azure_credential)

    async_session_maker = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with async_session_maker() as session:
        try:
            async with session.begin():
                items = (await session.scalars(select(Item))).all()
                logger.info(f"Found {len(items)} items to process.")

                for item in tqdm(items, desc="Processing items"):
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

                    session.add(item)
                await session.commit()
        finally:
            await session.close()
            await engine.dispose()
            await azure_credential.close()  # Ensure the Azure credential client session is closed

if __name__ == "__main__":
    asyncio.run(update_embeddings())
