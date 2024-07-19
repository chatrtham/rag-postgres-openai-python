from openai import AsyncOpenAI
from pgvector.utils import to_db
from sqlalchemy import String, Float, Integer, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from fastapi_app.embeddings import compute_text_embedding
from fastapi_app.postgres_models import Item
from fastapi_app.google_search import google_search_function


class PostgresSearcher:
    def __init__(
        self,
        engine,
        openai_embed_client: AsyncOpenAI,
        embed_deployment: str | None,  # Not needed for non-Azure OpenAI or for retrieval_mode="text"
        embed_model: str,
        embed_dimensions: int,
    ):
        self.async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
        self.openai_embed_client = openai_embed_client
        self.embed_model = embed_model
        self.embed_deployment = embed_deployment
        self.embed_dimensions = embed_dimensions

    def build_filter_clause(self, filters, use_or=False) -> tuple[str, str]:
        if filters is None:
            return "", ""
        filter_clauses = []
        for filter in filters:
            if isinstance(filter["value"], str):
                filter["value"] = f"'{filter['value']}'"
            filter_clauses.append(f"{filter['column']} {filter['comparison_operator']} {filter['value']}")
        filter_clause = f" {'OR' if use_or else 'AND'} ".join(filter_clauses)
        if len(filter_clause) > 0:
            return f"WHERE {filter_clause}", f"AND {filter_clause}"
        return "", ""

    async def simple_sql_search(
        self, 
        filters: list[dict]
    ) -> list[Item]:
        """
        Search items by simple SQL query with filters.
        """
        filter_clause_where, _ = self.build_filter_clause(filters, use_or=True)
        sql = f"""
        SELECT url FROM packages_all
        {filter_clause_where}
        LIMIT 10
        """
        
        async with self.async_session_maker() as session:
            results = (
                await session.execute(
                    text(sql).columns(url=String)
                )
            ).fetchall()

            # Convert results to Item models
            items = []
            for result in results:
                item_url = result.url
                item = await session.execute(select(Item).where(Item.url == item_url))
                items.append(item.scalar())
            return items
        
    async def google_search(self, query_text: str, top: int) -> list[str]:
        """
        Search items by query text using Google search.
        """
        results = google_search_function(query_text)
        async with self.async_session_maker() as session:
            items = []
            for result in results:
                item = await session.execute(select(Item).where(Item.url == result))
                item = item.scalar()
                if item:
                    items.append(item)
            return items[:top]
        
    
    # async def get_product_cards_info(self, urls: list[str]) -> list[dict]:
    #     """
    #     Fetch detailed information about items using their URLs as identifiers.
    #     """
    #     sql = """
    #     SELECT package_name, package_picture, url, price FROM packages_all WHERE url = ANY(:urls)
    #     """
        
    #     async with self.async_session_maker() as session:
    #         results = (
    #             await session.execute(
    #                 text(sql), {"urls": urls}
    #             )
    #         ).fetchall()

    #         # Convert results to dictionaries
    #         items = []
    #         for result in results:
    #             items.append(dict(result._mapping))
    #         return items