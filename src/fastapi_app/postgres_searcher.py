from openai import AsyncOpenAI
from pgvector.utils import to_db
from sqlalchemy import String, Float, Integer, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from fastapi_app.embeddings import compute_text_embedding
from fastapi_app.postgres_models import Item


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

    async def hybrid_search(
        self,
        query_text: str | None,
        query_vector: list[float] | list,
        top: int = 5,
        filters: list[dict] | None = None,
    ):
        filter_clause_where, filter_clause_and = self.build_filter_clause(filters)

        vector_query = f"""
            WITH closest_embedding AS (
                SELECT 
                    url,
                    LEAST(
                        embedding_package_name <=> :embedding, 
                        COALESCE(embedding_locations <=> :embedding, 1)
                    ) AS min_distance
                FROM 
                    packages
                {filter_clause_where}
            )
            SELECT 
                url, 
                RANK() OVER (ORDER BY min_distance) AS rank
            FROM 
                closest_embedding
            ORDER BY 
                min_distance
            LIMIT 20
            """

        fulltext_query = f"""
            SELECT url, RANK () OVER (ORDER BY ts_rank_cd(
                to_tsvector('thai', COALESCE(package_name, '')) ||
                to_tsvector('thai', COALESCE(locations, '')), query) DESC)
            FROM packages, plainto_tsquery('thai', :query) query
            WHERE (
                to_tsvector('thai', COALESCE(package_name, '')) ||
                to_tsvector('thai', COALESCE(locations, ''))
            ) @@ query {filter_clause_and}
            ORDER BY ts_rank_cd(
                to_tsvector('thai', COALESCE(package_name, '')) ||
                to_tsvector('thai', COALESCE(locations, '')), query) DESC
            LIMIT 20
        """

        hybrid_query = f"""
        WITH vector_search AS (
            {vector_query}
        ),
        fulltext_search AS (
            {fulltext_query}
        )
        SELECT
            COALESCE(vector_search.url, fulltext_search.url) AS url,
            COALESCE(1.0 / (:k + vector_search.rank), 0.0) +
            COALESCE(1.0 / (:k + fulltext_search.rank), 0.0) AS score
        FROM vector_search
        FULL OUTER JOIN fulltext_search ON vector_search.url = fulltext_search.url
        ORDER BY score DESC
        LIMIT 20
        """

        if query_text is not None and len(query_vector) > 0:
            sql = text(hybrid_query).columns(url=String, score=Float)
        elif len(query_vector) > 0:
            sql = text(vector_query).columns(url=String, rank=Integer)
        elif query_text is not None:
            sql = text(fulltext_query).columns(url=String, rank=Integer)
        else:
            raise ValueError("Both query text and query vector are empty")

        async with self.async_session_maker() as session:
            results = (
                await session.execute(
                    sql,
                    {"embedding": to_db(query_vector), "query": query_text, "k": 60},
                )
            ).fetchall()

            # Convert results to Item models
            items = []
            for url, _ in results[:top]:
                item = await session.execute(select(Item).where(Item.url == url))
                items.append(item.scalar())
            return items

    async def search_and_embed(
        self,
        query_text: str,
        top: int = 5,
        enable_vector_search: bool = False,
        enable_text_search: bool = False,
        filters: list[dict] | None = None,
    ) -> list[Item]:
        """
        Search items by query text. Optionally converts the query text to a vector if enable_vector_search is True.
        """
        vector: list[float] = []
        if enable_vector_search:
            vector = await compute_text_embedding(
                query_text,
                self.openai_embed_client,
                self.embed_model,
                self.embed_deployment,
                self.embed_dimensions,
            )
        if not enable_text_search:
            query_text = None

        return await self.hybrid_search(query_text, vector, top, filters)

    async def simple_sql_search(
        self, 
        filters: list[dict]
    ) -> list[Item]:
        """
        Search items by simple SQL query with filters.
        """
        filter_clause_where, _ = self.build_filter_clause(filters, use_or=True)
        sql = f"""
        SELECT url FROM packages_all_staging
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
        
    
    # async def get_product_cards_info(self, urls: list[str]) -> list[dict]:
    #     """
    #     Fetch detailed information about items using their URLs as identifiers.
    #     """
    #     sql = """
    #     SELECT package_name, package_picture, url, price FROM packages_all_staging WHERE url = ANY(:urls)
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