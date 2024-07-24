from sqlalchemy import String, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from fastapi_app.google_search import google_search_function
from fastapi_app.postgres_models import Package


class PostgresSearcher:
    def __init__(
        self,
        engine,
    ):
        self.async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

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

    async def simple_sql_search(self, filters: list[dict]) -> list[Package]:
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
            results = (await session.execute(text(sql).columns(url=String))).fetchall()

            # Convert results to Package models
            items = []
            for result in results:
                item_url = result.url
                package = await session.execute(select(Package).where(Package.url == item_url))
                items.append(package.scalar())
            return items

    async def google_search(self, query_text: str, top: int = 3) -> list[str]:
        """
        Search items by query text using Google search.
        """
        results = google_search_function(query_text)
        async with self.async_session_maker() as session:
            items = []
            for result in results:
                package = await session.execute(select(Package).where(Package.url == result))
                package = package.scalar()
                if package:
                    items.append(package)
            return items[:top]
