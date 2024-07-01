import base64
import json

import fastapi
from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from fastapi_app.api_models import ChatRequest
from fastapi_app.globals import global_storage
from fastapi_app.postgres_models import Item
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_advanced import AdvancedRAGChat
from fastapi_app.rag_simple import SimpleRAGChat

# Initialize FastAPI app
app = FastAPI()

# Configure OpenTelemetry
resource = Resource(attributes={"service.name": "peen-gpt4o-ragapp-fastapi"})
trace.set_tracer_provider(TracerProvider(resource=resource))
tracer = trace.get_tracer(__name__)

# Set up OTLP exporter
exporter = AzureMonitorTraceExporter.from_connection_string("InstrumentationKey=b12bd989-0aac-4ac5-b59d-55b708d6e691;IngestionEndpoint=https://eastus-8.in.applicationinsights.azure.com/;LiveEndpoint=https://eastus.livediagnostics.monitor.azure.com/;ApplicationId=11040ff0-0c04-41e9-8d67-9a37a235e245")
span_processor = BatchSpanProcessor(exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Instrument SQLAlchemy
SQLAlchemyInstrumentor().instrument(engine=global_storage.engine)

# Instrument requests library
RequestsInstrumentor().instrument()

# Custom middleware to log request bodies
async def log_request_body(request: Request, call_next):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("http_request") as span:
        # Log method and url
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))

        # Get request body
        body = await request.body()
        
        # Log request body
        try:
            body_str = body.decode()
            # Try to parse as JSON for prettier logging
            body_json = json.loads(body_str)
            span.set_attribute("http.request_body", json.dumps(body_json, indent=2))
        except json.JSONDecodeError:
            # If it's not valid JSON, log as is
            span.set_attribute("http.request_body", body_str)
        except UnicodeDecodeError:
            # If it can't be decoded as a string, log as base64
            span.set_attribute("http.request_body", f"<binary data, base64 encoded> {base64.b64encode(body).decode()}")

        # Proceed with the request
        response = await call_next(request)

        return response

router = fastapi.APIRouter()

@router.get("/items/{id}")
async def item_handler(id: int):
    """A simple API to get an item by ID."""
    async_session_maker = async_sessionmaker(global_storage.engine, expire_on_commit=False)
    async with async_session_maker() as session:
        with tracer.start_as_current_span("database_query"):
            item = (await session.scalars(select(Item).where(Item.id == id))).first()
    return item.to_dict()

@router.get("/similar")
async def similar_handler(id: int, n: int = 5):
    """A similarity API to find items similar to items with given ID."""
    async_session_maker = async_sessionmaker(global_storage.engine, expire_on_commit=False)
    async with async_session_maker() as session:
        with tracer.start_as_current_span("similarity_query"):
            item = (await session.scalars(select(Item).where(Item.id == id))).first()
            closest = await session.execute(
                select(Item, Item.embedding.l2_distance(item.embedding))
                .filter(Item.id != id)
                .order_by(Item.embedding.l2_distance(item.embedding))
                .limit(n)
            )
    return [item.to_dict() | {"distance": round(distance, 2)} for item, distance in closest]

@router.get("/search")
async def search_handler(query: str, top: int = 5, enable_vector_search: bool = True, enable_text_search: bool = True):
    """A search API to find items based on a query."""
    with tracer.start_as_current_span("search_query"):
        searcher = PostgresSearcher(
            global_storage.engine,
            openai_embed_client=global_storage.openai_embed_client,
            embed_deployment=global_storage.openai_embed_deployment,
            embed_model=global_storage.openai_embed_model,
            embed_dimensions=global_storage.openai_embed_dimensions,
        )
        results = await searcher.search_and_embed(
            query, top=top, enable_vector_search=enable_vector_search, enable_text_search=enable_text_search
        )
    return [item.to_dict() for item in results]

@router.post("/chat")
async def chat_handler(chat_request: ChatRequest):
    with tracer.start_as_current_span("chat_query"):
        messages = [message.model_dump() for message in chat_request.messages]
        overrides = chat_request.context.get("overrides", {})
        searcher = PostgresSearcher(
            global_storage.engine,
            openai_embed_client=global_storage.openai_embed_client,
            embed_deployment=global_storage.openai_embed_deployment,
            embed_model=global_storage.openai_embed_model,
            embed_dimensions=global_storage.openai_embed_dimensions,
        )
        if overrides.get("use_advanced_flow"):
            ragchat = AdvancedRAGChat(
                searcher=searcher,
                openai_chat_client=global_storage.openai_chat_client,
                chat_model=global_storage.openai_chat_model,
                chat_deployment=global_storage.openai_chat_deployment,
            )
        else:
            ragchat = SimpleRAGChat(
                searcher=searcher,
                openai_chat_client=global_storage.openai_chat_client,
                chat_model=global_storage.openai_chat_model,
                chat_deployment=global_storage.openai_chat_deployment,
            )
        response = await ragchat.run(messages, overrides=overrides)
    return response

# Add routes to the app
app.include_router(router)

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)