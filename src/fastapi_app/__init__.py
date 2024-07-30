import contextlib
import logging
import os

import azure.identity.aio
import fastapi
from azure.monitor.opentelemetry import configure_azure_monitor
from dotenv import load_dotenv
from environs import Env
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

from .globals import global_storage
from .openai_clients import create_openai_chat_client
from .postgres_engine import create_postgres_engine_from_env

logger = logging.getLogger("ragapp")


@contextlib.asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    load_dotenv(override=True)

    azure_credential = None
    try:
        if client_id := os.getenv("APP_IDENTITY_ID"):
            # Authenticate using a user-assigned managed identity on Azure
            # See web.bicep for value of APP_IDENTITY_ID
            logger.info(
                "Using managed identity for client ID %s",
                client_id,
            )
            azure_credential = azure.identity.aio.ManagedIdentityCredential(client_id=client_id)
        else:
            azure_credential = azure.identity.aio.DefaultAzureCredential()
    except Exception as e:
        logger.warning("Failed to authenticate to Azure: %s", e)

    engine = await create_postgres_engine_from_env(azure_credential)
    global_storage.engine = engine

    openai_chat_client, openai_chat_model = await create_openai_chat_client(azure_credential)
    global_storage.openai_chat_client = openai_chat_client
    global_storage.openai_chat_model = openai_chat_model

    if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)
    yield

    await engine.dispose()


def create_app():
    env = Env()

    if not os.getenv("RUNNING_IN_PRODUCTION"):
        env.read_env(".env")
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Turn off particularly noisy INFO level logs from Azure Core SDK:
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

    if os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"):
        logger.info("Configuring Azure Monitor")
        configure_azure_monitor(logger_name="ragapp")

    app = fastapi.FastAPI(docs_url="/docs", lifespan=lifespan)

    from . import api_routes  # noqa
    from . import frontend_routes  # noqa

    app.include_router(api_routes.router)
    app.mount("/", frontend_routes.router)

    return app
