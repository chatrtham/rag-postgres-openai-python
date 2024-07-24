import fastapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from fastapi_app.api_models import ChatRequest
from fastapi_app.globals import global_storage
from fastapi_app.postgres_models import Package
from fastapi_app.postgres_searcher import PostgresSearcher
from fastapi_app.rag_advanced import AdvancedRAGChat
from fastapi_app.utils import update_urls_with_utm

router = fastapi.APIRouter()


@router.get("/packages/{url}")
async def package_handler(url: str):
    """A simple API to get an package by URL."""
    async_session_maker = async_sessionmaker(global_storage.engine, expire_on_commit=False)
    async with async_session_maker() as session:
        package = (await session.scalars(select(Package).where(Package.url == url))).first()
        return package.to_dict()


@router.post("/chat")
async def chat_handler(chat_request: ChatRequest):
    """API to chat with the RAG model."""
    messages = [message.model_dump() for message in chat_request.messages]

    searcher = PostgresSearcher(global_storage.engine)

    ragchat = AdvancedRAGChat(
        searcher=searcher,
        openai_chat_client=global_storage.openai_chat_client,
        chat_model=global_storage.openai_chat_model,
        chat_deployment=global_storage.openai_chat_deployment,
    )

    chat_resp = await ragchat.run(messages)
    chat_resp_content = chat_resp["choices"][0]["message"]["content"]

    # Update URLs with UTM parameters
    url_pattern = r"https:\/\/hdmall\.co\.th\/[\w.,@?^=%&:\/~+#-]+"
    chat_resp_content = update_urls_with_utm(chat_resp_content, url_pattern)

    # Update the chat response with the modified content
    chat_resp["choices"][0]["message"]["content"] = chat_resp_content
    return chat_resp
