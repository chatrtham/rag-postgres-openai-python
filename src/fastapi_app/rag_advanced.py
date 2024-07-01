import re
import copy
import logging
import pathlib
from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai_messages_token_helper import get_token_limit
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_random_exponential

from .api_models import ThoughtStep
from .postgres_searcher import PostgresSearcher
from .query_rewriter import (
    build_hybrid_search_function,
    build_specify_package_function,
    extract_search_arguments,
    handle_specify_package_function_call,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedRAGChat:
    def __init__(
        self,
        *,
        searcher: PostgresSearcher,
        openai_chat_client: AsyncOpenAI,
        chat_model: str,
        chat_deployment: str | None,  # Not needed for non-Azure OpenAI
    ):
        self.searcher = searcher
        self.openai_chat_client = openai_chat_client
        self.chat_model = chat_model
        self.chat_deployment = chat_deployment
        self.chat_token_limit = get_token_limit(chat_model, default_to_minimum=True)
        current_dir = pathlib.Path(__file__).parent
        self.specify_package_prompt_template = open(current_dir / "prompts/specify_package.txt").read()
        self.query_prompt_template = open(current_dir / "prompts/query.txt").read()
        self.answer_prompt_template = open(current_dir / "prompts/answer.txt").read()

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6), before_sleep=before_sleep_log(logger, logging.WARNING))
    async def openai_chat_completion(self, *args, **kwargs) -> ChatCompletion:
        return await self.openai_chat_client.chat.completions.create(*args, **kwargs)

    async def hybrid_search(self, messages, top, vector_search, text_search):
        # Generate an optimized keyword search query based on the chat history and the last question
        query_messages = copy.deepcopy(messages)
        query_messages.insert(0, {"role": "system", "content": self.query_prompt_template})
        query_response_token_limit = 500

        query_chat_completion: ChatCompletion = await self.openai_chat_completion(
            messages=query_messages,
            model=self.chat_deployment if self.chat_deployment else self.chat_model,
            temperature=0.0,
            max_tokens=query_response_token_limit,
            n=1,
            tools=build_hybrid_search_function(),
            tool_choice="auto",
        )

        query_text, filters = extract_search_arguments(query_chat_completion)

        # Retrieve relevant items from the database with the GPT optimized query
        results = await self.searcher.search_and_embed(
            query_text,
            top=top,
            enable_vector_search=vector_search,
            enable_text_search=text_search,
            filters=filters,
        )

        sources_content = [f"[{(item.url)}]:{item.to_str_for_broad_rag()}\n\n" for item in results]

        thought_steps = [
            ThoughtStep(
                title="Prompt to generate search arguments",
                description=[str(message) for message in query_messages],
                props={"model": self.chat_model, "deployment": self.chat_deployment} if self.chat_deployment else {"model": self.chat_model}
            ),
            ThoughtStep(
                title="Generated search arguments",
                description=query_text,
                props={"filters": filters}
            ),
            ThoughtStep(
                title="Hybrid Search results",
                description=[result.to_dict() for result in results],
                props={
                    "top": top,
                    "vector_search": vector_search,
                    "text_search": text_search
                }
            )
        ]
        return sources_content, thought_steps

    async def get_product_cards_details(self, urls: list[str]) -> list[dict]:
        return await self.searcher.get_product_cards_info(urls)

    async def run(
        self, messages: list[dict], overrides: dict[str, Any] = {}
    ) -> dict[str, Any] | AsyncGenerator[dict[str, Any], None]:
        # Normalize the message format
        for message in messages:
            if isinstance(message['content'], str):
                message['content'] = [{'type': 'text', 'text': message['content']}]

        # Determine the search mode and the number of results to return
        text_search = overrides.get("retrieval_mode") in ["text", "hybrid", None]
        vector_search = overrides.get("retrieval_mode") in ["vectors", "hybrid", None]
        top = overrides.get("top", 3)

        # Generate a prompt to specify the package if the user is referring to a specific package
        specify_package_messages = copy.deepcopy(messages)
        specify_package_messages.insert(0, {"role": "system", "content": self.specify_package_prompt_template})
        specify_package_token_limit = 300

        specify_package_chat_completion: ChatCompletion = await self.openai_chat_completion(
            messages=specify_package_messages,
            model=self.chat_deployment if self.chat_deployment else self.chat_model,
            temperature=0.0,
            max_tokens=specify_package_token_limit,
            n=1,
            tools=build_specify_package_function()
        )

        specify_package_filters = handle_specify_package_function_call(specify_package_chat_completion)

        if specify_package_filters:  # Simple SQL search
            results = await self.searcher.simple_sql_search(filters=specify_package_filters)

            if results:
                sources_content = [f"[{(item.url)}]:{item.to_str_for_narrow_rag()}\n\n" for item in results]

                thought_steps = [
                    ThoughtStep(
                        title="Prompt to specify package",
                        description=[str(message) for message in specify_package_messages],
                        props={"model": self.chat_model, "deployment": self.chat_deployment} if self.chat_deployment else {"model": self.chat_model}
                    ),
                    ThoughtStep(
                        title="Specified package filters",
                        description=specify_package_filters,
                        props={}
                    ),
                    ThoughtStep(
                        title="SQL search results",
                        description=[result.to_dict() for result in results],
                        props={}
                    )
                ]
            else:
                # No results found with SQL search, fall back to the hybrid search
                sources_content, thought_steps = await self.hybrid_search(messages, top, vector_search, text_search)
        else:  # Hybrid search
            sources_content, thought_steps = await self.hybrid_search(messages, top, vector_search, text_search)

        content = "\n".join(sources_content)

        # Build messages for the final chat completion
        messages.insert(0, {"role": "system", "content": self.answer_prompt_template})
        messages[-1]["content"].append({"type": "text", "text": "\n\nSources:\n" + content})
        response_token_limit = 4096

        chat_completion_response = await self.openai_chat_completion(
            model=self.chat_deployment if self.chat_deployment else self.chat_model,
            messages=messages,
            temperature=overrides.get("temperature", 0.3),
            max_tokens=response_token_limit,
            n=1,
            stream=False,
        )
        chat_resp = chat_completion_response.model_dump()

        chat_resp_content = chat_resp["choices"][0]["message"]["content"]
        package_urls = re.findall(r'https:\/\/hdmall\.co\.th\/[\w.,@?^=%&:\/~+#-]+', chat_resp_content)
        
        if package_urls:
            product_cards_details = await self.get_product_cards_details(package_urls)
        else:
            product_cards_details = []

        chat_resp["choices"][0]["context"] = {
            "data_points": {"text": sources_content},
            "thoughts": thought_steps + [
                ThoughtStep(
                    title="Prompt to generate answer",
                    description=[str(message) for message in messages],
                    props=(
                        {"model": self.chat_model, "deployment": self.chat_deployment}
                        if self.chat_deployment
                        else {"model": self.chat_model}
                    ),
                ),
                ThoughtStep(
                    title="Product Cards Details",
                    description=product_cards_details,
                    props={}
                )
            ]
        }
        return chat_resp