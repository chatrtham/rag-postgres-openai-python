import json

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionToolParam,
)


def build_hybrid_search_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search PostgreSQL database for relevant products based on user query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "Query string to use for full text search, e.g. 'ตรวจสุขภาพ'",
                        },
                        "price_filter": {
                            "type": "object",
                            "description": "Filter search results based on price in Thai Baht of the package",
                            "properties": {
                                "comparison_operator": {
                                    "type": "string",
                                    "description": "Operator to compare the column value, either '>', '<', '>=', '<=', '='",
                                },
                                "value": {
                                    "type": "number",
                                    "description": "Value to compare against, e.g. 30",
                                },
                            },
                        },
                    },
                    "required": ["search_query"],
                },
            },
        }
    ]

def extract_search_arguments(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    search_query = None
    filters = []
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type != "function":
                continue
            function = tool.function
            if function.name == "search_database":
                arg = json.loads(function.arguments)
                search_query = arg.get("search_query")
                if "price_filter" in arg and arg["price_filter"]:
                    price_filter = arg["price_filter"]
                    filters.append(
                        {
                            "column": "price",
                            "comparison_operator": price_filter["comparison_operator"],
                            "value": price_filter["value"],
                        }
                    )
    elif query_text := response_message.content:
        search_query = query_text.strip()
    return search_query, filters


def build_specify_package_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "specify_package",
                "description": """
                Specify the exact URL or package name from past messages if they are relevant to the most recent user's message.
                This tool is intended to find specific packages previously mentioned and should not be used for general inquiries or price-based requests.
                """,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": """
                            The exact URL of the package from past messages,
                            e.g. 'https://hdmall.co.th/dental-clinics/xray-for-orthodontics-1-csdc'
                            """
                        },
                        "package_name": {
                            "type": "string",
                            "description": """
                            The exact package name from past messages,
                            always contains the package name and the hospital name,
                            e.g. 'เอกซเรย์สำหรับการจัดฟัน ที่ CSDC'
                            """
                        }
                    },
                    "required": [],
                },
            },
        }
    ]

def handle_specify_package_function_call(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    filters = []
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "specify_package":
                args = json.loads(tool.function.arguments)
                url = args.get("url")
                package_name = args.get("package_name")
                if url:
                    filters.append(
                        {
                            "column": "url",
                            "comparison_operator": "ILIKE",
                            "value": f"%{url}%",
                        }
                    )
                if package_name:
                    filters.append(
                        {
                            "column": "package_name",
                            "comparison_operator": "ILIKE",
                            "value": f"%{package_name}%",
                        }
                    )
    return filters

def build_handover_to_cx_function() -> list[ChatCompletionToolParam]:
    return [
        {
            "type": "function",
            "function": {
                "name": "handover_to_cx",
                "description": """
                This function is used to seamlessly transfer the current conversation to a live
                customer support agent when the user's message indicates a strong intent to buy a
                particular package or service, or mentions booking, cancellation, scheduling,
                or rescheduling, or other post-purchase intents. Upon invocation,
                this function will notify the customer support team to take over the conversation,
                ensuring that the user's needs are promptly and accurately addressed by a human representative.
                Don't call this function if the user wants to talk to a pharmacist
                """,
                "parameters": {},
            },
        }
    ]

def is_handover_to_cx(chat_completion: ChatCompletion):
    response_message = chat_completion.choices[0].message
    if response_message.tool_calls:
        for tool in response_message.tool_calls:
            if tool.type == "function" and tool.function.name == "handover_to_cx":
                return True
    return False