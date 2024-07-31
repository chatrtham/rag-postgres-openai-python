# Backend

## Overview
This directory contains code for an advanced chat-based application that uses a combination of Google search and OpenAI's chat completion to generate responses to user queries. The application is built using Python and includes the following main components:
- `AdvancedRAGChat`: This class provides the main functionality for the chat-based application. It includes methods for generating chat completions using Azure OpenAI's chat completion API, performing Google searches, and running the chat-based application.
- `PostgresSearcher`: This class provides functionality for searching a Postgres database.
- `llm_tools.py`: This module contains code for building and using custom functions within the chat-based application, such as Google search and specifying a package.
- `postgres_models.py`: This module contains data models for the chat-based application's database, including thoughts and metadata.
- `api_routes.py`: This module contains the FastAPI routes for the application, including the `/chat` route.
- `google_search.py`: This module contains a function `google_search_function` for performing a Google search given a search query.
- `seed_hd_data.py`: This script is for inserting **HD**'s data into the database.
- `get_token.py`: This script is for generating an Azure OAuth token, used as a temporary password for the PostgreSQL Database.


## Usage
To use this codebase locally, you can these steps:
1. Run `python3 -m uvicorn fastapi_app:create_app --factory --reload`.
1. Send a `POST` request to the `/chat` route.