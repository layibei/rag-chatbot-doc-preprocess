# User Query Processing Requirements

## Overview

This document outlines the requirements for processing user queries in the chatbot.

## User Query Processing

### 1. Input

#### 1.1. User Query Input format and query endpoint

- The user query is a natural language query from the user, and passed in JSON format with key of user_input
- The API endpoint is POST:/chat/query, and the request body is a JSON object with the following fields:
    - user_input: A string containing the user's query.
- The query endpoint has following mandatory headers:
    - Authorization: A string containing the user's access token.
    - Content-Type: application/json
    - X-User-Id: A string containing the user's id.
    - X-Session-Id: A string containing the session id.

#### 1.2. User Query Output format

- The response is a JSON object with the following fields:
    - data: A field containing the response to the user's query, it might be a string or a JSON object or a list of JSON
      objects.
    - user_input: A string containing the user's query.
- The response has following mandatory headers:
    - Content-Type: application/json
    - X-User-Id: A string containing the user's id.
    - X-Session-Id: A string containing the session id.
    - X-Request-Id: A string containing the request id.

### 2. User Query Processing

#### 2.1. User Intent Recognition

- The user intent recognition is the process of identifying the user's intent from the user's query.
    - If the user's query is about greetings, then response a greeting message directly without any further processing.
    - For others, call tool - generic_query_handler tool to process the user's query.

#### 2.2. Generic query handler

- The generic query handler is the tool that processes the user's query and returns the response to the user.
- The generic query handler is a tool that can be called by the user query processing.
- The handler is implemented in the form of a function in the LangChain library.
- The handler consist of two major parts:
    - conversation history attachment - which is used to attach the conversation history to the user query.
    - query process workflow - which is majorly leveraged the Langgraph framework.
    - query history storage - which is used to store the user's query and the response to the user.
- The user query processing is the process of processing the user's query and returning the response to the user.

##### 2.2.1. Load Conversation history

- Load the conversation history from the database to enrich the user query prompt by the user_id + session_id.
- Variable name: conversation_history

#### 2.2.2. Query Process Workflow

- Use the user query to retrieve relevant documents from the vector store.
- If no relevant documents are found + query rewritten flag is false, call the tool/node - rewrite query node to rewrite
  the query, and set the query_rewritten flag to true.
- If no relevant documents are found + query rewritten flag is true, then call web_search node to search the web for the
  answer when web_search_enabled in config app.yaml, and add the web search
  result to the relevant documents array.
- If no relevant documents are found + query rewritten flag is true, and the web search is disabled, return fallback
  message,e.g. "I'm sorry, I don't have the information you are looking for."
- If relevant documents are found, rerank the relevant documents using the rerank model and pick up the top 5 results.
- Generate the output using the relevant documents.
- Return the output to the user.

#### 2.2.3. Query History Storage

- Store the user query and the response to the user into the postgres database.
- The conversation history should have following key fields:
    - User_id
    - Session_id
    - Request_id
    - User_input
    - Response
    - Liked(boolean) -- this can be updated by the user later via the chatbot interface called on UI when the user
      clicks like/unlike icon for a response

### 3. Error Handling

- The error handling is the process of handling errors that occur during the user query processing.
- The error response should be well-structured and contain the following fields:
    - status: A string containing the status of the response.
    - error_message: A string containing the error message.
    - user_input: A string containing the user's query.
    - error_code: An integer or short string containing the error code.

### 4. Performance Monitoring & Auditing & Tracing

- The performance monitoring is the process of monitoring the performance of the user query processing.
- The user_id,session_id,request_id should be used for tracing, and those fields should be included in the all the
  system logs to trace the whole process or call chain.
- All the inbound and outbound requests to/from LLM should be saved into database table
- The token usage and time cost should be recorded and saved into database table as well