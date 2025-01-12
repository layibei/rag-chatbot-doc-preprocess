# Document Embedding System Requirements

## Overview
This document outlines the requirements for the document embedding system of the RAG-CHAT application. The system processes various document types (PDF, CSV, JSON, etc.) and embeds them into a vector database for use in the RAG (Retrieval-Augmented Generation) chatbot.

## Core Components

### 1. Document Processing API Endpoints

#### 1.1 Upload Document (POST /embedding/docs/upload)
- Accepts file upload with source type specification
- Validates file extension against source type
- Sanitizes file names for safe storage
- Stores files in staging directory
- Creates index log with PENDING status
- Returns document metadata and processing status

#### 1.2 Add Document by Path (POST /embedding/docs)
- Accepts document path and source type
- Validates document existence and accessibility
- Creates index log with PENDING status
- Returns document metadata and processing status

#### 1.3 Document Status (GET /embedding/docs/{log_id})
- Returns current processing status of a document
- Includes error messages if processing failed

#### 1.4 List Documents (GET /embedding/docs)
- Supports pagination (page, page_size)
- Supports search across multiple fields:
  - source
  - created_by
  - modified_by
- Returns list of documents with their statuses

### 2. Document Processing Workflow

#### 2.1 Document Validation
- Checksum calculation for content change detection
- Duplicate detection based on:
  - Source path + source type combination
  - Content checksum
- Existing document handling:
  - No changes: Return "already processed" message
  - Changes detected: Remove old embeddings and reprocess

#### 2.2 Processing Pipeline
- Status tracking through states:
  - PENDING: Initial state
  - IN_PROGRESS: During processing
  - COMPLETED: Successfully processed
  - FAILED: Processing error occurred
- Document loading based on type (PDF, CSV, JSON, etc.)
- Metadata attachment to chunks
- Vector store integration
- Automatic file archiving after successful processing

### 3. Scheduled Jobs

#### 3.1 Pending Documents Processor
- Runs every 5 minutes
- Uses distributed locking for multi-instance safety
- Processes one pending document at a time
- Moves processed documents to archive directory
- Updates source path in index log after archiving

#### 3.2 INPUT Directory Scanner
- Runs every 5 minutes
- Scans configured input directory for new documents
- Creates index logs for unprocessed documents
- Uses system user for automated processing
- Prevents duplicate processing via checksum checks

### 4. Infrastructure

#### 4.1 Database Schema
- index_logs table:
  - Tracks document processing status
  - Stores document metadata
  - Maintains audit trail
  - Includes error tracking
- distributed_locks table:
  - Enables distributed processing
  - Prevents concurrent processing
  - Tracks lock ownership

#### 4.2 File Storage
- Staging directory for uploaded files
- Archive directory for processed files
- Configurable paths via CommonConfig

#### 4.3 API Requirements
Mandatory headers for all API calls:
- user-id: User identifier
- session-id: Session tracking
- request-id: Request tracing

## Configuration
All configurable values managed through:
- app.yaml
- CommonConfig class
- Environment variables

## Error Handling
- Detailed error logging
- Error message storage in index_logs
- Automatic cleanup of failed uploads
- Transaction management for database operations

## Audit Trail
Automatic tracking of:
- Creation timestamp and user
- Modification timestamp and user
- Processing status changes
- Error conditions
```




