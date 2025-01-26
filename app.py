import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from api.embedding_routes import router as embedding_router
from config.common_settings import CommonConfig
from preprocess.doc_embedding_job import DocEmbeddingJob
from utils.logging_util import logger, set_context, clear_context

# Global config instance
base_config = CommonConfig()


class LoggingContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Get user_id from headers (required)
        user_id = request.headers.get('X-User-Id', 'unknown')

        # Set context using keyword arguments
        set_context(
            user_id=user_id,
            request_path=request.url.path,
            request_method=request.method,
            start_time=start_time
        )

        try:
            response = await call_next(request)
            # Add timing information
            request_time = time.time() - start_time
            set_context(request_time_ms=int(request_time * 1000))
            logger.info(f"Request completed in {request_time:.2f}s")
            return response
        finally:
            clear_context()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application"""
    embedding_job = None
    try:
        # Startup
        logger.info("Initializing application...")

        # Initialize config and setup proxy first
        proxy_result = await base_config.asetup_proxy()
        logger.info(f"Proxy setup is {'enabled' if proxy_result else 'disabled'}")

        # Initialize other components (make this non-blocking)
        embedding_job = DocEmbeddingJob()
        init_result = await embedding_job.initialize()
        logger.info(f"Document embedding job initialization {'successful' if init_result else 'failed'}")

        logger.info("Application startup completed")

        # Important: yield here to let FastAPI take control
        yield

    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    finally:
        # Shutdown
        if embedding_job and embedding_job.scheduler:
            embedding_job.scheduler.shutdown()
        logger.info("Shutting down application...")


app = FastAPI(lifespan=lifespan)
app.add_middleware(LoggingContextMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, adjust as needed
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],  # Allow all headers, adjust as needed
    expose_headers=["X-Session-Id", "X-Request-Id", "X-User-Id"],
)

app.include_router(embedding_router, prefix="/embedding")


if __name__ == "__main__":
    # set_debug(True)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
