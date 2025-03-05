import os
from functools import lru_cache
from typing import Any, Dict, Union

import dotenv
import spacy
import yaml

from langchain_community.llms.sparkllm import SparkLLM
from langchain_core.vectorstores import VectorStore
from neo4j import GraphDatabase
from spacy import Language
from spacy.vectors import Path

from config.database.database_manager import DatabaseManager
from utils.logger_init import logger

# Get the absolute path of the current file
CURRENT_FILE_PATH = os.path.abspath(__file__)
# Get the directory containing the current file
BASE_DIR = os.path.dirname(CURRENT_FILE_PATH)


class CommonConfig:
    def __init__(self, config_path: str = None):
        self.logger = logger
        dotenv.load_dotenv(dotenv_path=BASE_DIR + '/../.env')

        if config_path:
            path = BASE_DIR + config_path
            if not os.path.exists(path):
                raise ConfigError("Config file not found")
            self.config = self.load_yaml_file(path)
        else:
            default_path = BASE_DIR + "/app.yaml"
            if not os.path.exists(default_path):
                raise ConfigError("Config file not found")
            self.config = self.load_yaml_file(default_path)

        if not self.config:
            raise ConfigError("Invalid configuration")

    def check_config(self, config, path, message):
        """Helper function to check configuration and raise an error if necessary."""
        current = config
        for key in path:
            if key not in current:
                raise ConfigError(message)
            current = current[key]

    def _get_llm_model(self):
        self.check_config(config, ["app", "models", "llm", "type"],
                          "LLM model is not found")
        self.logger.debug(f"LLM model: {self.config['app']['models']['llm']}")

        if self.config["app"]["models"]["llm"].get("type") == "sparkllm":
            return SparkLLM()

        if self.config["app"]["models"]["llm"].get("type") == "ollama" and self.config["app"]["models"]["llm"].get(
                "model") is not None:
            from langchain_ollama import OllamaLLM
            return OllamaLLM(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

        if self.config["app"]["models"]["llm"].get("type") == "gemini" and self.config["app"]["models"]["llm"].get(
                "model") is not None:
            from langchain_google_genai import GoogleGenerativeAI
            return GoogleGenerativeAI(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

        if self.config["app"]["models"]["llm"].get("type") == "anthropic" and self.config["app"]["models"][
            "llm"].get("model") is not None:
            from langchain_anthropic import AnthropicLLM
            return AnthropicLLM(model=self.config["app"]["models"]["llm"].get("model"), temperature=0.85)

        raise RuntimeError("Not found the model type");

    def _get_embedding_model(self):
        self.check_config(self.config, ["app", "models", "embedding", "type"],
                          "Embedding model is not found")
        self.logger.debug(f"Embedding model: {self.config['app']['models']['embedding']}")

        if self.config["app"]["models"]["embedding"].get("type") == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=self.config["app"]["models"]["embedding"].get("model"))

        raise RuntimeError("Not found the embedding model type");

    def _get_chatllm_model(self):
        """Get chat LLM model with proxy configuration"""
        try:
            self.check_config(self.config, ["app", "models", "chatllm", "type"],
                              "ChatLLM model is not found")
            model_config = self.config["app"]["models"]["chatllm"]
            model_type = model_config.get("type")

            if model_type == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model_config.get("model"),
                    temperature=0.85,
                )
            elif model_type == "anthropic":
                from langchain_anthropic import ChatAnthropic
                return ChatAnthropic(
                    model=model_config.get("model"),
                    temperature=0.85,
                )
            # Add other model types as needed...

            raise RuntimeError(f"Unsupported chatllm model type: {model_type}")

        except Exception as e:
            self.logger.error(f"Error initializing chat LLM: {str(e)}")
            raise

    def get_model(self, type):
        """Get model by type"""
        self.logger.debug(f"Get model by type: {type}")
        if not isinstance(type, str):
            raise TypeError("Model type must be a string")

        if type == "embedding":
            return self._get_embedding_model()
        elif type == "llm":
            return self._get_llm_model()

        elif type == "chatllm":
            return self._get_chatllm_model()

        else:
            raise ValueError("Invalid model type")

    def get_embedding_config(self, key: str = None, default_value: Any = None) -> Any:
        self.logger.debug(f"Embedding config: {self.config['app']['embedding']}")
        self.check_config(self.config, ["app", "embedding"], "app embedding is not found.")
        self.check_config(self.config, ["app", "embedding", "input_path"], "input path in app embedding is not found.")
        embedding_config = {
            "input_path": self.config["app"]["embedding"].get("input_path"),
            "staging_path": self.config["app"]["embedding"].get("staging_path"),
            "archive_path": self.config["app"]["embedding"].get("archive_path"),
            "trunk_size": self.config["app"]["embedding"].get("trunk_size", 1024),
            "overlap": self.config["app"]["embedding"].get("overlap", 100),
            "confluence": {
                "url": os.environ.get("CONFLUENCE_URL"),
                "username": os.environ.get("CONFLUENCE_USER_NAME"),
                "api_key": os.environ.get("CONFLUENCE_API_KEY"),
                "token": os.environ.get("CONFLUENCE_TOKEN"),
            },
            "vector_store": {
                "enabled": self.config["app"]['embedding']["vector_store"].get("enabled", False),
            },
            "graph_store": {
                "enabled": self.config["app"]['embedding']["graph_store"].get("enabled", False),
            },
        }

        if key is None:
            return embedding_config

        # Handle nested key access
        keys = key.split(".")
        value = embedding_config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default_value

    @lru_cache(maxsize=1)
    def get_vector_store(self) -> VectorStore:
        """Get vector store"""
        self.logger.debug("Get vector store.")
        self.check_config(self.config, ["app", "embedding", "vector_store"], "app vector_store is not found.")
        vector_store_type = self.config["app"]["embedding"]["vector_store"].get("type")

        if vector_store_type == "qdrant":
            from langchain_qdrant import QdrantVectorStore
            return QdrantVectorStore.from_documents(
                documents=[],
                embedding=self.get_model("embedding"),
                collection_name="rag_docs",
                url=os.environ["QDRANT_URL"],
                api_key=os.environ["QDRANT_API_KEY"],
            )

        elif vector_store_type == "redis":
            from langchain_redis import RedisConfig, RedisVectorStore
            config = RedisConfig(
                index_name="rag_docs",
                redis_url=os.environ["REDIS_URL"],
                distance_metric="COSINE",  # Options: COSINE, L2, IP
            )
            return RedisVectorStore(self.get_model("embedding"), config=config)

        elif vector_store_type == "pgvector":
            from langchain_postgres import PGVector
            return PGVector(
                embeddings=self.get_model("embedding"),
                collection_name="rag_docs",
                connection=os.environ["POSTGRES_URI"],
                use_jsonb=True,
            )
        else:
            raise RuntimeError("Not found the vector store type")
    def get_nlp_spacy(self) -> Language:
        """Get NLP model"""
        # Load spaCy model from local path
        model_path = Path(os.path.join(BASE_DIR, "../models/spacy/en_core_web_md"))
        if not model_path.exists():
            raise RuntimeError(
                "spaCy model not found. Please run scripts/download_spacy_model.py first"
            )
        return spacy.load(str(model_path))
    def setup_proxy(self):
        """Setup proxy configuration"""
        try:
            if self.config["app"]["proxy"].get("enabled", False):
                # Set proxy environment variables
                proxy_config = {
                    "http_proxy": self.config["app"]["proxy"].get("http_proxy"),
                    "https_proxy": self.config["app"]["proxy"].get("https_proxy"),
                    "no_proxy": self.config["app"]["proxy"].get("no_proxy")
                }

                # Set environment variables
                for key, value in proxy_config.items():
                    if value:
                        os.environ[key] = value
                        os.environ[key.upper()] = value  # Some libraries use uppercase

                self.logger.info("Proxy configuration set successfully")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to setup proxy: {str(e)}")
            raise ConfigError(f"Proxy setup failed: {str(e)}")

    async def asetup_proxy(self):
        """Setup proxy configuration"""
        try:
            if self.config["app"]["proxy"].get("enabled", False):
                # Set proxy environment variables
                proxy_config = {
                    "http_proxy": self.config["app"]["proxy"].get("http_proxy"),
                    "https_proxy": self.config["app"]["proxy"].get("https_proxy"),
                    "no_proxy": self.config["app"]["proxy"].get("no_proxy")
                }

                # Set environment variables
                for key, value in proxy_config.items():
                    if value:
                        os.environ[key] = value
                        os.environ[key.upper()] = value  # Some libraries use uppercase

                self.logger.info("Proxy configuration set successfully")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to setup proxy: {str(e)}")
            raise ConfigError(f"Proxy setup failed: {str(e)}")

    @lru_cache(maxsize=1)
    def get_db_manager(self) -> DatabaseManager:
        return DatabaseManager(os.environ["POSTGRES_URI"])

    @staticmethod
    def load_yaml_file(file_path: str):
        try:
            with open(file_path, 'r') as file:
                # Use the safe loader to avoid security risks
                data = yaml.safe_load(file)
                return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return None
        except yaml.YAMLError as exc:
            logger.error(f"Error in YAML file: {exc}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return None

    @lru_cache(maxsize=100)
    def get_logging_config(self, package_name: str = None) -> Union[Dict[str, str], str]:
        """
        Get logging configuration for packages with hierarchical path support.
        Args:
            package_name: Optional package name to get specific log level
        Returns:
            Dict of package log levels or specific level string
        """
        self.logger.debug(f"Getting logging config for package: {package_name}")

        try:
            # Get logging config with default fallback
            logging_levels = self.config.get("app", {}).get("logging.level", {})
            root_level = logging_levels.get("root", "INFO")

            if package_name:
                # Find the most specific matching package path
                matching_level = root_level
                matching_length = 0

                for pkg_path, level in logging_levels.items():
                    if pkg_path != "root" and package_name.startswith(pkg_path):
                        path_length = len(pkg_path.split('.'))
                        if path_length > matching_length:
                            matching_level = level
                            matching_length = path_length

                return matching_level

            return logging_levels
        except Exception as e:
            self.logger.error(f"Error getting logging config: {str(e)}")
            return "INFO" if package_name else {"root": "INFO"}

    @lru_cache(maxsize=1)
    def get_graph_store(self) -> GraphDatabase:
        """Get Neo4j graph store"""
        try:
            if not self.config["app"]["embedding"]["graph_store"].get("enabled", False):
                self.logger.info("Graph store is disabled")
                return None

            uri = os.environ.get("NEO4J_URI")
            username = os.environ.get("NEO4J_USERNAME")
            password = os.environ.get("NEO4J_PASSWORD")

            if not all([uri, username, password]):
                self.logger.error("Missing Neo4j credentials in environment variables")
                return None

            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Test the connection
            try:
                driver.verify_connectivity()
                self.logger.info("Successfully connected to Neo4j")
                return driver
            except Exception as e:
                self.logger.error(f"Failed to connect to Neo4j: {str(e)}")
                if driver:
                    driver.close()
                return None

        except Exception as e:
            self.logger.error(f"Failed to initialize graph store: {str(e)}")
            return None


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


if __name__ == "__main__":
    config = CommonConfig()
    # config.setup_proxy()
    # llm = config.get_model("chatllm")
    # logger.info(llm.invoke("What is the capital of France?"))

    print(config.get_embedding_config("graph_store.enabled"))

    print(config.get_logging_config("utils.lock"))
