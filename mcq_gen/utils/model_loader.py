import os
import sys
import json
from dotenv import load_dotenv
from mcq_gen.utils.config_loader import load_config

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from mcq_gen.logger import GLOBAL_LOGGER as log
from mcq_gen.exception.custom_exception import ProjectException


load_dotenv() 

class ApiKeyManager:
    REQUIRED_KEYS = ["MISTRAL_API_KEY"] 
    def __init__(self):
        self.api_keys = {}
        raw = os.getenv("apikeys")
        # if .env has -> a dict of api keys
        if raw:
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError("API_KEYS in not a valid JSON object.")
                self.api_keys = parsed
                log.info("Loaded API_KEYS from ECS secret.")
            except Exception as e:
                log.warning("Failed to parse API_KEYS as JSON", error=str(e))

        # if .env has individual api keys
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from individual env var")

        # check for missing keys
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        if missing:
            log.error("Missing required API keys", missing_keys=missing)
            raise ProjectException("Missing API keys", sys)
        
        log.info("API keys loaded", keys={k: v[:6] + "..." for k, v in self.api_keys.items()})


class ModelLoader:
    
    def __init__(self):
        """"
        If the environment is not production, 
        load the .env file and run in local/development mode; 
        otherwise, assume production environment variables are already set.
        """
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("RUNNING IN LOCAL MODE: .env loaded")
        else:
            log.info("RUNNING IN PRODUCTION MODE!!!")
        
        self.api_key_manager = ApiKeyManager()
        self.config = load_config()
        log.info("YAML CONFIG LOADED", config_keys=list(self.config.keys()))

    def load_llm(self):
        """
        Load and return the configured LLM model.
        """

        # dynamically pick which LLM provider to use based on your environment
        llm_block = self.config["llm"]
        provider_key = os.getenv("LLM_PROVIDER", "mistral") # If LLM_PROVIDER does not exist, it returns "mistral" (the default value).

        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider=provider_key)
            raise ValueError(f"LLM provide '{provider_key}' not found in config")
        
        llm_config = llm_block[provider_key]
        provider = llm_config.get("procider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", 0.2)

        log.info("Loading LLM", provider=provider, model=model_name)

        if provider_key == "mistral":
            return ChatMistralAI(
                model = model_name,
                temperature=temperature,
            )
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")



    def load_embeddings(self):
        """
        load and return embedding model from Mistral AI
        """
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)
            return MistralAIEmbeddings(
                model=model_name
            )
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise ProjectException("Failed to load embedding model", sys)











