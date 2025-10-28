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








