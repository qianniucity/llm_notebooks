import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME")

NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
NEWS_BASE_URL = os.environ.get("NEWS_BASE_URL")

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
