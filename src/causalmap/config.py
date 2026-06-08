from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELLED_DIR = DATA_DIR / "labelled"
DEMOGRAPHICS_DIR = DATA_DIR / "demographics"

TRANSCRIPT_PATH = PROJECT_ROOT / "transcription.xlsx"
EXPORT_DIR = PROJECT_ROOT / "275e2244-1edb-497f-b201-8e02ed99991d"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "deepseek")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

UTTERANCE_GAP_THRESHOLD_S = 1.5
WINDOW_SIZE = 3
WINDOW_OVERLAP = 1
