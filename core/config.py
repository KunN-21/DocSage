import os
import subprocess
from dotenv import load_dotenv

load_dotenv()


def _openrouter_api_base() -> str:
    # Ưu tiên tên biến đúng theo ChatOpenRouter docs.
    return os.getenv(
        "OPENROUTER_API_BASE",
        os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )

class Config():
    AZURE_DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTEL_ENDPOINT", "")
    AZURE_DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTEL_KEY", "")

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

    OPENROUTER_API_BASE = _openrouter_api_base()
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
    OPENROUTER_FAST_MODEL = os.getenv("OPENROUTER_FAST_MODEL", "openrouter/auto")

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    OLLAMA_FAST_MODEL = os.getenv("OLLAMA_FAST_MODEL", "qwen3.5:2b-q4_K_M")

    def azure_available(self) -> bool:
        return bool(self.AZURE_DOC_INTEL_ENDPOINT and self.AZURE_DOC_INTEL_KEY)
    
    def openrouter_available(self) -> bool:
        return bool(self.OPENROUTER_API_KEY)
    
    def ollama_url(self) -> str:
        if self.OLLAMA_BASE_URL:
            return self.OLLAMA_BASE_URL
        try:
            windows_host_ip = subprocess.check_output(
                "ip route show | grep -i default | awk '{ print $3}'", 
                shell=True
            ).decode().strip()
            
            url = f"http://{windows_host_ip}:11434"
            print(f"Auto-detected Ollama URL (WSL): {url}")
            return url
        except Exception:
            print("Không thể tìm IP, dùng localhost.")
            return "http://localhost:11434"
    
config = Config()