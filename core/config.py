import os
import subprocess
from dotenv import load_dotenv

load_dotenv()

class Config():
    AZURE_DOC_INTEL_ENDPOINT = os.getenv("AZURE_DOC_INTEL_ENDPOINT", "")
    AZURE_DOC_INTEL_KEY = os.getenv("AZURE_DOC_INTEL_KEY", "")

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_FAST_MODEL = os.getenv("OPENAI_FAST_MODEL", "gpt-4o-mini")

    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    OLLAMA_FAST_MODEL = os.getenv("OLLAMA_FAST_MODEL", "qwen3.5:2b-q4_K_M")

    def azure_available(self) -> bool:
        return bool(self.AZURE_DOC_INTEL_ENDPOINT and self.AZURE_DOC_INTEL_KEY)
    
    def openai_available(self) -> bool:
        return bool(self.OPENAI_API_KEY)
    
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