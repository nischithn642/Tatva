"""
Free Local LLM Engine & Cloud API Integration Provider.

Queries local Ollama instance (http://localhost:11434) for free offline execution,
live NVIDIA API models (https://integrate.api.nvidia.com/v1/models),
and fallback to Anthropic Claude API.
"""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


def get_local_ollama_models(endpoint: str = "http://localhost:11434") -> List[str]:
    """
    Query local Ollama server tags endpoint to fetch available models.
    Returns list of model names (e.g., ['qwen2.5-coder:32b', 'qwen2.5-coder:7b', 'deepseek-coder-v2']).
    """
    url = f"{endpoint.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Tatva-Studio/1.2"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", []) if m.get("name")]
                return models
    except Exception:
        pass
    return []


def fetch_nvidia_models(api_key: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
    """
    Query live NVIDIA build.nvidia.com catalog endpoint /v1/models.
    Returns (model_ids, error_message).
    """
    if not api_key:
        try:
            from tatva.config import get_nvidia_api_key
            api_key = get_nvidia_api_key()
        except Exception:
            api_key = None

    if not api_key or not api_key.strip():
        return [], "Failed to fetch NVIDIA model catalog. Verify your nvapi-... key."

    api_key = api_key.strip()
    url = "https://integrate.api.nvidia.com/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "Tatva-Studio/1.2",
        "Accept": "application/json",
    }

    try:
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models_data = data.get("data", [])
                if not models_data and isinstance(data, list):
                    models_data = data
                
                model_ids = [m.get("id") for m in models_data if isinstance(m, dict) and m.get("id")]
                if model_ids:
                    return model_ids, None
                return [], "No models returned from NVIDIA catalog endpoint."
            else:
                return [], f"Failed to fetch NVIDIA model catalog. Status {resp.status}."
    except urllib.error.HTTPError as e:
        if e.code == 429:
            return [], "HTTP 429 Rate Limit Exceeded: NVIDIA API rate limit reached. You can retry or switch back to local Ollama inference."
        return [], "Failed to fetch NVIDIA model catalog. Verify your nvapi-... key."
    except Exception as e:
        return [], f"Failed to fetch NVIDIA model catalog. Verify your nvapi-... key. ({e})"


class LLMProvider:
    """
    LLM query provider supporting local Ollama models, NVIDIA API, and cloud Anthropic Claude API.
    """

    def __init__(self, ollama_endpoint: str = "http://localhost:11434") -> None:
        self.ollama_endpoint = ollama_endpoint.rstrip("/")

    def get_available_models(self, nvidia_key: Optional[str] = None) -> List[str]:
        """Return combined list of NVIDIA models, local Ollama models, and cloud API models."""
        nvidia_models, err = fetch_nvidia_models(nvidia_key)
        formatted_nvidia = [f"NVIDIA: {m}" for m in nvidia_models] if nvidia_models else []

        local_models = get_local_ollama_models(self.ollama_endpoint)
        formatted_local = [f"Ollama: {m} (Local / Free)" for m in local_models]

        cloud_models = ["Claude 3.5 Sonnet (Anthropic)", "DeepSeek-R1 (Cloud)", "Custom Endpoint"]

        return formatted_nvidia + formatted_local + cloud_models

    def query(
        self,
        prompt: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model_name: str,
        api_key: Optional[str] = None,
        stream_callback: Optional[Any] = None,
    ) -> Tuple[str, float]:
        """
        Query specified model. Returns (response_text, cost_usd).
        """
        # Case 1: NVIDIA API models
        if "NVIDIA:" in model_name or "nvidia" in model_name.lower() or (api_key and api_key.startswith("nvapi-")):
            try:
                from tatva.config import get_nvidia_api_key
                nv_key = api_key if (api_key and api_key.startswith("nvapi-")) else get_nvidia_api_key()
            except Exception:
                nv_key = api_key

            if not nv_key:
                raise RuntimeError("Failed to fetch NVIDIA model catalog. Verify your nvapi-... key.")

            res_text = self._query_nvidia(prompt, system_prompt, messages, model_name, nv_key, stream_callback=stream_callback)
            return res_text, 0.00000

        # Case 2: Local Ollama
        if "Ollama:" in model_name or "localhost" in model_name or "qwen" in model_name.lower():
            raw_model = model_name.replace("Ollama:", "").replace("(Local / Free)", "").strip()
            res_text = self._query_ollama(prompt, system_prompt, messages, raw_model)
            return res_text, 0.00000

        # Case 3: Anthropic Claude API
        if api_key and ("Claude" in model_name or "Anthropic" in model_name):
            try:
                import anthropic

                client = anthropic.Anthropic(api_key=api_key)
                msgs_payload = [{"role": m["role"], "content": m["content"]} for m in messages]
                if not msgs_payload or msgs_payload[-1]["content"] != prompt:
                    msgs_payload.append({"role": "user", "content": prompt})

                response = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    system=system_prompt,
                    messages=msgs_payload,
                )
                text = response.content[0].text
                in_tok = response.usage.input_tokens
                out_tok = response.usage.output_tokens
                cost = round(in_tok * (3.0 / 1e6) + out_tok * (15.0 / 1e6), 6)
                return text, cost
            except Exception as e:
                raise RuntimeError(f"Anthropic API Error: {e}")

        # Fallback note
        raise RuntimeError(f"No valid LLM provider configured for model '{model_name}'.")

    def _query_nvidia(
        self,
        prompt: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model_name: str,
        api_key: str,
        stream_callback: Optional[Any] = None,
    ) -> str:
        """
        Post request to NVIDIA build.nvidia.com /v1/chat/completions with stream=True.
        """
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        raw_model = model_name.replace("NVIDIA:", "").strip()

        formatted_messages = []
        if system_prompt:
            formatted_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            if isinstance(m, dict) and "role" in m and "content" in m:
                formatted_messages.append({"role": m["role"], "content": m["content"]})
        if not formatted_messages or formatted_messages[-1].get("content") != prompt:
            formatted_messages.append({"role": "user", "content": prompt})

        payload = {
            "model": raw_model,
            "messages": formatted_messages,
            "temperature": 0.6,
            "top_p": 0.7,
            "max_tokens": 4096,
            "stream": True,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "Tatva-Studio/1.2",
        }

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            full_text = []
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                for line_bytes in resp:
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        content_str = line[6:].strip()
                        if content_str == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(content_str)
                            choices = chunk_json.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                token = delta.get("content", "")
                                if not token and "text" in choices[0]:
                                    token = choices[0].get("text", "")
                                if token:
                                    full_text.append(token)
                                    if stream_callback:
                                        stream_callback(token)
                        except Exception:
                            pass
            return "".join(full_text)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                raise RuntimeError(
                    "HTTP 429 Rate Limit Exceeded: NVIDIA API rate limit reached. "
                    "You can retry or switch back to local Ollama inference."
                )
            raise RuntimeError(
                f"NVIDIA API Error (HTTP {e.code}): Failed to complete request. Verify your nvapi-... key."
            )
        except Exception as e:
            raise RuntimeError(f"NVIDIA API Communication Error: {e}")

    def _query_ollama(
        self,
        prompt: str,
        system_prompt: str,
        messages: List[Dict[str, str]],
        model: str,
    ) -> str:
        """Post request to Ollama /api/generate or /api/chat endpoint."""
        url = f"{self.ollama_endpoint}/api/generate"
        full_prompt = f"System: {system_prompt}\n"
        for m in messages:
            full_prompt += f"{m['role'].capitalize()}: {m['content']}\n"
        full_prompt += f"User: {prompt}\nAssistant:"

        payload = {
            "model": model or "qwen2.5-coder:7b",
            "prompt": full_prompt,
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        with urllib.request.urlopen(req, timeout=60.0) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            return res_data.get("response", "")
