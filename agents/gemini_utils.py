import os
import time
from google import genai
from google.genai import types

# The primary and fallback Gemini models to try in order of preference.
# gemini-2.5-flash-lite has a separate, higher quota on the free tier.
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite"]

def call_gemini_with_fallback(prompt: str, temperature: float = 0.4, max_tokens: int = 4096) -> str:
    """
    Sends a prompt to Gemini with fallback model iteration and 503 retry logic.
    
    If the call fails with a 503 (Unavailable) error, it retries up to 2 times with a 30s delay.
    If the retries are exhausted or it fails with a 429 (Resource Exhausted) error,
    it falls back immediately to the next model in the preference list.
    
    Args:
        prompt:      The prompt string to send.
        temperature: Controls randomness.
        max_tokens:  Max tokens to output.
        
    Returns:
        The text response from the model.
        
    Raises:
        ValueError: If GEMINI_API_KEY is not set.
        Exception: If all models and retries are exhausted.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your key from "
            "https://aistudio.google.com/app/apikey"
        )

    client = genai.Client(api_key=api_key)

    for i, model_name in enumerate(GEMINI_MODELS):
        print(f"[GeminiUtils] Trying model: {model_name}")
        print(f"[GeminiUtils] → Gemini ({model_name}): {len(prompt):,} chars, temp={temperature}")

        max_retries = 2
        for attempt in range(max_retries + 1):  # 0 (initial call), 1, 2 (retries)
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text
            except Exception as e:
                err_str = str(e)
                is_429 = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
                is_503 = "503" in err_str or "UNAVAILABLE" in err_str
                
                if is_429:
                    if i < len(GEMINI_MODELS) - 1:
                        next_model = GEMINI_MODELS[i + 1]
                        print(f"[GeminiUtils] Quota exhausted for {model_name}, falling back to {next_model}")
                        break  # Break retry loop to try the next model
                    else:
                        raise e
                elif is_503:
                    if attempt == max_retries:
                        if i < len(GEMINI_MODELS) - 1:
                            next_model = GEMINI_MODELS[i + 1]
                            print(f"[GeminiUtils] Model {model_name} unavailable after {max_retries} retries, falling back to {next_model}")
                            break  # Break retry loop to try the next model
                        else:
                            raise e
                    print(f"[GeminiUtils] Gemini returned 503, retrying in 30s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(30)
                else:
                    raise e
                    
    raise Exception("All models and retries were exhausted.")
