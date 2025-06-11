from .ai_model_gemini import GeminiAIModel
from .ai_model_openai import OpenAIAIModel
from .ai_model import AIModel


def get_ai_model(config) -> AIModel:
    ai_model = None
    if 'OpenAI_api_key' in list(config):
        ai_model = OpenAIAIModel(config)
        print("API key found for OpenAI. Returning OpenAI model")
    elif 'gemini_api_key' in list(config):
        ai_model = GeminiAIModel(config)
        print("API key found for Gemini. Returning Gemini model")
    else:
        raise ValueError("No OpenAI/Gemini API key found. Please provide an API key to proceed.")

    ai_model.init_llm_model()
    return ai_model
