import time

import pandas as pd
from pandas import Series

from .ai_model import AIModel
from google import genai

# https://ai.google.dev/gemini-api/docs/embeddings
class GeminiAIModel(AIModel):

    def __init__(self, config):
        super().__init__(config)
        self.ai_client = None
        self.embedding_model = "text-embedding-004"
        self.chat_model = "gemini-2.0-flash"


    def init_llm_model(self):
        if 'gemini_api_key' in list(self.config):
            self.ai_client = genai.Client(api_key=self.config['gemini_api_key'])
        else:
            raise ValueError("No Gemini API key found. Please provide an API key to proceed.")
        return self

    def get_embedding(self, text):
        print("Requesting embeddings for text: {}".format(text))
        result = self.ai_client.models.embed_content(model=self.embedding_model, contents=text)
        [embedding] = result.embeddings
        return embedding.values

    def get_embeddings(self, text_series : Series):
        batch_size = 100
        batches = []
        for i in range(0, len(text_series), batch_size):
            batch = text_series[i : i + batch_size].tolist()
            print("Requesting embeddings for multiple elements, size: {}".format(len(batch)))
            result = self.ai_client.models.embed_content(model=self.embedding_model, contents=batch)
            batches.extend(result.embeddings)

        return pd.Series(e.values for e in batches)

    def get_llm_response(self, prompt):
        print("Requesting content for prompt: {}".format(prompt))
        llm_response = None
        try:
            llm_response = self.ai_client.models.generate_content(model=self.chat_model, contents=prompt)
        except Exception as e:
            llm_response = None
            print('Gemini failed to generate content, error: {}'.format(e))
        if llm_response:
            label = llm_response.text # type: ignore
            return ''.join(['*',label]) # add a * to indicate this description is AI generated
        else:
            return None
