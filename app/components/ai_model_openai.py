import time

from pandas import Series

from .ai_model import AIModel
from openai import OpenAI


class OpenAIAIModel(AIModel):

    def __init__(self, config):
        super().__init__(config)
        self.ai_client = None
        self.embedding_model = "text-embedding-ada-002"
        self.chat_model = "gpt-4o-mini"

    def init_llm_model(self):
        if 'OpenAI_api_key' in list(self.config):
            self.ai_client = OpenAI(api_key=self.config['OpenAI_api_key'])
        else:
            raise ValueError("No OpenAI API key found. Please provide an API key to proceed.")
        return self

    def get_embedding(self, text):
        text = str(text).replace("\n", " ")
        if self.ai_client:
            try:
                response = self.ai_client.embeddings.create(input=[text], model=self.embedding_model)
                return response.data[0].embedding
            except Exception as e:
                print(f"Error generating OpenAI embedding: {str(e)}")
        else:
            raise ValueError("No OpenAI client available. Please provide an OpenAI API key.")
        return None

    def get_embeddings(self, text_series: Series):
        return text_series.apply(lambda x: self.get_embedding(x))

    def get_llm_response(self, prompt):
        llm_response = None
        try:
            llm_response = self.ai_client.chat.completions.create(model=self.chat_model, messages=prompt)
        except:
            time.sleep(1)
            print('retry')
            try:
                llm_response = self.ai_client.chat.completions.create(model=self.chat_model, messages=prompt)
            except:
                llm_response = None
                print('openai fail')  # if all fail good chance the context length is too long
        if llm_response:
            label = llm_response.choices[0].message.content  # type: ignore
            return ''.join(['*', label])  # add a * to indicate this description is AI generated
        else:
            return None
