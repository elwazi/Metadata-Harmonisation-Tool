from abc import abstractmethod

class AIModel:
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def init_llm_model(self):
        pass

    @abstractmethod
    def get_embedding(self, text):
        pass

    @abstractmethod
    def get_embeddings(self, text):
        pass

    @abstractmethod
    def get_llm_response(self, text):
        pass
