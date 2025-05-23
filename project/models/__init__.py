from typing import Dict, List, Any, Union
import importlib


def find_model_using_name(model_class):
    """Import the module "models/[model_name].py".
    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseLLM,
    and it is case-insensitive.
    """
    model_filename = "models.base_model"
    modellib = importlib.import_module(model_filename)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_class.lower() \
           and issubclass(cls, BaseLLM):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, model_class))
        exit(0)

    return model


def find_embedding_using_name(model_class):
    model_filename = "models.embeddings"
    modellib = importlib.import_module(model_filename)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_class.lower() \
           and issubclass(cls, EmbeddingModel):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of EmbeddingModel with class name that matches %s in lowercase." % (model_filename, model_class))
        exit(0)

    return model




def create_model(opt):
    """Create a model given the option.
    This function warps the class CappedDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt.model_class)
    instance = model(opt.model_config)
    print("model [%s] was created" % type(instance).__name__)
    return instance


def create_embedding_model(embedding_config):
    model = find_embedding_using_name(embedding_config.get('type'))
    instance = model(embedding_config)
    print("model [%s] was created" % type(instance).__name__)
    return instance

    


class BaseLLM:
    """Base class for all LLM models"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Common model loading logic"""
        raise NotImplementedError
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Common generation interface"""
        raise NotImplementedError



class EmbeddingModel:
    """Base class for embedding models."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings for text.
        
        Args:
            text: Text or list of texts to embed
            
        Returns:
            List of embedding vectors
        """
        raise NotImplementedError