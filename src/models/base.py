from abc import ABC


class BaseInferencer(ABC):
    def __call__(self, images, text_prompt: str, *args, **kwargs):
        raise NotImplementedError
