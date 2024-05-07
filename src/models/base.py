from abc import ABC


class BaseInferencer(ABC):
    def __call__(self, w_latent, s_latent, c_latent, text_prompt: str, *args, **kwargs):
        raise NotImplementedError

    def to_image(self, w_latent, s_latent, c_latent, *args, **kwargs):
        raise NotImplementedError
