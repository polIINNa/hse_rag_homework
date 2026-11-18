from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler


class Models:
    def __init__(self):
        self.model, self.tokenizer = load('mlx-community/Qwen2.5-3B-Instruct-4bit', revision='4f83f8f146fdf28b512a06562b671d7af4fab457')

    def generate(self, messages):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return generate(
            self.model, self.tokenizer, prompt=prompt, max_tokens=600,
            sampler=make_sampler(temp=0), verbose=False,
        )
