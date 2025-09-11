import json
from openai import OpenAI
from typing import Dict, List
from tenacity import retry, stop_after_attempt, wait_exponential

class LLM:
    def __init__(
            self,
            base_url: str,
            api_key: str,
            model: str,
            pos_system_prompt: str,
            neg_system_prompt: str,
            pos_prompt: str,
            neg_prompt: str,
            raw_data_symbol: str,
            title_symbol: str,
            max_tokens: int,
            temperature: float,
            num_pos: int,
            num_neg: int
        ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        self.pos_system_prompt = pos_system_prompt
        self.neg_system_prompt = neg_system_prompt
        self.pos_prompt_template = pos_prompt
        self.neg_prompt_template = neg_prompt
        self.raw_data_symbol = raw_data_symbol
        self.title_symbol = title_symbol
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_pos = num_pos
        self.num_neg = num_neg
    
    def _build_prompt(self, template: str, raw_data: Dict, origin_title: str) -> str:
        raw_json = json.dumps(raw_data, ensure_ascii=False)
        prompt = template.replace(self.raw_data_symbol, raw_json) \
                         .replace(self.title_symbol, origin_title or "")
        return prompt

    def _call_chat(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content if response.choices else ""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate_pos(self, raw_data: Dict, origin_title: str) -> List[str]:
        """ 利用图表信息及原始标题生成正例 """
        user_prompt = self._build_prompt(self.pos_prompt_template, raw_data, origin_title)
        content = self._call_chat(self.pos_system_prompt, user_prompt)
        return [l.strip() for l in content.splitlines() if l.strip()] if content else []

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate_neg(self, raw_data: Dict, origin_title: str) -> List[str]:
        """ 利用图表信息及原始标题生成负例 """
        user_prompt = self._build_prompt(self.neg_prompt_template, raw_data, origin_title)
        content = self._call_chat(self.neg_system_prompt, user_prompt)
        return [l.strip() for l in content.splitlines() if l.strip()] if content else []