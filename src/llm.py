from openai import OpenAI



class llm:
    def __init__(self, base_url=None, api_key=None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    
    def generate_pos(self, chart, origin_title, num_pos=2):
        """ 利用图表信息及原始标题生成正例 """
        pass

    
    def generate_neg(self, num_neg=6):
        pass