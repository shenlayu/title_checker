这是一个利用 [ChartGalaxy](https://github.com/ChartGalaxy/ChartGalaxy) 数据集训练模型，检测标题与图表信息是否匹配的代码仓库。

# 数据格式
期待每一个原始数据对应在一个文件夹中的一个子文件夹，并且至少包含：
```
dir/subdir
├── chart.png
├── chart.svg
└── data.json
```
如果格式不符合，可以调整 src/utils/generate_utils/generate_utils.py 的 generate_basic_data_dict 函数。

# 启动
## 环境配置
```bash
conda create -n title_checker python=3.10
conda activate title_checker
pip install -r requirements.txt
```

## 环境变量
```bash
export OPENAI_API_KEY=YOUR_KEY
```

## 生成数据
```bash
python -m src.generate
```
这会依次在 data/{uid} 中生成 basic_data.jsonl 以及 final_data.jsonl. 如果需要将原始图像的标题 mask 住，可以使用 tools/mask_chart_title.py

## 训练模型
模型下载
```bash
huggingface-cli download --resume-download openai/clip-vit-base-patch32 --local-dir checkpoint/pretrained
```

可以在 configs/train.config.yaml 中选择不同的 loss (BCE / InfoNCE) 以及评估指标，并调整路径和超参数。而后运行:
```bash
python -m src.sft
```