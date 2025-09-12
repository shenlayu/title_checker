这是一个利用 [ChartGalaxy](https://github.com/ChartGalaxy/ChartGalaxy) 数据集训练模型，检测标题与图表信息是否匹配的代码仓库。

# 数据格式
TODO

# 启动
## 环境配置
```bash
conda create -n title_checker python=3.10
conda activate title_checker
pip install -r requirements.txt
```

环境变量与配置 TODO

## 生成数据
```bash
python -m src.generate
```

## 训练模型
模型下载 TODO

```bash
python -m src.sft
```