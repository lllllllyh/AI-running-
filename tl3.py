from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm  # 导入 tqdm 库用于显示进度条

# 下载模型检查点到本地目录 model_dir
# model_dir = snapshot_download('qwen/Qwen-VL')
model_dir = "/home/develop/.cache/modelscope/hub/qwen/Qwen-VL-Chat/"

# 加载本地检查点
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cuda",
    trust_remote_code=True
).eval()

# 定义一个提示模板来过滤掉非人类描述
prompt_template = """
请描述以下图片中的人物情况：
图片路径：<img>{image_path}</img>
问题：{user_question}
只描述图片中的人物，忽略图片中的背景和其他无生命的物体。
"""
import json


# 指定 JSONL 文件路径
jsonl_file_path = "/home/develop/Flickr30k-CN/train_texts.jsonl"

# 读取 JSON 文件并转换为 DataFrame
df_text = pd.read_json(jsonl_file_path, lines=True)

list_image_id = df_text.iloc[0:200, 2]

# 初始化历史记录
history = None
list_text_id = []
list_text = []
text_id = 0

# 使用 tqdm 创建进度条
with tqdm(total=40 * 5) as pbar:  # 总任务数量为 29 张图片，每张图片 5 次提问
    for i in range(0, 40):
        image_path1 = f'train_img2/{i}.png'  # 格式化图片路径

        for j in range(0, 5):
            # 获取用户输入
            list_user_input =[ '描述图中人物信息。','描述图上人的穿着','对图上人的穿着进行详细描述','图中人物在干什么','详细描述图上的人']
            query = list_user_input[j]
            input_text = prompt_template.format(image_path=image_path1, user_question=query)
            response, history = model.chat(tokenizer, query=input_text, history=None)
            list_text.append(response)
            list_text_id.append(text_id)
            text_id = text_id + 1
            pbar.update(1)  # 每次循环更新进度条

datadf = pd.DataFrame({
    'text_id': list_text_id,
    'text': list_text,
    'image_ids': list_image_id
})
print(datadf)
json_str = datadf.to_csv('out.txt',encoding='UTF-8',sep=',', index=False)
