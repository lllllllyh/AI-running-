import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import os
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def process_images_in_folder(folder_path, output_file, input_size=448, max_num=6, limit=40):
    # 确保文件夹路径存在
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    image_count = 0
    image_files = []
    for f in sorted(os.listdir(folder_path)):
        if f.endswith(('.png', '.jpg', '.jpeg')):
            image_files.append(os.path.join(folder_path, f))
            image_count += 1
            if image_count >= limit:  # 达到限制数量后停止添加
                break

    
    # 确保模型已经加载
    model = AutoModel.from_pretrained(
        "/home/develop/InternVL/",  # 使用你自己的模型路径
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained("/home/develop/InternVL/", trust_remote_code=True)  # 使用你自己的模型路径

    # 准备保存结果的文件
    with open(output_file, 'w') as f:
        for image_file in image_files:
            # 加载并处理图片
            pixel_values = load_image(image_file, input_size=input_size, max_num=max_num).to(torch.bfloat16).cuda()

            # 准备问题和生成配置
            questions = ['<image>\n描述图片']
            generation_config = {
                'num_beams': 1,
                'max_new_tokens': 1024,
                'do_sample': False,
            }

            # 进行批处理
            responses = model.batch_chat(tokenizer, pixel_values,
                                         num_patches_list=[pixel_values.size(0)],
                                         questions=questions,
                                         generation_config=generation_config)

            # 将结果写入文件
            for response in responses:
                f.write(response + '\n')

# 调用函数处理指定文件夹中的图片
process_images_in_folder("/home/develop/Qwen-VL/train_img2/", "output1.txt")


