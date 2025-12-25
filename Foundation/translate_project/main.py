import json
import fitz # PyMuPDF
from typing import List
import collections
import concurrent.futures
from openai_api import get_res
import os

# 思路
# 1.首先读取目标PDF中的所有文字块，将其保存在指定Json。
# 2.将目标PDF中的文字块全部删除，保留一个不包含文字块的PDF模板文件。
# 3.对Json进行处理，基于Json中的文本内容进行翻译操作，保存为新Json
# 4.基于Json，将文本块按照原格式排版写入PDF模板文件。


def remove_text_from_pdf(input_pdf_path, output_pdf_path):
    """
    从PDF文件中删除所有文本，并将结果保存为PDF模板。

    :parameters:
    - input_pdf_path (str): Path to the input PDF file.
    - output_pdf_path (str): Path where the output PDF template should be saved.
    """

    doc = fitz.open(input_pdf_path)

    # 遍历文档的每一页
    for page in doc:
        # 获取页面上的所有文字块
        text_dict = page.get_text("dict")

        # 遍历所有文字块
        for block in text_dict["blocks"]:
            if block["type"] == 0:  # 文本块类型为0
                # 获取文字块位置
                rect = fitz.Rect(block["bbox"])
                # 添加红线注释
                page.add_redact_annot(rect)

        # 应用并清除标记的内容，实际删除文本
        page.apply_redactions()

    # 保存修改后的PDF文件
    doc.save(output_pdf_path)
    doc.close()

def rgb_to_fitz_color(rgb):
    """
    将RGB值的列表转换为表示fitz中相应颜色的浮动列表。用于write_text_to_pdf_from_json读取文字颜色部分。

    :parameters:
    - rgb (List[int]): A list of three integers representing the RGB values.
    """
    # 使用位运算提取蓝色值：blue = rgb & 255。rgb & 255操作将保留rgb的低8位，即蓝色值。
    blue = rgb & 255
    # 使用右移和位运算提取绿色值：green = (rgb >> 8) & 255。rgb >> 8将rgb右移8位，然后& 255操作保留绿色值。
    green = (rgb >> 8) & 255
    # 使用右移和位运算提取红色值：red = (rgb >> 16) & 255。rgb >> 16将rgb右移16位，然后& 255操作保留红色值。
    red = (rgb >> 16) & 255
    # ：将提取的红色、绿色、蓝色值分别除以255，转换为0到1之间的浮点数，以便fitz库使用。
    rgb_result = [red / 255.0, green / 255.0, blue / 255.0]
    return rgb_result

def extract_pdf_text_to_json(input_pdf_path, output_json_path, default_fontname = "Helvetica"):
    """
    从指定的PDF中提取文本信息，并将其保存为JSON文件。

    :parameters:
    - input_pdf_path (str): Path to the input PDF file.
    - output_json_path (str): Path where the output JSON file should be saved.
    """
    with fitz.open(input_pdf_path) as input_pdf:

        # 初始化一个列表以保存提取的文本数据
        text_data = []

        # 迭代输入PDF中的每一页
        for page_num in range(len(input_pdf)):
            input_page = input_pdf[page_num]
            text_dict = input_page.get_text("dict")

            # 处理此页面上的文本块
            for block in text_dict["blocks"]:
                if block["type"] == 0:  # 文本块类型为0
                # Extract text from each line in the block
                    block_text = "\n".join(" ".join(span["text"] for span in line["spans"]) for line in block["lines"])
                    sizes = [span["size"] for line in block["lines"] for span in line["spans"]]
                    size_counter = collections.Counter(sizes)
                    common_size = size_counter.most_common(1)[0][0] if sizes else None
                    colors = [span["color"] for line in block["lines"] for span in line["spans"]]
                    color_counter = collections.Counter(colors)
                    common_color = color_counter.most_common(1)[0][0] if colors else None
                    final_color = rgb_to_fitz_color(common_color)
                    block_entry = {
                        "page": page_num,
                        "rect": list(fitz.Rect(block["bbox"])),
                        "text": block_text,
                        "size": common_size,
                        "color": final_color,
                        "fontname": block["fontname"] if "fontname" in block else default_fontname,
                    }
                    text_data.append(block_entry)

    # 将提取的文本数据保存为JSON文件
    with open(output_json_path, "w", encoding="utf-8") as json_file:
        json.dump(text_data, json_file, indent=2)
def get_pdf_context(json_path):
    # 打开并加载JSON文件
    with open(json_path, 'r') as f:
        text_data = json.load(f)

    # 创建一个空列表来存储所有文本块的文本
    blocks_text_list = []

    # 遍历每个文本块
    for block in text_data:
        blocks_text_list.append(block['text'])

    # 将文本块中文本的数量添加到列表中
    blocks_text_count = len(blocks_text_list)

    return blocks_text_list, blocks_text_count

def get_blocks_text_translated(blocks_text_list, model_name):
    simple_translate_prompt = """
       你是一名资深的翻译工作者，你的目标是帮助用户将指定文本内容翻译为中文。
       你必须满足以下要求：
       - 如果需要翻译的文本内容为空，则无需输出任何内容。请不要输出抱歉等任何说明和描述。
       - 下面为一些翻译时参考的角度，你需要考虑这些角度来确保翻译的准确性。
           - "准确性"：翻译内容需要尽可能准确地表达原文的意思。
           - "数字、公式、特殊符号与网址"：如果翻译内容涉及到数字、公式、特殊符号与网址，你无需对数字、公式、特殊符号与网址进行翻译，仅确保数字、公式、特殊符号与网址不变即可。
           - "术语"：在专业领域中，很多词汇有特定的含义，需要确保这些术语被准确地翻译。
           - "语境"：理解原文的语境对于准确翻译非常重要。你需要需要确认具体语境。
           - "语言风格"：如果原文是在特定的语言风格（如正式、口语、学术等）下写的，翻译时也应尽可能保持这种风格。
           - "文化差异"：有些表达方式可能在一种语言中很常见，但在另一种语言中却很少见。在翻译时，需要考虑这些文化差异。
           - "句子结构"：不同语种的句子结构有很大的不同，尤其是中文和英文。翻译时需要对这些差异有所了解。
           - "专业知识"：如果原文涉及到特定的专业知识，翻译时可能需要结合专业知识相关的内容以确保准确性。
           - "格式"：翻译内容需要保持原文的格式，包括段落、标题、列表等。
       下面为指定需要翻译的文本内容，你无需返回原文，无需给出任何说明和描述，仅提供最终翻译结果。
       {content}
       """

    # 初始化用于存储翻译后的文本块的列表
    translated_blocks_text_list = []
    print(blocks_text_list)
    # 创建一个线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
        # 使用map函数并发执行get_res函数
        results = executor.map(lambda block_text: get_res(simple_translate_prompt.format(content=block_text), model_name), blocks_text_list)

        # 遍历结果
        for result in results:
            try:
                # 获取翻译响应
                translate_response = result.choices[0].message.content
                # 将翻译后的响应添加到translated_blocks_text_list列表中
                translated_blocks_text_list.append(translate_response)
            except Exception as exc:
                print(f"An error occurred while translating: {exc}")

    # 返回翻译后的文本块列表
    return translated_blocks_text_list
def update_translated_json(json_path, new_json_path, blocks_text_count, translated_blocks_text_list):
    if len(translated_blocks_text_list) != blocks_text_count:
        raise ValueError("The number of translated text blocks does not match the original text blocks.")
    # 打开并加载JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        text_data = json.load(f)

    # 遍历每个文本块
    for i, block in enumerate(text_data):
        block['text'] = translated_blocks_text_list[i]
        print(block['text'])

    # 将更新后的文本数据保存为JSON文件
    with open(new_json_path, "w", encoding='utf-8') as f2:
        json.dump(text_data, f2, indent=2, ensure_ascii=False)
def write_text_to_pdf_from_json(input_json_path, template_pdf_path, output_pdf_path, ttf_path=None):
    """
    从JSON文件中读取文本信息并将其写入指定的PDF模板中。

    :parameters:
    - input_json_path (str): Path to the input JSON file containing text data.
    - template_pdf_path (str): Path to the template PDF file.
    - output_pdf_path (str): Path where the output PDF file should be saved.
    - default_fontname (str, optional): Default font name to use for the text in the output PDF. Defaults to "Helvetica".
    - ttf_path (str, optional): Path to the ttf file to be used as font. If not provided, default_fontname will be used.
    """
    # Load text data from the JSON file
    with open(input_json_path, "r",encoding="utf-8") as json_file:
        text_data = json.load(json_file)

    # 使用模板文件
    output_pdf = fitz.open(template_pdf_path)

    # 遍历文本数据
    for block in text_data:
        # 如果需要，将新页面添加到输出PDF
        while block["page"] >= len(output_pdf):
            output_pdf.new_page()

        output_page = output_pdf[block["page"]]
        
        if ttf_path is not None:
            default_fontname = "Sourcehan"
            output_page.insert_font(fontname=default_fontname,fontfile=ttf_path)
        rect = fitz.Rect(*block["rect"])

        rect_height = block["rect"][3] - block["rect"][1]
        rect_width = block["rect"][2] - block["rect"][0]
        rect_gradient = rect_height / rect_width
        # 计算文本块的宽高比
        if rect_gradient < 10:
            bbox = -1
            # 尝试将文本插入矩形框，判断是否能正常插入，通过减小字体的尺寸，直到可以正常插入为止
            while bbox <= 0:
                bbox = output_page.insert_textbox(
                    rect,
                    block["text"],
                    fontname=default_fontname,
                    fontsize=block["size"],
                    color=block["color"],
                    align=fitz.TEXT_ALIGN_LEFT,  # 左对齐
                )
                block["size"] -= 0.5
        else:
            x_1 = block["rect"][2]
            y_1 = block["rect"][3]
            output_page.insert_text(
                (x_1,y_1),
                block["text"],
                fontname=default_fontname,
                fontsize=block["size"],
                color=block["color"],
                # align=fitz.TEXT_ALIGN_LEFT,  # 左对齐
                rotate=90
            )
    
    # 保存新的PDF
    output_pdf.save(output_pdf_path)
    output_pdf.close()


if __name__ == '__main__':
    input_pdf_path = "input.pdf"
    text_json_path = "test.json"
    template_pdf_path = "no_text.pdf"
    output_pdf_path = "output.pdf"
    ttf_path = "ttf/SourceHanSerif-VF.ttf.ttc"
    new_json_path = "translated.json"
    
    # 从PDF中删除文本块，构建模版文件
    remove_text_from_pdf(input_pdf_path, template_pdf_path)
    print(f"PDF without text content has been saved to '{template_pdf_path}'")

    # 从PDF中提取文本信息，保存为Json
    extract_pdf_text_to_json(input_pdf_path, text_json_path)
    print(f"Text extracted from PDF and saved to '{text_json_path}'")

    # 从Json中获取文本内容
    blocks_text_list, blocks_text_count = get_pdf_context(text_json_path)
    model_name = "gpt-3.5-turbo"

    translated_blocks_text_list = get_blocks_text_translated(blocks_text_list, model_name)
    # 更新翻译后的JSON文件
    update_translated_json(text_json_path, new_json_path, blocks_text_count, translated_blocks_text_list)
    print(f"Translated text has been saved to 'translated.json'")

    write_text_to_pdf_from_json(new_json_path, template_pdf_path, output_pdf_path, ttf_path)
    print(f"PDF with translated text has been saved to '{output_pdf_path}'")
