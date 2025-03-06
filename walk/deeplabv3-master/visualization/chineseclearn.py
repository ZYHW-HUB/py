import os
import re

def remove_chinese(filename):
    """
    删除文件名中的所有中文字符。
    :param filename: 原始文件名。
    :return: 删除中文字符后的文件名。
    """
    # 使用正则表达式匹配所有中文字符
    pattern = re.compile('[\u4e00-\u9fff]+')
    return pattern.sub('', filename)

def clean_filename(filename):
    """
    清理文件名，去除最后一个 .jpg 扩展名并删除中文字符。
    :param filename: 原始文件名。
    :return: 清理后的文件名。
    """
    # 首先删除中文字符
    cleaned_filename = remove_chinese(filename)
    # 然后去除最后一个 .jpg 扩展名（如果存在）
    cleaned_filename = re.sub(r'\.png$', '', cleaned_filename)
    return cleaned_filename

def rename_images_in_directory(directory):
    """
    遍历指定目录下的所有图片文件，并删除文件名中的中文字符进行重命名。
    :param directory: 目标目录路径。
    """
    if not os.path.isdir(directory):
        print(f"目录 {directory} 不存在或不是一个有效的目录。")
        return

    # 支持的图片扩展名列表
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件扩展名是否在支持的图片扩展名列表中
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                old_file_path = os.path.join(root, file)
                new_filename = clean_filename(file)
                
                # 确保新的文件名不为空
                if not new_filename:
                    print(f"跳过文件 {file}：处理后文件名为空。")
                    continue
                
                # 添加原始扩展名
                _, ext = os.path.splitext(file)
                new_file_path = os.path.join(root, new_filename + ext)
                
                # 如果新旧文件名不同，则进行重命名
                if old_file_path != new_file_path:
                    try:
                        os.rename(old_file_path, new_file_path)
                        print(f"重命名: {old_file_path} -> {new_file_path}")
                    except Exception as e:
                        print(f"重命名失败: {old_file_path} -> {new_file_path}, 错误信息: {e}")

if __name__ == "__main__":
    # 指定需要处理的目录路径
    target_directory = "walk/leftImg8bit/demoVideo/stuttgart_03"

    # 调用函数重命名目录下的图片文件
    rename_images_in_directory(target_directory)