import os

def check_for_model(dir_path, model_extensions=('.pth', '.pt')):
    # 遍历目录中的所有文件
    if os.path.exists(dir_path) == False:
        return False
    for file_name in os.listdir(dir_path):
        # 检查文件是否有指定的模型扩展名
        if file_name.endswith(model_extensions):
            return True
    return False