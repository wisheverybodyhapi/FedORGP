import logging
import os

class Logger:
    def __init__(self, path, log_file, log_level=logging.DEBUG):
        # 配置日志记录器
        self.logger = logging.getLogger()
        self.logger.setLevel(log_level)
        
        if not os.path.exists(path):
            os.makedirs(path)

        self.log_file = os.path.join(path, log_file)
            
        # 移除所有已存在的 handler，避免重复日志
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 设置 FileHandler
        file_handler = logging.FileHandler(self.log_file, mode='a')

        # 强制无缓冲或行缓冲
        file_handler.stream = open(file_handler.baseFilename, mode='a', buffering=1)  # 行缓冲

        file_handler.setLevel(log_level)

        # 设置日志格式
        formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        # 将 handler 添加到 logger 中
        self.logger.addHandler(file_handler)

    def write(self, message):
        self.logger.debug(message)
        # 确保日志立即写入文件
        for handler in self.logger.handlers:
            handler.flush()