"""
资源模块

包含示例数据和静态资源
"""

import os

# 资源路径常量
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), 'sample_data')

__all__ = ['SAMPLE_DATA_DIR']