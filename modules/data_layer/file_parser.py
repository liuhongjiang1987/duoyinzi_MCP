"""
文件解析模块

支持CSV和Excel格式的文件解析
"""

import pandas as pd
from typing import Dict, Any, Union
import io
import base64


def parse_csv_file(file_content: str) -> Dict[str, Any]:
    """
    解析CSV文件内容
    
    Args:
        file_content: Base64编码的CSV文件内容或文件路径
        
    Returns:
        包含解析结果和基本信息的字典
    """
    try:
        # 尝试解码Base64内容
        try:
            decoded_content = base64.b64decode(file_content)
            df = pd.read_csv(io.BytesIO(decoded_content))
        except:
            # 如果不是Base64，尝试直接读取文件路径
            df = pd.read_csv(file_content)
        
        return {
            'success': True,
            'dataframe': df,
            'row_count': len(df),
            'column_count': len(df.columns),
            'column_names': df.columns.tolist(),
            'file_type': 'csv'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"CSV文件解析失败: {str(e)}"
        }


def parse_excel_file(file_content: str, sheet_name: str = None) -> Dict[str, Any]:
    """
    解析Excel文件内容
    
    Args:
        file_content: Base64编码的Excel文件内容或文件路径
        sheet_name: 指定工作表名称，默认为第一个工作表
        
    Returns:
        包含解析结果和基本信息的字典
    """
    try:
        # 尝试解码Base64内容
        try:
            decoded_content = base64.b64decode(file_content)
            df = pd.read_excel(io.BytesIO(decoded_content), sheet_name=sheet_name)
        except:
            # 如果不是Base64，尝试直接读取文件路径
            df = pd.read_excel(file_content, sheet_name=sheet_name)
        
        # 如果返回的是字典（多个工作表），取第一个
        if isinstance(df, dict):
            df = list(df.values())[0]
        
        return {
            'success': True,
            'dataframe': df,
            'row_count': len(df),
            'column_count': len(df.columns),
            'column_names': df.columns.tolist(),
            'file_type': 'excel'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Excel文件解析失败: {str(e)}"
        }


def detect_file_type(file_content: str) -> str:
    """
    自动检测文件类型
    
    Args:
        file_content: 文件内容或路径
        
    Returns:
        文件类型：'csv', 'excel', 或 'unknown'
    """
    if file_content.lower().endswith('.csv'):
        return 'csv'
    elif file_content.lower().endswith(('.xlsx', '.xls')):
        return 'excel'
    else:
        # 尝试通过内容检测
        try:
            # 简单的CSV检测：包含逗号分隔
            if ',' in file_content[:100]:
                return 'csv'
        except:
            pass
        return 'unknown'