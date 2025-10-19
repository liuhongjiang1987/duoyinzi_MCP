"""
字段分析模块

负责数值型和分类型字段的统计分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from scipy import stats


def analyze_numeric_fields(df: pd.DataFrame) -> Dict[str, Any]:
    """
    分析数值型字段的统计特征
    
    Args:
        df: 包含数值型字段的DataFrame
        
    Returns:
        数值字段分析结果
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    results = {}
    
    for col in numeric_columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        results[col] = {
            'basic_stats': {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'median': float(series.median()),
                'std': float(series.std()),
                'variance': float(series.var()),
                'range': float(series.max() - series.min())
            },
            'percentiles': {
                'p25': float(series.quantile(0.25)),
                'p50': float(series.quantile(0.50)),
                'p75': float(series.quantile(0.75)),
                'p90': float(series.quantile(0.90)),
                'p95': float(series.quantile(0.95))
            },
            'distribution': {
                'skewness': float(stats.skew(series)),
                'kurtosis': float(stats.kurtosis(series)),
                'iqr': float(series.quantile(0.75) - series.quantile(0.25))
            },
            'data_quality': {
                'missing_count': int(df[col].isna().sum()),
                'missing_percentage': float(df[col].isna().sum() / len(df) * 100),
                'unique_count': int(df[col].nunique())
            }
        }
    
    return results


def analyze_categorical_fields(df: pd.DataFrame) -> Dict[str, Any]:
    """
    分析分类型字段的统计特征
    
    Args:
        df: 包含分类型字段的DataFrame
        
    Returns:
        分类字段分析结果
    """
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    results = {}
    
    for col in categorical_columns:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        value_counts = series.value_counts()
        top_categories = value_counts.head(5).to_dict()
        
        # 计算熵值（多样性评估）
        probabilities = value_counts / len(series)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        
        results[col] = {
            'frequency_analysis': {
                'top_categories': top_categories,
                'total_categories': int(len(value_counts)),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            },
            'diversity': {
                'entropy': float(entropy),
                'max_entropy': float(np.log2(len(value_counts))) if len(value_counts) > 0 else 0,
                'normalized_entropy': float(entropy / np.log2(len(value_counts))) if len(value_counts) > 1 else 0
            },
            'data_quality': {
                'missing_count': int(df[col].isna().sum()),
                'missing_percentage': float(df[col].isna().sum() / len(df) * 100),
                'unique_count': int(df[col].nunique())
            }
        }
    
    return results


def auto_detect_polarity(df: pd.DataFrame, numeric_analysis: Dict) -> Dict[str, Any]:
    """
    自动检测字段极性，引导LLM评估字段极性并返回详细结果
    
    Args:
        df: 数据DataFrame
        numeric_analysis: 数值字段分析结果
        
    Returns:
        字段极性评估结果，包含极性建议、隶属度规则和调整策略
    """
    polarity_results = {}
    
    for col, analysis in numeric_analysis.items():
        series = df[col].dropna()
        
        # 基于字段名称和统计特征进行极性判断
        col_lower = col.lower()
        
        # 常见极大型指标关键词
        max_indicators = ['growth', 'rate', 'profit', 'efficiency', 'score', 'performance', 'quality', 
                         '周转率', '库领量', '销售额', '收益率', '满意度']
        # 常见极小型指标关键词  
        min_indicators = ['cost', 'loss', 'error', 'defect', 'time', 'delay', 'risk', '库存', '单价', 
                         '成本', '损耗', '等待时间']
        
        # 基于名称的关键词匹配
        is_max_indicator = any(indicator in col_lower for indicator in max_indicators)
        is_min_indicator = any(indicator in col_lower for indicator in min_indicators)
        
        # 基于统计特征的判断
        mean_val = analysis['basic_stats']['mean']
        std_val = analysis['basic_stats']['std']
        min_val = analysis['basic_stats']['min']
        max_val = analysis['basic_stats']['max']
        
        # 计算隶属度参数（a, b, c）
        if is_max_indicator and not is_min_indicator:
            polarity = 'max'
            # 对于越大越好的字段：a < b < c
            a = min_val
            c = max_val
            b = (a + c) / 2
        elif is_min_indicator and not is_max_indicator:
            polarity = 'min'
            # 对于越小越好的字段：a > b > c
            a = max_val
            c = min_val
            b = (a + c) / 2
        else:
            # 无法明确判断极性
            polarity = 'unknown'
            a = b = c = None
        
        # 构建隶属度规则描述
        membership_rules = ""
        if polarity == 'max' and a is not None:
            membership_rules = f"""
## 对于越大越好的字段（{col}）：
- a < b < c （如{col}：{a:.2f}→{b:.2f}→{c:.2f}）
- 当实际值 ≤ a时：隶属度=0%
- 当实际值 ≥ c时：隶属度=100%
"""
        elif polarity == 'min' and a is not None:
            membership_rules = f"""
## 对于越小越好的字段（{col}）：
- a > b > c （如{col}：{a:.2f}→{b:.2f}→{c:.2f}）
- 当实际值 ≥ a时：隶属度=0%
- 当实际值 ≤ c时：隶属度=100%
"""
        
        # 构建调整策略
        adjustment_strategy = ""
        if polarity == 'max':
            adjustment_strategy = f"极性调整方法：通过取倒数的方法将越大越好的字段转换为越小越好，如{col}值{a:.2f}转换为1/{a:.2f}≈{1/a:.4f}"
        elif polarity == 'min':
            adjustment_strategy = "无需调整，已为越小越好类型"
        
        polarity_results[col] = {
            'suggested_polarity': polarity,
            'confidence': 'high' if (is_max_indicator or is_min_indicator) else 'medium',
            'reasoning': f"基于字段名称'{col}'和统计特征判断",
            'membership_rules': membership_rules.strip(),
            'adjustment_strategy': adjustment_strategy,
            'parameters': {
                'a': a,
                'b': b,
                'c': c
            } if a is not None else None,
            'detection_successful': polarity != 'unknown'
        }
    
    return polarity_results


def apply_polarity_adjustment(df: pd.DataFrame, polarity_results: Dict[str, Any]) -> pd.DataFrame:
    """
    应用极性调整策略到数据
    
    Args:
        df: 原始数据DataFrame
        polarity_results: 极性检测结果
        
    Returns:
        调整后的DataFrame
    """
    adjusted_df = df.copy()
    
    for col, result in polarity_results.items():
        if result['suggested_polarity'] == 'max' and result['detection_successful']:
            # 对越大越好的字段取倒数（避免除零）
            adjusted_df[col] = 1 / (df[col] + 1e-10)
            print(f"✅ 已对字段 '{col}' 应用极性调整（取倒数）")
    
    return adjusted_df


def generate_polarity_report(polarity_results: Dict[str, Any]) -> str:
    """
    生成极性检测报告
    
    Args:
        polarity_results: 极性检测结果
        
    Returns:
        格式化报告文本
    """
    report_lines = ["# 字段极性智能检测报告\n"]
    
    successful_detections = [col for col, result in polarity_results.items() if result['detection_successful']]
    failed_detections = [col for col, result in polarity_results.items() if not result['detection_successful']]
    
    if successful_detections:
        report_lines.append("## 成功检测的字段极性\n")
        for col in successful_detections:
            result = polarity_results[col]
            report_lines.append(f"### 字段: {col}")
            report_lines.append(f"- **极性类型**: {'越大越好 (max)' if result['suggested_polarity'] == 'max' else '越小越好 (min)'}")
            report_lines.append(f"- **置信度**: {result['confidence']}")
            report_lines.append(f"- **调整策略**: {result['adjustment_strategy']}")
            if result['membership_rules']:
                report_lines.append(result['membership_rules'])
            report_lines.append("")
    
    if failed_detections:
        report_lines.append("## 无法评估极性的字段\n")
        report_lines.append("以下字段无法从名称判断极性，建议修改表头以明确字段含义：")
        for col in failed_detections:
            report_lines.append(f"- {col}")
        report_lines.append("\n**建议**: 修改表头名称，包含明确的极性指示词（如'成本'、'收益'、'效率'等）")
    
    return '\n'.join(report_lines)