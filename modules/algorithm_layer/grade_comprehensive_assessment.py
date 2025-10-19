"""
等级综合评定模块

实现基于TOPSIS结果的等级综合评定功能，包括级别特征值计算和二元语义评定
基于公式4-32和4-33实现等级综合评定算法

主要功能：
1. 级别特征值计算（公式4-32）
2. 二元语义评定（公式4-33）
3. 等级综合评定结果生成

应用场景：
- 风险等级综合评定
- 绩效等级综合评定  
- 产品质量等级综合评定
- 业务对象等级划分

算法流程：
TOPSIS相对接近度矩阵 → 归一化综合隶属度向量 → 级别特征值计算 → 二元语义评定
"""

import numpy as np
from typing import Dict, List, Tuple, Union, Any


class GradeComprehensiveAssessment:
    """等级综合评定类 - 基于TOPSIS结果的等级综合评定"""
    
    @staticmethod
    def calculate_level_characteristic_values(membership_scores: Dict[str, List[float]]) -> Dict[str, float]:
        """
        计算级别特征值（公式4-32）
        
        Args:
            membership_scores: 综合隶属度向量字典，格式为{"T1": [u1, u2, u3, u4], ...}
            
        Returns:
            级别特征值字典，格式为{"T1": v1, "T2": v2, ...}
        """
        level_characteristic_values = {}
        
        for obj_name, u_vector in membership_scores.items():
            # 计算级别特征值 v_j = sum_{k=1}^{h} k * u_{jk}
            v_value = 0.0
            for k, u_jk in enumerate(u_vector, start=1):
                v_value += k * u_jk
            
            level_characteristic_values[obj_name] = v_value
        
        return level_characteristic_values
    
    @staticmethod
    def perform_binary_semantic_assessment(level_characteristic_values: Dict[str, float], 
                                         num_levels: int = 4) -> Dict[str, Tuple[int, float]]:
        """
        执行二元语义评定（公式4-33）
        
        Args:
            level_characteristic_values: 级别特征值字典
            num_levels: 评估等级数量，默认为4级
            
        Returns:
            二元语义评定结果字典，格式为{"T1": (k, alpha), "T2": (k, alpha), ...}
        """
        binary_semantic_results = {}
        
        for obj_name, v_value in level_characteristic_values.items():
            # 计算风险等级 k = Round(v_j)
            k = round(v_value)
            
            # 确保k在有效范围内 [1, num_levels]
            k = max(1, min(k, num_levels))
            
            # 计算符号偏移值 alpha_{jk} = v_j - k
            alpha = v_value - k
            
            binary_semantic_results[obj_name] = (k, alpha)
        
        return binary_semantic_results
    
    @staticmethod
    def generate_comprehensive_assessment_report(membership_scores: Dict[str, List[float]],
                                               level_characteristic_values: Dict[str, float],
                                               binary_semantic_results: Dict[str, Tuple[int, float]]) -> str:
        """
        生成等级综合评定报告
        
        Args:
            membership_scores: 综合隶属度向量
            level_characteristic_values: 级别特征值
            binary_semantic_results: 二元语义评定结果
            
        Returns:
            格式化的评定报告字符串
        """
        report_lines = []
        
        # 报告标题
        report_lines.append("## 🎯 等级综合评定报告")
        report_lines.append("")
        
        # 评估概述
        num_objects = len(membership_scores)
        num_levels = len(next(iter(membership_scores.values())))
        report_lines.append("### 📊 评估概述")
        report_lines.append(f"- **评估对象数量**: {num_objects} 个")
        report_lines.append(f"- **评估等级数量**: {num_levels} 级")
        report_lines.append(f"- **评定方法**: 二元语义评定")
        report_lines.append("")
        
        # 评定结果表格
        report_lines.append("### 📈 等级综合评定结果")
        report_lines.append("| 业务对象 | 综合隶属度向量 | 级别特征值 | 二元语义 | 最终等级 |")
        report_lines.append("|---------|---------------|-----------|----------|----------|")
        
        for obj_name in membership_scores.keys():
            u_vector = membership_scores[obj_name]
            v_value = level_characteristic_values[obj_name]
            k, alpha = binary_semantic_results[obj_name]
            
            # 格式化综合隶属度向量
            u_formatted = ", ".join([f"{u:.3f}" for u in u_vector])
            
            # 格式化二元语义
            binary_semantic = f"({k}, {alpha:+.3f})"
            
            # 最终等级
            final_grade = f"{k}级"
            
            report_lines.append(f"| {obj_name} | ({u_formatted}) | {v_value:.3f} | {binary_semantic} | {final_grade} |")
        
        report_lines.append("")
        
        # 评定说明
        report_lines.append("### 📋 评定说明")
        report_lines.append("1. **综合隶属度向量**: 归一化后的相对接近度向量，表示业务对象在各等级上的隶属程度")
        report_lines.append("2. **级别特征值**: 综合隶属度向量的加权平均值，反映业务对象的整体等级倾向")
        report_lines.append("3. **二元语义**: (k, α) 表示评定等级为k级，符号偏移值为α")
        report_lines.append("4. **最终等级**: 根据二元语义确定的业务对象等级")
        report_lines.append("")
        
        # 符号偏移值解释
        report_lines.append("### 🔍 符号偏移值解释")
        report_lines.append("- **α > 0**: 级别特征值略高于当前等级，偏向更高等级")
        report_lines.append("- **α = 0**: 级别特征值正好等于当前等级")
        report_lines.append("- **α < 0**: 级别特征值略低于当前等级，偏向更低等级")
        report_lines.append("")
        
        return "\n".join(report_lines)
    
    @staticmethod
    def assess_comprehensive_grade(membership_scores: Dict[str, List[float]], 
                                  num_levels: int = 4) -> Dict[str, Any]:
        """
        执行完整的等级综合评定流程
        
        Args:
            membership_scores: 综合隶属度向量字典
            num_levels: 评估等级数量
            
        Returns:
            完整的评定结果字典
        """
        # 计算级别特征值
        level_characteristic_values = GradeComprehensiveAssessment.calculate_level_characteristic_values(
            membership_scores
        )
        
        # 执行二元语义评定
        binary_semantic_results = GradeComprehensiveAssessment.perform_binary_semantic_assessment(
            level_characteristic_values, num_levels
        )
        
        # 生成评定报告
        assessment_report = GradeComprehensiveAssessment.generate_comprehensive_assessment_report(
            membership_scores, level_characteristic_values, binary_semantic_results
        )
        
        # 返回完整结果
        return {
            "membership_scores": membership_scores,
            "level_characteristic_values": level_characteristic_values,
            "binary_semantic_results": binary_semantic_results,
            "assessment_report": assessment_report,
            "num_levels": num_levels,
            "num_objects": len(membership_scores)
        }


def test_grade_comprehensive_assessment():
    """测试等级综合评定功能"""
    
    # 测试数据：业务对象1的综合隶属度向量 (0.382, 0.297, 0.205, 0.116)
    test_membership_scores = {
        "业务对象1(T_1)": [0.382, 0.297, 0.205, 0.116],
        "业务对象2(T_2)": [0.250, 0.250, 0.250, 0.250],  # 等权重测试
        "业务对象3(T_3)": [0.100, 0.200, 0.300, 0.400],  # 偏向高等级
        "业务对象4(T_4)": [0.400, 0.300, 0.200, 0.100]   # 偏向低等级
    }
    
    # 执行等级综合评定
    assessment_result = GradeComprehensiveAssessment.assess_comprehensive_grade(
        test_membership_scores, num_levels=4
    )
    
    # 验证业务对象1的结果
    v_value = assessment_result["level_characteristic_values"]["业务对象1(T_1)"]
    k, alpha = assessment_result["binary_semantic_results"]["业务对象1(T_1)"]
    
    print("测试结果验证:")
    print(f"业务对象1(T_1) 级别特征值: {v_value:.3f} (期望: 2.056)")
    print(f"业务对象1(T_1) 二元语义: ({k}, {alpha:.3f}) (期望: (2, 0.056))")
    print(f"业务对象1(T_1) 最终等级: {k}级")
    
    # 打印完整报告
    print("\n" + assessment_result["assessment_report"])
    
    return assessment_result


if __name__ == "__main__":
    # 运行测试
    test_grade_comprehensive_assessment()