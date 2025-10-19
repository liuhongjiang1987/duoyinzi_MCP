"""
VIKOR综合评估模块

实现VIKOR方法用于多因子综合评估系统
基于妥协解的概念进行排序

主要功能：
1. VIKOR综合评估算法
2. 权重设置与管理
3. S值和R值计算
4. Q值计算与排序

应用场景：
- 风险综合评估
- 绩效综合评估
- 产品质量综合评估

算法流程：
规范化特征值矩阵 → 权重设置 → S值和R值计算 → Q值计算 → 排序
"""

import numpy as np
from typing import Union, List, Dict


class VIKORComprehensiveEvaluation:
    """VIKOR综合评估类 - 专注于多因子综合评估"""
    
    @staticmethod
    def calculate_comprehensive_scores(normalized_matrices: Dict[str, np.ndarray], 
                                     weights: List[float] = None,
                                     v: float = 0.5) -> Dict[str, Dict[str, Union[List[float], float]]]:
        """
        使用VIKOR方法计算综合得分
        
        Args:
            normalized_matrices: 规范化特征值矩阵字典，格式为{"T1": 矩阵, "T2": 矩阵, ...}
            weights: 指标权重列表，默认为等权重
            v: 策略权重，0≤v≤1，v=0表示以个体最大遗憾为基础，v=1表示以群体多数为基础
            
        Returns:
            综合得分字典，格式为{"T1": {"S": [s1, s2, ...], "R": [r1, r2, ...], "Q": [q1, q2, ...]}, ...}
        """
        comprehensive_scores = {}
        
        # 获取评估对象数量和级别数
        num_objects = len(normalized_matrices)
        sample_matrix = next(iter(normalized_matrices.values()))
        num_factors, num_levels = sample_matrix.shape
        
        # 默认权重为等权重
        if weights is None:
            weights = [1.0 / num_factors] * num_factors
        
        # 检查权重长度
        if len(weights) != num_factors:
            raise ValueError("权重数量必须等于因子数量")
        
        # 将权重转换为numpy数组便于计算
        weights_array = np.array(weights)
        
        # 对每个评估对象计算VIKOR综合得分
        for obj_name, matrix in normalized_matrices.items():
            # 初始化结果存储
            S_values = []
            R_values = []
            
            # 对每个级别计算S值和R值
            for level_idx in range(num_levels):
                # 获取当前级别的所有因子隶属度值
                level_memberships = matrix[:, level_idx]
                
                # 计算S值（权重策略标准）
                S = np.sum(weights_array * (1 - level_memberships))
                S_values.append(S)
                
                # 计算R值（个体遗憾标准）
                R = np.max(weights_array * (1 - level_memberships))
                R_values.append(R)
            
            # 计算Q值
            S_array = np.array(S_values)
            R_array = np.array(R_values)
            
            # 计算S*和S-（S值的最优解和最劣解）
            S_star = np.min(S_array)
            S_minus = np.max(S_array)
            
            # 计算R*和R-（R值的最优解和最劣解）
            R_star = np.min(R_array)
            R_minus = np.max(R_array)
            
            # 避免除零错误
            S_diff = S_minus - S_star
            R_diff = R_minus - R_star
            
            # 计算Q值
            if S_diff == 0 and R_diff == 0:
                Q_values = np.zeros_like(S_array)
            elif S_diff == 0:
                Q_values = v * np.zeros_like(S_array) + (1 - v) * (R_array - R_star) / R_diff
            elif R_diff == 0:
                Q_values = v * (S_array - S_star) / S_diff + (1 - v) * np.zeros_like(R_array)
            else:
                Q_values = v * (S_array - S_star) / S_diff + (1 - v) * (R_array - R_star) / R_diff
            
            comprehensive_scores[obj_name] = {
                "S": S_values,
                "R": R_values,
                "Q": Q_values.tolist()
            }
        
        return comprehensive_scores


def test_vikor_comprehensive_evaluation():
    """测试VIKOR综合评估"""
    from modules.algorithm_layer.membership_functions import LowerBoundMembershipFunctions
    
    print("=== VIKOR综合评估测试 ===")
    
    # 获取示例数据
    mf = LowerBoundMembershipFunctions()
    factors_data = mf.get_sample_factors_data()
    factors_params = mf.get_predefined_factors_params()
    
    # 计算规范化矩阵
    normalized_matrices = mf.calculate_normalized_matrix(factors_data, factors_params)
    
    # 创建VIKOR评估器
    vikor_evaluator = VIKORComprehensiveEvaluation()
    
    # 使用等权重计算综合得分
    weights = [0.25, 0.25, 0.25, 0.25]
    comprehensive_scores = vikor_evaluator.calculate_comprehensive_scores(normalized_matrices, weights)
    
    print("VIKOR综合得分矩阵:")
    print("业务对象  S1     S2     S3     S4     R1     R2     R3     R4     Q1     Q2     Q3     Q4")
    for obj_name, scores in comprehensive_scores.items():
        S_values = scores["S"]
        R_values = scores["R"]
        Q_values = scores["Q"]
        print(f"{obj_name}      {S_values[0]:.3f}  {S_values[1]:.3f}  {S_values[2]:.3f}  {S_values[3]:.3f}  "
              f"{R_values[0]:.3f}  {R_values[1]:.3f}  {R_values[2]:.3f}  {R_values[3]:.3f}  "
              f"{Q_values[0]:.3f}  {Q_values[1]:.3f}  {Q_values[2]:.3f}  {Q_values[3]:.3f}")


if __name__ == "__main__":
    test_vikor_comprehensive_evaluation()