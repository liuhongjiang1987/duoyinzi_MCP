"""
下界型隶属度函数模块

专注于下界型特征值级别隶属函数，用于多因子综合评估系统
基于公式4-11到4-13实现下界型级别隶属函数

主要功能：
1. 下界型特征值级别隶属函数
2. 规范化特征值级别矩阵计算
3. TOPSIS综合评估算法

应用场景：
- 员工等级评估
- 数据质量等级评估  
- 风险评估
- 绩效等级评估
- 产品质量等级评估

算法流程：
因子参数值 → 下界型级别隶属函数 → 规范化特征值矩阵 → TOPSIS综合评估
"""

import numpy as np
from typing import Union, List, Dict


class LowerBoundMembershipFunctions:
    """下界型隶属度函数类 - 专注于特征值级别隶属函数"""
    
    @staticmethod
    def lower_bound_level_membership(x: Union[float, np.ndarray], level_params: List[float], 
                                     level_index: int, total_levels: int) -> Union[float, np.ndarray]:
        """
        下界型级别隶属函数（基于公式4-11到4-13）
        
        根据参考内容中的公式实现，用于多因子综合评估中的级别特征值计算
        
        Args:
            x: 输入值（特征值）
            level_params: 级别参数列表，按顺序排列的边界值[a_ij1, a_ij2, ..., a_ijh]
            level_index: 当前级别索引（从1开始）
            total_levels: 总级别数
            
        Returns:
            隶属度值
        """
        if level_index < 1 or level_index > total_levels:
            raise ValueError("级别索引必须在1到总级别数之间")
        
        if len(level_params) != total_levels:
            raise ValueError("级别参数数量必须等于总级别数")
        
        # 获取当前级别的边界参数
        a_current = level_params[level_index - 1]
        
        if level_index == 1:
            # 公式(4-11): 第一级别
            a_next = level_params[1] if total_levels > 1 else 0
            
            if isinstance(x, (int, float)):
                if x >= a_current:
                    return 1.0
                elif a_next <= x < a_current:
                    return (x - a_next) / (a_current - a_next)
                else:
                    return 0.0
            else:
                result = np.zeros_like(x)
                mask1 = x >= a_current
                mask2 = (x >= a_next) & (x < a_current)
                result[mask1] = 1.0
                result[mask2] = (x[mask2] - a_next) / (a_current - a_next)
                return result
        
        elif level_index == total_levels:
            # 公式(4-13): 最后级别
            a_prev = level_params[level_index - 2]
            
            if isinstance(x, (int, float)):
                if x >= a_prev:
                    return a_prev / x if x > 0 else 0.0
                elif a_current <= x < a_prev:
                    return 1.0
                else:
                    # 修正：当x非常小（接近0）时，级别4隶属度应该为1.0
                    # 而不是x / a_current，因为对于下界型特征，值越小越好
                    if x <= 0.0:
                        return 1.0
                    else:
                        return x / a_current if a_current > 0 else 0.0
            else:
                result = np.zeros_like(x)
                mask1 = x >= a_prev
                mask2 = (x >= a_current) & (x < a_prev)
                mask3 = x < a_current
                
                result[mask1] = np.where(x[mask1] > 0, a_prev / x[mask1], 0.0)
                result[mask2] = 1.0
                # 修正：当x非常小（接近0）时，级别4隶属度应该为1.0
                result[mask3] = np.where(x[mask3] <= 0.0, 1.0, 
                                        np.where(a_current > 0, x[mask3] / a_current, 0.0))
                return result
        
        else:
            # 公式(4-12): 中间级别
            a_prev = level_params[level_index - 2]
            a_next = level_params[level_index]
            
            if isinstance(x, (int, float)):
                if x >= a_prev:
                    return a_prev / x if x > 0 else 0.0
                elif a_current <= x < a_prev:
                    return 1.0
                elif a_next <= x < a_current:
                    return (x - a_next) / (a_current - a_next)
                else:
                    return 0.0
            else:
                result = np.zeros_like(x)
                mask1 = x >= a_prev
                mask2 = (x >= a_current) & (x < a_prev)
                mask3 = (x >= a_next) & (x < a_current)
                
                result[mask1] = np.where(x[mask1] > 0, a_prev / x[mask1], 0.0)
                result[mask2] = 1.0
                result[mask3] = (x[mask3] - a_next) / (a_current - a_next)
                return result
    
    @staticmethod
    def calculate_normalized_matrix(factors_data: Dict[str, List[float]], 
                                   factors_params: Dict[str, List[float]]) -> Dict[str, np.ndarray]:
        """
        计算规范化特征值级别矩阵
        
        根据因子数据和参数，计算每个评估对象的规范化特征值矩阵
        
        Args:
            factors_data: 因子数据字典，格式为{"f1": [值1, 值2, ...], "f2": [...]}
            factors_params: 因子参数字典，格式为{"f1": [a1, a2, a3, a4], "f2": [...]}
            
        Returns:
            规范化特征值矩阵字典，格式为{"T1": 矩阵, "T2": 矩阵, ...}
        """
        normalized_matrices = {}
        
        # 获取评估对象数量
        num_objects = len(list(factors_data.values())[0])
        
        for obj_idx in range(num_objects):
            obj_name = f"T{obj_idx + 1}"
            matrix = []
            
            for factor_name, factor_data in factors_data.items():
                if factor_name not in factors_params:
                    raise ValueError(f"因子{factor_name}的参数未定义")
                
                x_value = factor_data[obj_idx]
                level_params = factors_params[factor_name]
                total_levels = len(level_params)
                
                row = []
                for level in range(1, total_levels + 1):
                    membership_value = LowerBoundMembershipFunctions.lower_bound_level_membership(
                        x_value, level_params, level, total_levels
                    )
                    row.append(membership_value)
                
                matrix.append(row)
            
            normalized_matrices[obj_name] = np.array(matrix)
        
        return normalized_matrices
    
    @staticmethod
    def get_predefined_factors_params() -> Dict[str, List[float]]:
        """
        获取预定义的因子参数（基于参考内容中的示例）
        
        Returns:
            因子参数字典
        """
        return {
            "f1": [15, 10, 5, 2],    # 示例因子1
            "f2": [180, 90, 60, 30],  # 示例因子2
            "f3": [20, 10, 5, 2],    # 示例因子3
            "f4": [3, 2, 1, 0.5]     # 示例因子4
        }
    
    @staticmethod
    def get_sample_factors_data() -> Dict[str, List[float]]:
        """
        获取示例因子数据（基于参考内容中的示例）
        
        Returns:
            因子数据字典
        """
        return {
            "f1": [16.92, 0, 0, 2.0, 20.0, 0, 18.68],      # 示例因子1数据
            "f2": [160.17, 86.91, 0, 507.12, 618.24, 0, 823.42],  # 示例因子2数据
            "f3": [70.53, 0, 0, 1.0, 41.63, 9.0, 40.28],  # 示例因子3数据
            "f4": [3.5, 6.0, 5.5, 6.5, 8.5, 1.5, 3.5]     # 示例因子4数据
        }


def calculate_topsis_comprehensive_membership(normalized_matrices: Dict[str, np.ndarray], 
                                           weights: List[float] = None) -> Dict[str, Dict[str, float]]:
    """
    使用TOPSIS方法计算综合隶属度
    
    Args:
        normalized_matrices: 规范化特征值矩阵字典，格式为{"T1": 矩阵, "T2": 矩阵, ...}
        weights: 指标权重列表，默认为等权重
        
    Returns:
        综合隶属度字典，格式为{"T1": {"D+": [d1+, d2+, ...], "D-": [d1-, d2-, ...]}, ...}
    """
    comprehensive_membership = {}
    
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
    
    # 对每个评估对象计算TOPSIS综合隶属度
    for obj_name, matrix in normalized_matrices.items():
        # 初始化结果存储
        d_positive = []
        d_negative = []
        
        # 对每个级别计算距离
        for level_idx in range(num_levels):
            # 获取当前级别的所有因子隶属度值
            level_memberships = matrix[:, level_idx]
            
            # 定义最优解和最劣解（基于参考内容）
            # 最优解：所有因子隶属度都为1（最大业务风险）
            # 最劣解：所有因子隶属度都为0（最小业务风险）
            ideal_positive = np.ones(num_factors)
            ideal_negative = np.zeros(num_factors)
            
            # 计算与最优解的欧氏距离D+
            distance_positive = np.sqrt(np.sum(weights * (level_memberships - ideal_positive) ** 2))
            
            # 计算与最劣解的欧氏距离D-
            distance_negative = np.sqrt(np.sum(weights * (level_memberships - ideal_negative) ** 2))
            
            d_positive.append(distance_positive)
            d_negative.append(distance_negative)
        
        comprehensive_membership[obj_name] = {
            "D+": d_positive,
            "D-": d_negative
        }
    
    return comprehensive_membership


def test_lower_bound_membership_functions():
    """测试下界型隶属度函数"""
    mf = LowerBoundMembershipFunctions()
    
    print("=== 下界型级别隶属函数测试 ===")
    
    # 测试超量库存评价因子（f1）
    level_params_f1 = [15, 10, 5, 2]
    total_levels = 4
    
    print("超量库存评价因子（f1）测试 (参数=[15, 10, 5, 2]):")
    test_values = [0, 2, 5, 8, 10, 12, 15, 18, 20]
    
    for level in range(1, total_levels + 1):
        print(f"\n级别 {level}:")
        for val in test_values:
            result = mf.lower_bound_level_membership(val, level_params_f1, level, total_levels)
            print(f"  x={val}, μ={result:.3f}")
    
    print("\n=== 规范化评价因子级别特征值矩阵计算测试 ===")
    
    # 测试规范化矩阵计算
    evaluation_factors_data = mf.get_sample_factors_data()
    evaluation_factors_params = mf.get_predefined_factors_params()
    
    normalized_matrices = mf.calculate_normalized_matrix(evaluation_factors_data, evaluation_factors_params)
    
    for obj_name, matrix in normalized_matrices.items():
        print(f"\n{obj_name}的规范化特征值矩阵:")
        print("        e1    e2    e3    e4")
        for i, row in enumerate(matrix):
            print(f"f{i+1}    {row[0]:.3f}  {row[1]:.3f}  {row[2]:.3f}  {row[3]:.3f}")
    
    print("\n=== TOPSIS综合隶属度计算测试 ===")
    
    # 测试TOPSIS综合隶属度计算
    # 使用等权重
    weights = [0.25, 0.25, 0.25, 0.25]
    comprehensive_membership = calculate_topsis_comprehensive_membership(normalized_matrices, weights)
    
    print("TOPSIS综合隶属度矩阵:")
    print("业务对象  与最大业务风险的欧式距离        与最小业务风险的欧式距离")
    print("          e1     e2     e3     e4          e1     e2     e3     e4")
    for obj_name, distances in comprehensive_membership.items():
        d_positive = distances["D+"]
        d_negative = distances["D-"]
        print(f"{obj_name}      {d_positive[0]:.3f}  {d_positive[1]:.3f}  {d_positive[2]:.3f}  {d_positive[3]:.3f}        "
              f"{d_negative[0]:.3f}  {d_negative[1]:.3f}  {d_negative[2]:.3f}  {d_negative[3]:.3f}")
    
    print("\n下界型隶属度函数测试完成")


if __name__ == "__main__":
    test_lower_bound_membership_functions()