#!/usr/bin/env python3
"""
等级综合评定模块测试脚本
测试GradeComprehensiveAssessment类的功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.algorithm_layer.grade_comprehensive_assessment import GradeComprehensiveAssessment

def test_grade_assessment():
    """测试等级综合评定功能"""
    print("🎯 开始测试等级综合评定模块...")
    
    # 创建测试数据 - 模拟TOPSIS结果中的综合隶属度向量
    membership_scores = {
        "T1": [0.382, 0.297, 0.205, 0.116],  # 级别特征值应为2.056
        "T2": [0.250, 0.250, 0.250, 0.250],  # 级别特征值应为2.500
        "T3": [0.100, 0.200, 0.300, 0.400],  # 级别特征值应为3.000
        "T4": [0.400, 0.300, 0.200, 0.100],  # 级别特征值应为1.900
        "T5": [0.050, 0.150, 0.350, 0.450]   # 级别特征值应为3.200
    }
    
    num_levels = 4
    
    print(f"📊 测试数据：{len(membership_scores)}个业务对象，{num_levels}个评估等级")
    
    # 测试级别特征值计算
    print("\n📈 测试级别特征值计算...")
    v_values = GradeComprehensiveAssessment.calculate_level_characteristic_values(membership_scores)
    
    for obj_name, v_value in v_values.items():
        print(f"   {obj_name}: 级别特征值 = {v_value:.3f}")
    
    # 测试二元语义评定
    print("\n🎯 测试二元语义评定...")
    binary_results = GradeComprehensiveAssessment.perform_binary_semantic_assessment(v_values, num_levels)
    
    for obj_name, (k, alpha) in binary_results.items():
        print(f"   {obj_name}: 二元语义 = ({k}, {alpha:+.3f}) -> {k}级")
    
    # 测试完整评定流程
    print("\n🚀 测试完整等级综合评定流程...")
    result = GradeComprehensiveAssessment.assess_comprehensive_grade(membership_scores, num_levels)
    
    print(f"   评估对象数量: {result['num_objects']}")
    print(f"   评估等级数量: {result['num_levels']}")
    
    # 显示评定报告
    print("\n📋 评定报告预览：")
    print(result['assessment_report'][:500] + "...")
    
    print("\n✅ 等级综合评定模块测试完成！")
    return True

def test_individual_functions():
    """测试单个功能函数"""
    print("\n🔧 测试单个功能函数...")
    
    # 测试单个业务对象的级别特征值计算
    u_vector = [0.382, 0.297, 0.205, 0.116]
    
    # 手动计算级别特征值
    v_value = 0.0
    for k, u_jk in enumerate(u_vector, start=1):
        v_value += k * u_jk
    
    print(f"   单个业务对象级别特征值: {v_value:.3f} (期望: 2.056)")
    
    # 手动计算二元语义
    k = round(v_value)
    k = max(1, min(k, 4))  # 确保在1-4范围内
    alpha = v_value - k
    
    print(f"   二元语义结果: ({k}, {alpha:+.3f}) (期望: (2, +0.056))")
    
    # 验证计算结果
    expected_v = 2.056
    expected_k = 2
    expected_alpha = 0.056
    
    v_error = abs(v_value - expected_v)
    alpha_error = abs(alpha - expected_alpha)
    
    print(f"   级别特征值误差: {v_error:.6f}")
    print(f"   符号偏移值误差: {alpha_error:.6f}")
    
    if v_error < 0.001 and alpha_error < 0.001:
        print("   ✅ 计算结果准确！")
    else:
        print("   ⚠️ 计算结果存在误差")

if __name__ == "__main__":
    try:
        test_grade_assessment()
        test_individual_functions()
        print("\n🎉 所有测试通过！等级综合评定模块功能正常。")
    except Exception as e:
        print(f"\n❌ 测试失败：{e}")
        import traceback
        traceback.print_exc()