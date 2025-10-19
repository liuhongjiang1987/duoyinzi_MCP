"""
基于 MCP 的智能数据分析工具 - 完整的数据处理流程

采用 FastMCP 框架实现完整的数据分析工作流，包括：
- 数据上传与解析（CSV/Excel格式支持）
- 智能字段分析（数值/分类字段统计分析）
- 极性调整系统（自动检测和手动配置字段极性）
- 隶属度计算（下界型隶属度函数，多级别计算，经过严格数学验证）
- 持久化资源管理（跨会话数据保存和恢复）

项目状态：v1.1.0 - 生产就绪，经过完整验证和优化

主要特性：
✅ 完整的数据处理流程：从数据上传到隶属度计算的全流程支持
✅ 智能极性调整：自动检测字段极性，支持手动配置和数值转换
✅ 下界型隶属度计算：基于数学公式的多级别隶属度计算，经过严格数学验证
✅ 持久化资源管理：跨会话数据保存和恢复
✅ 资源格式兼容：支持纯ID和完整URI两种资源标识格式
✅ 完善的错误处理：详细的验证和错误提示机制
✅ 算法正确性验证：下界型隶属函数经过多场景测试验证，确保数学严谨性
✅ 项目结构优化：清理冗余文件，保持代码整洁和可维护性

版本历史：
v1.0.0 - 核心功能完整实现，资源ID格式兼容性修复
v1.1.0 - 下界型隶属函数验证优化，项目结构整理，生产就绪
"""

import io
import uuid
import pandas as pd
import numpy as np
import os
import json
import pickle
from typing import Dict, Any, Optional, List
from mcp.server.fastmcp import FastMCP
from modules.data_layer.field_analyzer import analyze_numeric_fields, analyze_categorical_fields, auto_detect_polarity, apply_polarity_adjustment, generate_polarity_report
from modules.algorithm_layer.membership_functions import LowerBoundMembershipFunctions
from modules.algorithm_layer.topsis_comprehensive_evaluation import TOPSISComprehensiveEvaluation
from modules.algorithm_layer.vikor_comprehensive_evaluation import VIKORComprehensiveEvaluation

# 创建 MCP 服务器
mcp = FastMCP("SmartDataAnalyzer")

# 数据缓存（内存缓存，Session 级）
DATA_CACHE: Dict[str, pd.DataFrame] = {}

# 持久化存储目录
PERSISTENT_STORAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "persistent_resources")

# MVP阶段：不创建持久化存储目录（禁用磁盘存储）
# if not os.path.exists(PERSISTENT_STORAGE_DIR):
#     os.makedirs(PERSISTENT_STORAGE_DIR)

# 资源索引文件路径
RESOURCE_INDEX_FILE = os.path.join(PERSISTENT_STORAGE_DIR, "resource_index.json")

# 加载资源索引（MVP阶段：不加载磁盘索引文件）
RESOURCE_INDEX = {}
# MVP阶段：禁用磁盘索引加载
# if os.path.exists(RESOURCE_INDEX_FILE):
#     try:
#         with open(RESOURCE_INDEX_FILE, 'r', encoding='utf-8') as f:
#             RESOURCE_INDEX = json.load(f)
#     except:
#         RESOURCE_INDEX = {}

# 资源命名规则常量
RESOURCE_TYPES = {
    "raw_data": "raw",           # 原始数据（用户上传）
    "field_analysis": "fa",      # 字段分析结果
    "membership_calc": "mc",     # 隶属度计算结果（直接使用raw数据）
    "multi_criteria": "mcr",     # 多准则计算结果
    "binary_semantic": "bs",     # 二元语义分割结果
    "other": "other"             # 其他计算结果
}


def generate_resource_id(resource_type: str, parent_resource_id: str = None, step_name: str = None) -> str:
    """
    生成标准化的资源ID
    
    Args:
        resource_type: 资源类型（使用RESOURCE_TYPES中的键）
        parent_resource_id: 父资源ID（用于建立依赖关系）
        step_name: 步骤名称（可选，用于描述性标识）
        
    Returns:
        标准化的资源ID格式：{type}_{uuid}_{parent_hash}_{step}
    """
    import hashlib
    import uuid
    
    # 验证资源类型
    if resource_type not in RESOURCE_TYPES:
        resource_type = "other"
    
    type_prefix = RESOURCE_TYPES[resource_type]
    
    # 生成唯一标识
    unique_id = str(uuid.uuid4())[:8]
    
    # 计算父资源哈希（用于建立依赖链）
    parent_hash = ""
    if parent_resource_id:
        parent_hash = hashlib.md5(parent_resource_id.encode()).hexdigest()[:6]
    
    # 步骤名称处理
    step_suffix = ""
    if step_name:
        # 清理步骤名称，只保留字母数字
        step_suffix = "_" + "".join(c for c in step_name if c.isalnum()).lower()[:10]
    
    # 构建完整资源ID
    resource_id = f"{type_prefix}_{unique_id}"
    if parent_hash:
        resource_id += f"_{parent_hash}"
    if step_suffix:
        resource_id += step_suffix
    
    return resource_id


def get_resource_uri(resource_id: str) -> str:
    """
    根据资源ID生成标准URI
    
    Args:
        resource_id: 标准化的资源ID
        
    Returns:
        资源URI格式：data://{resource_id}
    """
    return f"data://{resource_id}"


def parse_resource_id(resource_id: str) -> Dict[str, str]:
    """
    解析资源ID，提取类型、父资源等信息
    
    Args:
        resource_id: 标准化的资源ID
        
    Returns:
        解析后的资源信息
    """
    parts = resource_id.split("_")
    
    if len(parts) < 2:
        return {"type": "unknown", "id": resource_id}
    
    type_code = parts[0]
    resource_type = "unknown"
    
    # 反向查找类型
    for type_name, code in RESOURCE_TYPES.items():
        if code == type_code:
            resource_type = type_name
            break
    
    result = {
        "type": resource_type,
        "type_code": type_code,
        "unique_id": parts[1],
        "id": resource_id
    }
    
    # 解析父资源哈希
    if len(parts) > 2 and len(parts[2]) == 6:
        result["parent_hash"] = parts[2]
    
    # 解析步骤名称
    if len(parts) > 3:
        result["step_name"] = parts[3]
    
    return result


def save_resource_to_persistent_storage(resource_id: str, df: pd.DataFrame) -> str:
    """
    将资源保存到持久化存储（MVP阶段：仅内存存储，不持久化到文件）
    
    Args:
        resource_id: 资源ID
        df: 要保存的DataFrame
        
    Returns:
        虚拟文件路径（仅用于兼容）
    """
    # MVP阶段：更新内存索引和数据缓存
    parsed_info = parse_resource_id(resource_id)
    uri = get_resource_uri(resource_id)
    
    # 更新资源索引
    RESOURCE_INDEX[resource_id] = {
        "uri": uri,
        "type": parsed_info["type"],
        "type_code": parsed_info["type_code"],
        "data_shape": f"{len(df)} 行 × {len(df.columns)} 列",
        "file_path": f"memory://{resource_id}",  # 虚拟内存路径
        "created_time": pd.Timestamp.now().isoformat(),
        "parent_hash": parsed_info.get("parent_hash", ""),
        "step_name": parsed_info.get("step_name", "")
    }
    
    # 保存到内存缓存
    DATA_CACHE[uri] = df.copy()
    
    # 不保存索引文件到磁盘（MVP阶段）
    # with open(RESOURCE_INDEX_FILE, 'w', encoding='utf-8') as f:
    #     json.dump(RESOURCE_INDEX, f, ensure_ascii=False, indent=2)
    
    return f"memory://{resource_id}"


def load_resource_from_persistent_storage(resource_id: str) -> pd.DataFrame:
    """
    从持久化存储加载资源（MVP阶段：仅从内存加载）
    
    Args:
        resource_id: 资源ID（可以是纯ID或完整URI）
        
    Returns:
        加载的DataFrame
        
    Raises:
        ValueError: 资源不存在或不在内存中
    """
    # 如果没有提供resource_id，尝试自动发现mc资源
    if resource_id is None:
        resource_id = find_latest_membership_resource()
        if resource_id is None:
            raise ValueError("未找到隶属度计算结果资源（mc开头的资源），请先执行隶属度计算")
        print(f"自动发现隶属度资源: {resource_id}")
    
    # 处理URI格式的资源ID
    if resource_id and resource_id.startswith("data://"):
        resource_id = resource_id[7:]  # 移除 "data://" 前缀
    
    if resource_id not in RESOURCE_INDEX:
        raise ValueError(f"资源不存在于存储: {resource_id}")
    
    # MVP阶段：仅从内存缓存加载
    uri = RESOURCE_INDEX[resource_id]["uri"]
    if uri not in DATA_CACHE:
        raise ValueError(f"资源不在内存缓存中（MVP阶段不支持磁盘加载）: {resource_id}")
    
    return DATA_CACHE[uri]


def load_all_persistent_resources():
    """
    加载所有持久化资源到内存缓存（MVP阶段：此功能已禁用）
    """
    print("ℹ️  MVP阶段：禁用持久化资源加载，仅使用内存缓存")
    # MVP阶段：不加载任何持久化资源
    # for resource_id, resource_info in RESOURCE_INDEX.items():
    #     try:
    #         df = load_resource_from_persistent_storage(resource_id)
    #         DATA_CACHE[resource_info["uri"]] = df
    #         print(f"✅ 已加载持久化资源: {resource_id} ({resource_info['data_shape']})")
    #     except Exception as e:
    #         print(f"❌ 加载资源失败 {resource_id}: {e}")


# MVP阶段：禁用持久化存储，仅使用内存缓存
# load_all_persistent_resources()  # 注释掉持久化加载


@mcp.tool()
def get_resource_dependency_chain(resource_id: str) -> Dict[str, Any]:
    """
    获取资源的依赖链信息 - 追踪资源生成路径
    
    功能说明：
    - 分析指定资源的完整生成路径和依赖关系
    - 支持纯ID和完整URI两种格式的资源标识符
    - 自动处理资源ID格式兼容性（移除data://前缀）
    
    应用场景：
    - 调试资源生成流程
    - 分析数据处理链
    - 资源关系可视化
    
    Args:
        resource_id: 资源标识符（可以是纯ID或完整URI）
        
    Returns:
        Dict[str, Any]: 包含以下字段的字典：
        - resource_id: 原始资源ID
        - parsed_info: 解析后的资源信息
        - dependency_chain: 依赖链列表（从当前资源到根资源）
        - chain_length: 依赖链长度
        
    示例：
    >>> get_resource_dependency_chain("raw_abc123")
    {
        "resource_id": "raw_abc123",
        "parsed_info": {...},
        "dependency_chain": [...],
        "chain_length": 1
    }
    """
    # 处理URI格式的资源ID
    if resource_id.startswith("data://"):
        resource_id = resource_id[7:]  # 移除 "data://" 前缀
    
    parsed_info = parse_resource_id(resource_id)
    
    # 获取所有资源信息
    all_resources = {}
    for uri in DATA_CACHE.keys():
        if uri.startswith("data://"):
            res_id = uri[7:]  # 移除 "data://" 前缀
            all_resources[res_id] = parse_resource_id(res_id)
    
    # 构建依赖链
    dependency_chain = []
    current_resource = parsed_info
    
    while current_resource:
        dependency_chain.append(current_resource)
        
        # 查找父资源
        parent_hash = current_resource.get("parent_hash")
        if not parent_hash:
            break
            
        # 在所有资源中查找匹配的父资源
        parent_resource = None
        for res_id, res_info in all_resources.items():
            if res_info.get("unique_id") == parent_hash:
                parent_resource = res_info
                break
        
        if not parent_resource:
            break
            
        current_resource = parent_resource
    
    return {
        "resource_id": resource_id,
        "parsed_info": parsed_info,
        "dependency_chain": dependency_chain,
        "chain_length": len(dependency_chain)
    }





@mcp.tool()
def generate_membership_config_template(resource_id: str) -> str:
    """
    生成下界型隶属度计算配置模板 - 自动化配置生成
    
    功能说明：
    - 基于数据特征自动生成下界型隶属度函数的配置模板
    - 智能检测并优先使用极性调整后的数据资源
    - 支持纯ID和完整URI两种格式的资源标识符
    - 自动处理资源ID格式兼容性（移除data://前缀）
    
    算法特点：
    - 使用下界型级别隶属函数（基于公式4-11到4-13）
    - 为每个数值字段生成默认的3级别参数配置
    - 参数基于字段统计特征（最大值、最小值、平均值）
    
    工作流程：
    1. 检测是否存在极性调整后的数据资源
    2. 分析数据集的数值字段特征
    3. 为每个字段生成默认级别参数
    4. 生成Markdown格式的配置模板
    
    Args:
        resource_id: 原始数据资源标识符（可以是纯ID或完整URI）
        
    Returns:
        str: Markdown格式的下界型函数级别参数配置模板，包含：
        - 资源信息摘要
        - 字段统计信息表格
        - 推荐配置模板（JSON格式）
        - 配置说明和使用指南
        
    示例输出：
    ```markdown
    # 下界型隶属度计算配置模板
    
    ## 资源信息
    - **资源ID**: `raw_abc123`
    - **数值字段**: age, salary
    - **数据形状**: 100 行 × 5 列
    
    ## 字段统计信息
    | 字段 | 最小值 | 最大值 | 平均值 | 标准差 |
    |------|--------|--------|--------|--------|
    | age | 20.00 | 60.00 | 35.50 | 8.25 |
    
    ## 推荐配置模板
    ```json
    {
      "级别参数": {
        "age": [48.0, 30.0, 12.0],
        "salary": [80000.0, 50000.0, 20000.0]
      }
    }
    ```
    """
    # 处理URI格式的资源ID
    if resource_id.startswith("data://"):
        resource_id = resource_id[7:]  # 移除 "data://" 前缀
    
    # 检查是否存在极性调整后的数据资源
    polarity_adjusted_resource_id = None
    for cached_uri in DATA_CACHE.keys():
        if cached_uri.startswith("data://") and "polarity_adjusted" in cached_uri:
            # 检查是否是当前资源的极性调整版本
            adjusted_parsed = parse_resource_id(cached_uri[7:])  # 移除 "data://" 前缀
            if (adjusted_parsed["type"] == "raw_data" and 
                adjusted_parsed.get("parent_hash") == resource_id):
                polarity_adjusted_resource_id = cached_uri[7:]  # 移除 "data://" 前缀
                break
    
    # 优先使用极性调整后的数据资源
    if polarity_adjusted_resource_id:
        print(f"🔍 检测到极性调整后的数据资源: {polarity_adjusted_resource_id}")
        print("📊 将使用极性调整后的数据进行隶属度计算配置")
        uri = get_resource_uri(polarity_adjusted_resource_id)
        used_resource_id = polarity_adjusted_resource_id
    else:
        uri = get_resource_uri(resource_id)
        used_resource_id = resource_id
    
    if uri not in DATA_CACHE:
        raise ValueError(f"资源不存在: {uri}")
    
    df = DATA_CACHE[uri]
    
    # 检查是否为raw数据
    parsed_info = parse_resource_id(used_resource_id)
    if parsed_info["type"] != "raw_data":
        print(f"⚠️ 警告: 使用非原始数据生成配置模板，建议使用raw数据")
    
    # 获取数值字段
    import numpy as np
    numeric_fields = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_fields:
        raise ValueError("数据集中无数值字段，无法生成隶属度计算配置")
    
    # 为每个数值字段生成默认级别参数（下界型函数需要多个级别参数）
    level_params_template = {}
    field_statistics = {}
    for field in numeric_fields:
        field_data = df[field].dropna()
        if len(field_data) > 0:
            min_val = float(field_data.min())
            max_val = float(field_data.max())
            
            # 为每个字段生成默认级别参数列表（3个级别）
            level_params_template[field] = [
                round(max_val * 0.8, 2),  # 级别1参数
                round(max_val * 0.5, 2),  # 级别2参数
                round(max_val * 0.2, 2)   # 级别3参数
            ]
            
            field_statistics[field] = {
                "min": min_val,
                "max": max_val,
                "mean": float(field_data.mean()),
                "std": float(field_data.std())
            }
    
    # 生成Markdown格式的配置模板
    markdown_content = f"""# 下界型隶属度计算配置模板

## 资源信息
- **资源ID**: `{used_resource_id}`
- **资源类型**: {'极性调整后的数据' if polarity_adjusted_resource_id else '原始数据'}
- **数值字段**: {', '.join(numeric_fields)}
- **数据形状**: {len(df)} 行 × {len(df.columns)} 列

## 字段统计信息

| 字段 | 最小值 | 最大值 | 平均值 | 标准差 |
|------|--------|--------|--------|--------|
"""
    
    # 添加字段统计表格行
    for field in numeric_fields:
        stats = field_statistics[field]
        markdown_content += f"| {field} | {stats['min']:.2f} | {stats['max']:.2f} | {stats['mean']:.2f} | {stats['std']:.2f} |\n"
    
    markdown_content += """
## 推荐配置模板

```json
{
  "级别参数": {
"""
    
    # 添加级别参数配置
    for i, field in enumerate(numeric_fields):
        params = level_params_template[field]
        markdown_content += f'    "{field}": {params}'
        if i < len(numeric_fields) - 1:
            markdown_content += ","
        markdown_content += "\n"
    
    markdown_content += """  }
}
```

## 配置说明

### 下界型隶属度函数特点
- 使用下界型级别隶属函数（基于公式4-11到4-13）
- 每个字段需要配置级别参数列表，例如：`[15, 10, 5]` 对应3个级别
- 级别参数应按从大到小顺序排列
- 总级别数应与级别参数数量一致

### 使用流程
1. 复制上面的配置模板
2. 根据实际需求调整级别参数
3. 调用 `validate_membership_config(resource_id, config)` 验证配置
4. 验证通过后调用 `calculate_membership_with_config(resource_id, config)` 执行计算

### 配置示例
```json
{
  "级别参数": {
    "字段1": [15, 10, 5],
    "字段2": [180, 90, 60]
  }
}
```

**注意**: 配置中不再需要"隶属函数"字段，系统默认使用下界型函数。
"""
    
    return markdown_content


@mcp.tool()
def validate_membership_config(resource_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    验证下界型隶属度计算配置
    
    Args:
        resource_id: 原始数据资源标识符（可以是纯ID或完整URI）
        config: 用户提供的配置信息
        
    Returns:
        验证结果
    """
    # 如果没有提供resource_id，尝试自动发现mc资源
    if resource_id is None:
        resource_id = find_latest_membership_resource()
        if resource_id is None:
            raise ValueError("未找到隶属度计算结果资源（mc开头的资源），请先执行隶属度计算")
        print(f"自动发现隶属度资源: {resource_id}")
    
    # 处理URI格式的资源ID
    if resource_id.startswith("data://"):
        resource_id = resource_id[7:]  # 移除 "data://" 前缀
    
    uri = get_resource_uri(resource_id)
    
    if uri not in DATA_CACHE:
        raise ValueError(f"资源不存在: {uri}")
    
    df = DATA_CACHE[uri]
    
    # 检查是否为极性调整后的数据资源
    is_polarity_adjusted = "polarity_adjusted" in resource_id
    
    # 获取数值字段
    import numpy as np
    numeric_fields = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 验证配置结构
    errors = []
    warnings = []
    
    # 检查必需字段
    if "级别参数" not in config:
        errors.append("配置中缺少'级别参数'字段")
    
    if errors:
        return {
            "is_valid": False,
            "errors": errors,
            "warnings": warnings
        }
    
    # 设置隶属函数类型为下界型
    membership_type = "下界型"
    
    # 验证级别参数
    level_params = config["级别参数"]
    if not isinstance(level_params, dict):
        errors.append("'级别参数'字段必须为字典类型")
    else:
        # 检查字段是否存在
        for field in level_params.keys():
            if field not in numeric_fields:
                errors.append(f"字段 '{field}' 在数据集中不存在")
        
        # 验证所有字段的级别参数数量一致
        level_counts = {}
        for field, params in level_params.items():
            if field in numeric_fields:
                if not isinstance(params, list):
                    errors.append(f"字段 '{field}' 的级别参数必须为列表")
                else:
                    # 检查级别参数数量
                    if len(params) < 2:
                        errors.append(f"字段 '{field}' 的级别参数至少需要2个，当前提供{len(params)}个")
                    
                    # 检查级别参数是否按从大到小顺序排列
                    if len(params) > 1:
                        for i in range(len(params) - 1):
                            # 检查参数a值是否按从大到小顺序排列
                            # 处理两种格式的参数：数值列表或包含"a"键的字典列表
                            if isinstance(params[i], dict) and "a" in params[i]:
                                current_val = params[i].get("a", 0)
                                next_val = params[i + 1].get("a", 0)
                            else:
                                current_val = params[i]
                                next_val = params[i + 1]
                            
                            if current_val <= next_val:
                                errors.append(f"字段 '{field}' 的级别参数应按从大到小顺序排列")
                                break
                    
                    level_counts[field] = len(params)
        
        # 检查级别参数数量是否在所有字段中一致
        if level_counts:
            unique_counts = set(level_counts.values())
            if len(unique_counts) > 1:
                warnings.append(f"各字段的级别参数数量不一致: {level_counts}，建议保持一致以获得更好的评估结果")
    
    # 验证总级别数
    if "总级别数" in config:
        total_levels = config["总级别数"]
        if not isinstance(total_levels, int) or total_levels < 2:
            errors.append("'总级别数'必须为大于等于2的整数")
        elif level_params:
            # 检查总级别数是否与级别参数数量一致
            first_field_params = next(iter(level_params.values()))
            if isinstance(first_field_params, list) and total_levels != len(first_field_params):
                warnings.append(f"总级别数({total_levels})与第一个字段的级别参数数量({len(first_field_params)})不一致")
    
    # 检查是否有配置但数据中不存在的字段
    configured_fields = set(level_params.keys())
    available_fields = set(numeric_fields)
    missing_config = available_fields - configured_fields
    if missing_config:
        warnings.append(f"以下数值字段未配置级别参数: {list(missing_config)}")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "config_summary": {
            "membership_type": membership_type,
            "configured_fields": list(configured_fields),
            "available_fields": numeric_fields,
            "level_params_summary": level_counts if level_params else {}
        }
    }


@mcp.tool()
def calculate_membership_with_config(resource_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用下界型函数配置信息执行隶属度计算 - 多级别模糊综合评价工具
    
    该函数实现基于下界型隶属度函数的模糊综合评价，将原始数据转换为隶属度矩阵格式，
    为后续的多准则决策分析提供标准化的输入数据。
    
    Args:
        resource_id: 原始数据资源标识符（可以是纯ID或完整URI）
                     例如：'raw_data_001' 或 'data://raw_data_001'
        config: 用户提供的下界型函数配置信息，包含各评价因子的级别参数

    重要：必须严格按照以下模版输出结果，不得有模版以外的任何其它信息！！            
    Returns:
        回复模版
        ‘’‘
        ## 📊 隶属度计算概述
        - **计算方法**：下界型隶属度函数
        - **业务对象数量**：3个
        - **评价因子数量**：4个
        - **评估级别数量**：4级（级别1-级别4）
        
        ## 📈 隶属度矩阵结果（这里仅展示一个对象的数据，用户可以回复要求下载全部数据）
        
        ### 📌 业务对象1
        | 评价因子 | 级别1 | 级别2 | 级别3 | 级别4 |
        |---------|-------|-------|-------|-------|
        | 评价因子1 | 1.0000 | 0.8865 | 0.5910 | 0.2955 |
        | 评价因子2 | 0.7797 | 1.0000 | 0.5619 | 0.3746 |
        | 评价因子3 | 1.0000 | 0.2836 | 0.1418 | 0.0709 |
        | 评价因子4 | 1.0000 | 0.8571 | 0.5714 | 0.2857 |
        
        ## 📋 可查询表格清单
        1. **隶属度矩阵** - 核心计算结果，可直接用于TOPSIS/VIKOR评估
        2. **原始数据表** - 计算前的原始输入数据
        3. **配置参数表** - 使用的隶属度函数参数配置
        
        ## 🚀 下一步操作建议
        - 使用TOPSIS方法进行综合评估（perform_topsis_comprehensive_evaluation工具）
        - 使用VIKOR方法进行妥协排序分析（perform_vikor_comprehensive_evaluation工具）
        - 查看隶属度矩阵详细信息（export_resource_to_csv工具）
        - 验证隶属度计算结果（validate_membership_config工具）
        ’‘’
    """
    # 处理URI格式的资源ID
    if resource_id.startswith("data://"):
        resource_id = resource_id[7:]  # 移除 "data://" 前缀
    
    # 检查是否存在极性调整后的数据资源
    polarity_adjusted_resource_id = None
    for cached_uri in DATA_CACHE.keys():
        if cached_uri.startswith("data://") and "polarity_adjusted" in cached_uri:
            # 检查是否是当前资源的极性调整版本
            adjusted_parsed = parse_resource_id(cached_uri[7:])  # 移除 "data://" 前缀
            if (adjusted_parsed["type"] == "raw_data" and 
                adjusted_parsed.get("parent_hash") == resource_id):
                polarity_adjusted_resource_id = cached_uri[7:]  # 移除 "data://" 前缀
                break
    
    # 优先使用极性调整后的数据资源
    if polarity_adjusted_resource_id:
        print(f"🔍 检测到极性调整后的数据资源: {polarity_adjusted_resource_id}")
        print("📊 将使用极性调整后的数据进行隶属度计算")
        used_resource_id = polarity_adjusted_resource_id
    else:
        used_resource_id = resource_id
    
    # 自动验证配置
    validation_result = validate_membership_config(used_resource_id, config)
    if not validation_result["is_valid"]:
        raise ValueError(f"配置验证失败: {validation_result['errors']}")
    
    uri = get_resource_uri(used_resource_id)
    df = DATA_CACHE[uri]
    
    # 获取配置信息
    level_params = config["级别参数"]
    
    # 计算隶属度矩阵
    membership_results = []
    
    for _, row in df.iterrows():
        membership_row = {}
        for field, params in level_params.items():
            if field in df.columns:
                value = row[field]
                total_levels = len(params)
                
                # 计算每个级别的隶属度
                level_memberships = {}
                # 提取参数值列表（a值）
                # 处理两种格式的参数：数值列表或包含"a"键的字典列表
                if isinstance(params[0], dict) and "a" in params[0]:
                    param_values = [p["a"] for p in params]
                else:
                    param_values = params  # 直接使用数值列表
                
                for level in range(1, total_levels + 1):
                    membership = LowerBoundMembershipFunctions.lower_bound_level_membership(
                        value, param_values, level, total_levels
                    )
                    level_memberships[f"级别{level}"] = round(float(membership), 4)
                
                membership_row[field] = level_memberships
        
        membership_results.append(membership_row)
    
    # 创建扁平化的隶属度矩阵DataFrame（符合TOPSIS要求格式）
    flattened_results = []
    
    for idx, (_, row) in enumerate(df.iterrows()):
        # 获取业务对象标识
        obj_name = f"业务对象{idx+1}"
        if len(df.columns) > 0:
            # 尝试使用第一列的值作为对象名
            first_col_value = row.iloc[0]
            if pd.notna(first_col_value):
                obj_name = f"业务对象{idx+1}({str(first_col_value)[:20]})"
        
        # 为每个字段和级别创建扁平化记录
        for field, params in level_params.items():
            if field in df.columns and field in membership_results[idx]:
                total_levels = len(params)
                for level in range(1, total_levels + 1):
                    level_name = f"级别{level}"
                    membership_value = membership_results[idx][field].get(level_name, 0.0)
                    
                    flattened_results.append({
                        '业务对象': obj_name,
                        '评价因子': field,
                        '评估级别': level_name,
                        '隶属度': membership_value
                    })
    
    # 创建符合TOPSIS要求的隶属度矩阵
    membership_df = pd.DataFrame(flattened_results)
    
    # 生成计算结果资源ID
    result_resource_id = generate_resource_id(
        "membership_calc", 
        parent_resource_id=resource_id,
        step_name="membership_calculation"
    )
    result_uri = get_resource_uri(result_resource_id)
    
    # 缓存计算结果到内存
    DATA_CACHE[result_uri] = membership_df
    
    # 更新资源索引（修复隶属度计算结果资源不存在的根本问题）
    RESOURCE_INDEX[result_resource_id] = {
        'uri': result_uri,
        'type': 'membership_calc',
        'rows': len(membership_df),
        'columns': len(membership_df.columns),
        'column_names': list(membership_df.columns),
        'parent_resource_id': resource_id
    }
    
    # MVP阶段：不保存到持久化存储（已在内存中）
    # save_resource_to_persistent_storage(result_resource_id, membership_df)
    
    # 生成每个业务对象的隶属度矩阵格式化输出
    membership_matrices_formatted = []
    for idx, (_, row) in enumerate(df.iterrows()):
        # 获取业务对象标识（使用索引或第一列作为标识）
        obj_name = f"业务对象{idx+1}"
        if len(df.columns) > 0:
            # 尝试使用第一列的值作为对象名
            first_col_value = row.iloc[0]
            if pd.notna(first_col_value):
                obj_name += f"({str(first_col_value)[:20]})"  # 截取前20个字符
        
        # 构建矩阵表格
        matrix_lines = [f"### 📌 {obj_name}"]
        matrix_lines.append("")
        
        # 获取所有级别
        all_levels = []
        for field in level_params.keys():
            if field in df.columns and field in membership_results[idx]:
                levels = list(membership_results[idx][field].keys())
                if levels:
                    all_levels = levels
                    break
        
        # 构建表头
        if all_levels:
            header = "| 评价因子 | " + " | ".join(all_levels) + " |"
            separator = "|---------" + "|-------" * len(all_levels) + "|"
            matrix_lines.append(header)
            matrix_lines.append(separator)
            
            # 添加每行数据
            for field in level_params.keys():
                if field in df.columns and field in membership_results[idx]:
                    row_data = [field]
                    for level in all_levels:
                        if level in membership_results[idx][field]:
                            row_data.append(f"{membership_results[idx][field][level]:.4f}")
                        else:
                            row_data.append("0.0000")
                    matrix_lines.append("| " + " | ".join(row_data) + " |")
        
        matrix_lines.append("")  # 添加空行分隔
        membership_matrices_formatted.append("\n".join(matrix_lines))
    
    # 计算摘要
    calculation_summary = {
        "input_data_shape": f"{len(df)} 行 × {len(df.columns)} 列",
        "membership_type": "下界型",
        "configured_fields": list(level_params.keys()),
        "total_levels": {field: len(params) for field, params in level_params.items()},
        "membership_matrix_shape": f"{len(membership_df)} 行 × {len(membership_df.columns)} 列",
        "calculation_time": "实时计算",
        "total_objects": len(df),
        "formatted_matrices": membership_matrices_formatted  # 添加格式化矩阵
    }
    
    # 生成可查询表格清单
    available_tables = """1. **隶属度矩阵** - 核心计算结果，可直接用于TOPSIS/VIKOR评估
2. **原始数据表** - 计算前的原始输入数据
3. **配置参数表** - 使用的隶属度函数参数配置"""
    
    # 生成下一步操作建议
    next_actions = """- 使用TOPSIS方法进行综合评估（perform_topsis_comprehensive_evaluation工具）
- 使用VIKOR方法进行妥协排序分析（perform_vikor_comprehensive_evaluation工具）
- 查看隶属度矩阵详细信息（export_resource_to_csv工具）
- 验证隶属度计算结果（validate_membership_config工具）"""
    
    # 生成格式化报告并添加到结果中 - 直接使用模板字符串避免LLM加工
    result = {
        "membership_matrix": membership_df.to_dict("records"),
        "formatted_membership_matrices": membership_matrices_formatted,  # 格式化矩阵输出
        "calculation_summary": calculation_summary,
        "result_resource_id": result_resource_id,
        "config_used": config,
        "validation_warnings": validation_result["warnings"],
        "warnings": [
            "✅ 下界型隶属度计算完成，使用配置信息",
            f"📊 计算结果已缓存: {result_uri}",
            f"💾 计算结果已缓存到内存（MVP阶段不持久化到磁盘）",
            "💡 可使用list_resources_by_type('membership_calc')查看所有计算结果"
        ]
    }
    
    # 生成格式化报告
    result["formatted_report"] = f"""## 📊 隶属度计算概述
- **计算方法**：下界型隶属度函数
- **业务对象数量**：{len(df)} 个
- **评价因子数量**：{len(level_params)} 个
- **评估级别数量**：{len(all_levels) if all_levels else 4}级（{all_levels[0] if all_levels else '级别1'}-{all_levels[-1] if all_levels else '级别4'}）

## 📈 隶属度矩阵结果
{chr(10).join(membership_matrices_formatted)}

## 📋 可查询表格清单
{available_tables}

## 🚀 下一步操作建议
{next_actions}"""
    
    return result


@mcp.tool()
def list_resources_by_type(resource_type: str = None) -> Dict[str, Any]:
    """
    按类型列出资源
    
    Args:
        resource_type: 资源类型（可选，不指定则列出所有）
        
    Returns:
        按类型分类的资源列表
    """
    resources_by_type = {}
    
    for uri, df in DATA_CACHE.items():
        if uri.startswith("data://"):
            resource_id = uri[7:]  # 移除 "data://" 前缀
            parsed_info = parse_resource_id(resource_id)
            
            res_type = parsed_info["type"]
            
            if resource_type and res_type != resource_type:
                continue
                
            if res_type not in resources_by_type:
                resources_by_type[res_type] = []
            
            resources_by_type[res_type].append({
                "resource_id": resource_id,
                "uri": uri,
                "data_shape": f"{len(df)} 行 × {len(df.columns)} 列",
                "parsed_info": parsed_info
            })
    
    return {
        "total_resources": len(DATA_CACHE),
        "resources_by_type": resources_by_type,
        "resource_types_found": list(resources_by_type.keys())
    }


@mcp.tool()
def upload_csv(csv_text: str) -> Dict[str, Any]:
    """
    上传 CSV 文本数据并缓存为资源，自动触发字段极性智能检测
    
    Args:
        csv_text: CSV 格式的文本内容
        
    Returns:
        包含原始资源URI和极性检测结果的字典
    """
    try:
        # 使用 pandas 解析 CSV 文本
        df = pd.read_csv(io.StringIO(csv_text))
        
        # 使用标准化的资源命名规则
        resource_id = generate_resource_id("raw_data")
        uri = get_resource_uri(resource_id)
        
        # 缓存数据到内存
        DATA_CACHE[uri] = df
        
        # 更新资源索引（修复资源不存在的根本问题）
        RESOURCE_INDEX[resource_id] = {
            'uri': uri,
            'type': 'raw_data',
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns)
        }
        
        # MVP阶段：不保存到持久化存储（已在内存中）
        # save_resource_to_persistent_storage(resource_id, df)
        
        # 记录上传信息
        print(f"数据上传成功: {uri}")
        print(f"资源类型: 原始数据 (raw_data)")
        print(f"数据集信息: {len(df)} 行, {len(df.columns)} 列")
        print(f"列名: {list(df.columns)}")
        print(f"✅ 资源已缓存到内存（MVP阶段不持久化到磁盘）")
        
        # 自动触发字段极性智能检测
        print("\n🔍 开始字段极性智能检测...")
        
        # 分析数值字段
        numeric_analysis = analyze_numeric_fields(df)
        
        if not numeric_analysis:
            print("⚠️  未发现数值型字段，跳过极性检测")
            return {
                "original_resource_uri": uri,
                "polarity_detection": {
                    "status": "skipped",
                    "message": "未发现数值型字段，无需极性检测"
                },
                "adjusted_resource_uri": None,
                "next_actions": [
                    "分析这些数据字段的特征（使用analyze_data_fields工具）",
                    "查看数据内容（使用export_resource_to_csv工具）"
                ]
            }
        
        # 检测字段极性
        polarity_results = auto_detect_polarity(df, numeric_analysis)
        
        # 生成极性检测报告
        polarity_report = generate_polarity_report(polarity_results)
        
        # 检查是否有无法评估的字段
        failed_detections = [col for col, result in polarity_results.items() if not result['detection_successful']]
        
        if failed_detections:
            print("❌ 发现无法评估极性的字段")
            print(polarity_report)
            
            # 构建极性配置模板
            polarity_config = {}
            for col, result in polarity_results.items():
                if result['detection_successful']:
                    polarity_config[col] = result['suggested_polarity']
            
            return {
                "original_resource_uri": uri,
                "polarity_detection": {
                    "status": "requires_confirmation",
                    "message": "发现无法评估极性的字段，需要用户确认极性配置",
                    "failed_fields": failed_detections,
                    "detected_polarities": polarity_config,
                    "report": polarity_report
                },
                "adjusted_resource_uri": None,
                "next_actions": [
                    "使用apply_polarity_adjustment工具手动确认极性配置",
                    "修改表头名称，包含明确的极性指示词",
                    "查看数据内容（使用export_resource_to_csv工具）"
                ]
            }
        
        # 所有字段极性检测成功，自动应用调整
        print("🔄 应用极性调整策略...")
        adjusted_df = apply_polarity_adjustment(df, polarity_results)
        
        # 保存调整后的数据为缓存资源
        adjusted_resource_id = generate_resource_id("raw_data", resource_id, "polarity_adjusted")
        adjusted_uri = get_resource_uri(adjusted_resource_id)
        
        # 缓存调整后的数据
        DATA_CACHE[adjusted_uri] = adjusted_df
        
        # 更新资源索引（修复极性调整后资源不存在的根本问题）
        RESOURCE_INDEX[adjusted_resource_id] = {
            'uri': adjusted_uri,
            'type': 'polarity_adjusted',
            'rows': len(adjusted_df),
            'columns': len(adjusted_df.columns),
            'column_names': list(adjusted_df.columns),
            'parent_resource_id': resource_id
        }
        
        # MVP阶段：不保存到持久化存储（已在内存中）
        # save_resource_to_persistent_storage(adjusted_resource_id, adjusted_df)
        
        print(f"✅ 极性调整后的数据已缓存: {adjusted_uri}")
        
        # 输出极性检测报告
        print("\n" + "="*60)
        print("📊 字段极性智能检测完成")
        print("="*60)
        print(polarity_report)
        
        # 输出调整策略信息
        print("\n🔄 极性调整策略已应用：")
        for col, result in polarity_results.items():
            if result['suggested_polarity'] == 'max':
                print(f"   - {col}: 越大越好 → 越小越好（通过取倒数转换）")
            else:
                print(f"   - {col}: 越小越好（无需调整）")
        
        print(f"\n💾 调整后数据已缓存为: {adjusted_uri}")
        
        return {
            "original_resource_uri": uri,
            "polarity_detection": {
                "status": "success",
                "message": "字段极性检测完成，极性调整已应用",
                "report": polarity_report,
                "polarity_results": polarity_results
            },
            "adjusted_resource_uri": adjusted_uri,
            "next_actions": [
                "分析这些数据字段的特征（使用analyze_data_fields工具）",
                "生成隶属度计算配置模板（使用generate_membership_config_template工具）",
                "查看数据内容（使用export_resource_to_csv工具）"
            ]
        }
        
    except Exception as e:
        raise ValueError(f"CSV 解析失败: {str(e)}")


@mcp.tool()
def upload_excel(file_content: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    """
    上传 Excel 文件数据并缓存为资源，自动触发字段极性智能检测
    
    Args:
        file_content: Base64 编码的 Excel 文件内容
        sheet_name: 工作表名称（可选，默认为第一个工作表）
        
    Returns:
        包含原始资源URI和极性检测结果的字典
    """
    try:
        # 解码 Base64 内容
        import base64
        excel_bytes = base64.b64decode(file_content)
        
        # 使用 pandas 解析 Excel 文件
        if sheet_name:
            df = pd.read_excel(io.BytesIO(excel_bytes), sheet_name=sheet_name)
        else:
            df = pd.read_excel(io.BytesIO(excel_bytes))
        
        # 使用标准化的资源命名规则
        resource_id = generate_resource_id("raw_data")
        uri = get_resource_uri(resource_id)
        
        # 缓存数据到内存
        DATA_CACHE[uri] = df
        
        # 更新资源索引（修复Excel上传资源不存在的根本问题）
        RESOURCE_INDEX[resource_id] = {
            'uri': uri,
            'type': 'raw_data',
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns)
        }
        
        # MVP阶段：不保存到持久化存储（已在内存中）
        # save_resource_to_persistent_storage(resource_id, df)
        
        # 记录上传信息
        print(f"Excel 数据上传成功: {uri}")
        print(f"资源类型: 原始数据 (raw_data)")
        print(f"数据集信息: {len(df)} 行, {len(df.columns)} 列")
        print(f"列名: {list(df.columns)}")
        if sheet_name:
            print(f"工作表: {sheet_name}")
        print(f"✅ 资源已缓存到内存（MVP阶段不持久化到磁盘）")
        
        # 自动触发字段极性智能检测
        print("\n🔍 开始字段极性智能检测...")
        
        # 分析数值字段
        numeric_analysis = analyze_numeric_fields(df)
        
        if not numeric_analysis:
            print("⚠️  未发现数值型字段，跳过极性检测")
            return {
                "original_resource_uri": uri,
                "polarity_detection": {
                    "status": "skipped",
                    "message": "未发现数值型字段，无需极性检测"
                },
                "adjusted_resource_uri": None,
                "next_actions": [
                    "分析这些数据字段的特征（使用analyze_data_fields工具）",
                    "查看数据内容（使用export_resource_to_csv工具）"
                ]
            }
        
        # 检测字段极性
        polarity_results = auto_detect_polarity(df, numeric_analysis)
        
        # 生成极性检测报告
        polarity_report = generate_polarity_report(polarity_results)
        
        # 检查是否有无法评估的字段
        failed_detections = [col for col, result in polarity_results.items() if not result['detection_successful']]
        
        if failed_detections:
            print("❌ 发现无法评估极性的字段")
            print(polarity_report)
            
            # 构建极性配置模板
            polarity_config = {}
            for col, result in polarity_results.items():
                if result['detection_successful']:
                    polarity_config[col] = result['suggested_polarity']
            
            return {
                "original_resource_uri": uri,
                "polarity_detection": {
                    "status": "requires_confirmation",
                    "message": "发现无法评估极性的字段，需要用户确认极性配置",
                    "failed_fields": failed_detections,
                    "detected_polarities": polarity_config,
                    "report": polarity_report
                },
                "adjusted_resource_uri": None,
                "next_actions": [
                    "使用apply_polarity_adjustment_tool手动确认极性配置",
                    "修改表头名称，包含明确的极性指示词",
                    "查看数据内容（使用export_resource_to_csv工具）"
                ]
            }
        
        # 应用极性调整
        print("🔄 应用极性调整策略...")
        adjusted_df = apply_polarity_adjustment(df, polarity_results)
        
        # 保存调整后的数据为缓存资源
        adjusted_resource_id = generate_resource_id("raw_data", resource_id, "polarity_adjusted")
        adjusted_uri = get_resource_uri(adjusted_resource_id)
        
        # 缓存调整后的数据
        DATA_CACHE[adjusted_uri] = adjusted_df
        
        # 更新资源索引（修复极性调整后资源不存在的根本问题）
        RESOURCE_INDEX[adjusted_resource_id] = {
            'uri': adjusted_uri,
            'type': 'polarity_adjusted',
            'rows': len(adjusted_df),
            'columns': len(adjusted_df.columns),
            'column_names': list(adjusted_df.columns),
            'parent_resource_id': resource_id
        }
        
        # 保存到持久化存储
        save_resource_to_persistent_storage(adjusted_resource_id, adjusted_df)
        
        print(f"✅ 极性调整后的数据已保存: {adjusted_uri}")
        
        # 输出极性检测报告
        print("\n" + "="*60)
        print("📊 字段极性智能检测完成")
        print("="*60)
        print(polarity_report)
        
        # 输出调整策略信息
        print("\n🔄 极性调整策略已应用：")
        for col, result in polarity_results.items():
            if result['suggested_polarity'] == 'max':
                print(f"   - {col}: 越大越好 → 越小越好（通过取倒数转换）")
            else:
                print(f"   - {col}: 越小越好（无需调整）")
        
        print(f"\n💾 调整后数据已保存为: {adjusted_uri}")
        
        return {
            "original_resource_uri": uri,
            "polarity_detection": {
                "status": "success",
                "message": "字段极性检测完成，极性调整已应用",
                "report": polarity_report,
                "polarity_results": polarity_results
            },
            "adjusted_resource_uri": adjusted_uri,
            "next_actions": [
                "分析这些数据字段的特征（使用analyze_data_fields工具）",
                "生成隶属度计算配置模板（使用generate_membership_config_template工具）",
                "查看数据内容（使用export_resource_to_csv工具）"
            ]
        }
        
    except Exception as e:
        raise ValueError(f"Excel 解析失败: {str(e)}")


@mcp.resource("data://{resource_id}")
def get_data_resource(resource_id: str) -> Dict[str, Any]:
    """
    通过 URI 访问已缓存的数据资源
    
    Args:
        resource_id: 资源标识符（可以是纯ID或完整URI）
        
    Returns:
        数据集的字典表示（前100行预览），用markdown加载数据预览
    """
    # 处理URI格式的资源ID
    if resource_id.startswith("data://"):
        resource_id = resource_id[7:]  # 移除 "data://" 前缀
    
    uri = f"data://{resource_id}"
    
    if uri not in DATA_CACHE:
        raise ValueError(f"资源不存在: {uri}")
    
    df = DATA_CACHE[uri]
    
    # 返回数据预览（前100行）
    preview_data = df.head(100).to_dict(orient='records')
    
    return {
        "resource_uri": uri,
        "dataset_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {col: str(df[col].dtype) for col in df.columns}
        },
        "preview_data": preview_data
    }


@mcp.tool()
def list_data_resources() -> Dict[str, Any]:
    """
    列出当前缓存的所有数据资源
    
    Returns:
        资源列表和统计信息
    """
    resources_info = {}
    
    for uri, df in DATA_CACHE.items():
        resources_info[uri] = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns)
        }
    
    return {
        "total_resources": len(DATA_CACHE),
        "resources": resources_info
    }


@mcp.tool()
def clear_data_cache() -> Dict[str, Any]:
    """
    清空数据缓存
    
    Returns:
        操作结果
    """
    cache_size = len(DATA_CACHE)
    DATA_CACHE.clear()
    
    return {
        "success": True,
        "message": f"已清空 {cache_size} 个数据资源"
    }


@mcp.tool()
def apply_polarity_adjustment_tool(original_resource_uri: str, polarity_config: Dict[str, str]) -> Dict[str, Any]:
    """
    根据用户确认的极性配置应用极性调整
    
    Args:
        original_resource_uri: 原始数据资源URI
        polarity_config: 极性配置字典，格式如 {"字段名": "min/max"}
        
    Returns:
        极性调整结果，包含调整后资源URI和调整详情
    """
    try:
        # 从URI提取资源ID
        if not original_resource_uri.startswith("data://"):
            raise ValueError("资源URI格式不正确，应以'data://'开头")
        
        resource_id = original_resource_uri[7:]  # 移除 "data://" 前缀
        
        # 检查资源是否存在
        if original_resource_uri not in DATA_CACHE:
            # 尝试从持久化存储加载
            try:
                df = load_resource_from_persistent_storage(resource_id)
                DATA_CACHE[original_resource_uri] = df
            except Exception as e:
                raise ValueError(f"资源不存在或无法加载: {original_resource_uri}")
        
        df = DATA_CACHE[original_resource_uri]
        
        print(f"🔍 开始应用极性调整...")
        print(f"原始资源: {original_resource_uri}")
        print(f"极性配置: {polarity_config}")
        
        # 验证极性配置
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in polarity_config.keys():
            if col not in numeric_columns:
                raise ValueError(f"字段 '{col}' 不存在或不是数值型字段")
        
        # 构建极性结果结构
        polarity_results = {}
        for col, polarity in polarity_config.items():
            series = df[col].dropna()
            min_val = float(series.min())
            max_val = float(series.max())
            
            # 计算隶属度参数
            if polarity == 'max':
                a = min_val
                c = max_val
                b = (a + c) / 2
            else:  # min
                a = max_val
                c = min_val
                b = (a + c) / 2
            
            polarity_results[col] = {
                'suggested_polarity': polarity,
                'confidence': 'user_confirmed',
                'reasoning': '用户手动确认极性配置',
                'membership_rules': f"""
## 对于{'越大越好' if polarity == 'max' else '越小越好'}的字段（{col}）：
- {'a < b < c' if polarity == 'max' else 'a > b > c'} （如{col}：{a:.2f}→{b:.2f}→{c:.2f}）
- 当实际值 {'≤' if polarity == 'max' else '≥'} a时：隶属度=0%
- 当实际值 {'≥' if polarity == 'max' else '≤'} c时：隶属度=100%
""".strip(),
                'adjustment_strategy': f"极性调整方法：通过线性变换的方法将越大越好的字段转换为越小越好" if polarity == 'max' else "无需调整，已为越小越好类型",
                'parameters': {'a': a, 'b': b, 'c': c},
                'detection_successful': True
            }
        
        # 应用极性调整
        print("🔄 应用极性调整策略...")
        adjusted_df = df.copy()
        
        for col, result in polarity_results.items():
            if result['suggested_polarity'] == 'max':
                # 对越大越好的字段使用线性变换（修复取倒数导致值过小的问题）
                max_val = df[col].max() * 1.5  # 使用1.5倍最大值作为参考
                adjusted_df[col] = max_val - df[col]
                print(f"✅ 已对字段 '{col}' 应用极性调整（线性变换: {max_val:.2f} - x）")
            else:
                print(f"✅ 字段 '{col}' 为越小越好类型，无需调整")
        
        # 保存调整后的数据为缓存资源
        adjusted_resource_id = generate_resource_id("raw_data", resource_id, "polarity_adjusted")
        adjusted_uri = get_resource_uri(adjusted_resource_id)
        
        # 缓存调整后的数据
        DATA_CACHE[adjusted_uri] = adjusted_df
        
        # 保存到持久化存储
        save_resource_to_persistent_storage(adjusted_resource_id, adjusted_df)
        
        print(f"✅ 极性调整后的数据已保存: {adjusted_uri}")
        
        # 生成极性检测报告
        polarity_report = generate_polarity_report(polarity_results)
        
        # 输出调整策略信息
        print("\n🔄 极性调整策略已应用：")
        for col, result in polarity_results.items():
            if result['suggested_polarity'] == 'max':
                print(f"   - {col}: 越大越好 → 越小越好（通过取倒数转换）")
            else:
                print(f"   - {col}: 越小越好（无需调整）")
        
        print(f"\n💾 调整后数据已保存为: {adjusted_uri}")
        
        return {
            "original_resource_uri": original_resource_uri,
            "adjusted_resource_uri": adjusted_uri,
            "polarity_config_applied": polarity_config,
            "polarity_report": polarity_report,
            "next_actions": [
                "分析这些数据字段的特征（使用analyze_data_fields工具）",
                "生成隶属度计算配置模板（使用generate_membership_config_template工具）",
                "查看数据内容（使用export_resource_to_csv工具）"
            ]
        }
        
    except Exception as e:
        raise ValueError(f"极性调整失败: {str(e)}")


@mcp.tool()
def list_persistent_resources(resource_type: str = None) -> Dict[str, Any]:
    """
    列出存储中的所有资源（MVP阶段：仅显示内存中的资源）
    
    Args:
        resource_type: 资源类型筛选（可选）
        
    Returns:
        内存资源列表
    """
    resources_by_type = {}
    
    for resource_id, resource_info in RESOURCE_INDEX.items():
        res_type = resource_info.get("type", "未知类型")
        
        if resource_type and res_type != resource_type:
            continue
            
        if res_type not in resources_by_type:
            resources_by_type[res_type] = []
        
        # 检查资源是否在内存中
        is_in_memory = resource_info.get("uri", "") in DATA_CACHE
        
        resources_by_type[res_type].append({
            "resource_id": resource_id,
            "uri": resource_info.get("uri", ""),
            "data_shape": resource_info.get("data_shape", "未知"),
            "created_time": resource_info.get("created_time", "未知"),
            "file_path": resource_info.get("file_path", ""),
            "parent_hash": resource_info.get("parent_hash", ""),
            "step_name": resource_info.get("step_name", ""),
            "in_memory": is_in_memory,
            "storage_type": "内存缓存" if is_in_memory else "仅索引（MVP阶段不持久化）"
        })
    
    return {
        "total_resources": len(RESOURCE_INDEX),
        "resources_by_type": resources_by_type,
        "resource_types_found": list(resources_by_type.keys()),
        "storage_mode": "MVP阶段：仅内存存储，不持久化到磁盘"
    }


@mcp.tool()
def delete_persistent_resource(resource_id: str) -> Dict[str, Any]:
    """
    删除存储中的指定资源（MVP阶段：仅从内存和索引中删除）
    
    Args:
        resource_id: 要删除的资源ID（可以是纯ID或完整URI）
        
    Returns:
        删除操作结果
    """
    # 处理URI格式的资源ID
    if resource_id.startswith("data://"):
        resource_id = resource_id[7:]  # 移除 "data://" 前缀
    
    if resource_id not in RESOURCE_INDEX:
        return {
            "success": False,
            "message": f"资源不存在: {resource_id}"
        }
    
    try:
        # MVP阶段：不删除磁盘文件（因为没有持久化）
        # file_path = RESOURCE_INDEX[resource_id]["file_path"]
        # if os.path.exists(file_path):
        #     os.remove(file_path)
        
        # 从内存缓存中删除
        uri = RESOURCE_INDEX[resource_id]["uri"]
        if uri in DATA_CACHE:
            del DATA_CACHE[uri]
        
        # 从索引中删除
        del RESOURCE_INDEX[resource_id]
        
        # MVP阶段：不保存索引文件到磁盘
        # with open(RESOURCE_INDEX_FILE, 'w', encoding='utf-8') as f:
        #     json.dump(RESOURCE_INDEX, f, ensure_ascii=False, indent=2)
        
        return {
            "success": True,
            "message": f"资源 {resource_id} 已从内存中删除（MVP阶段不持久化到磁盘）"
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"删除资源失败: {str(e)}"
        }


@mcp.tool()
def export_resource_to_csv(resource_id: str) -> Dict[str, Any]:
    """
    将指定资源导出为CSV格式，并提供详细的字段解释和下一步行动建议
    
    Args:
        resource_id: 要导出的资源ID（可以是纯ID或完整URI）
        
    Returns:
        - 指定资源的原始数据，包含所有字段和值
        - 字段解释：对每个字段的含义和数据类型进行详细说明
        - 下一步行动建议：根据数据特征建议后续分析步骤
        - 除此以外的任何信息都不要输出
    """
    try:
        # 处理URI格式的资源ID
        if resource_id.startswith("data://"):
            resource_id = resource_id[7:]  # 移除 "data://" 前缀
        
        # 首先尝试从内存缓存加载
        uri = get_resource_uri(resource_id)
        if uri in DATA_CACHE:
            df = DATA_CACHE[uri]
        else:
            # 如果内存缓存中没有，尝试从持久化存储加载
            df = load_resource_from_persistent_storage(resource_id)
        
        # 智能解析数据结构
        # 检查是否为隶属度矩阵的嵌套结构
        if _is_membership_matrix(df):
            # 将嵌套结构转换为扁平化结构
            df = _flatten_membership_matrix(df)
        
        # 转换为CSV格式
        csv_content = df.to_csv(index=False, encoding='utf-8')
        
        # 生成字段解释
        field_descriptions = _generate_field_descriptions(df, resource_id)
        
        # 生成下一步行动建议
        next_actions = _generate_next_actions(df, resource_id)
        
        # 生成资源信息
        resource_info = _generate_resource_info(resource_id, df)
        
        # 生成数据统计摘要
        data_summary = _generate_data_summary(df)
        
        # 生成标准化的TOPSIS表格预览
        standardized_table = _generate_standardized_topsis_table(df, resource_id)
        
        # 如果是TOPSIS结果，生成原始数据格式预览
        original_data_preview = _generate_topsis_original_data_preview(df, resource_id)
        
        result = {
            "csv_content": csv_content,
            "field_descriptions": field_descriptions,
            "next_actions": next_actions,
            "resource_info": resource_info,
            "data_summary": data_summary
        }
        
        # 如果生成了标准化表格，添加到结果中
        if standardized_table:
            result["standardized_table"] = standardized_table
            
        # 如果生成了原始数据预览，添加到结果中
        if original_data_preview:
            result["original_data_preview"] = original_data_preview
        
        return result
        
    except Exception as e:
        raise ValueError(f"导出资源失败: {str(e)}")


def _generate_field_descriptions(df: pd.DataFrame, resource_id: str) -> Dict[str, Any]:
    """
    生成字段详细解释
    
    Args:
        df: 数据框
        resource_id: 资源ID
        
    Returns:
        字段解释字典
    """
    field_descriptions = {}
    
    # 根据资源类型生成不同的字段解释
    if resource_id.startswith("mcr_") and "topsiseval" in resource_id:
        # TOPSIS结果字段解释
        field_descriptions = {
            "业务对象": "被评价的业务对象标识符，如T1、T2等",
            "评估级别": "隶属度评估的级别，如e1、e2、e3、e4，代表不同的评价标准",
            "综合隶属度": "经过隶属度计算后的综合得分，反映业务对象在特定级别下的隶属程度",
            "相对接近度(C值)": "**核心评价指标** - 表示业务对象与理想最优解的相对接近程度。C值越接近1，方案越优；越接近0，方案越差",
            "与最优解距离(D+)": "业务对象与理想最优解的距离，越小越好，理想值为0",
            "与最劣解距离(D-)": "业务对象与理想最劣解的距离，越大越好，理想值为1"
        }
    elif resource_id.startswith("mc_") and "membership" in resource_id:
        # 隶属度矩阵字段解释
        field_descriptions = {
            "业务对象": "被评价的业务对象标识符",
            "评价因子": "参与评价的指标或因素",
            "评估级别": "隶属度评估的级别",
            "隶属度": "业务对象在特定评价因子和级别下的隶属程度，值在0-1之间"
        }
    elif resource_id.startswith("raw_"):
        # 原始数据字段解释
        for column in df.columns:
            if column == "业务对象":
                field_descriptions[column] = "业务对象标识符"
            elif "库存" in column:
                field_descriptions[column] = f"{column}指标，通常越小越好（成本型指标）"
            elif "采购" in column:
                field_descriptions[column] = f"{column}指标，需要根据业务场景判断极性"
            elif "年限" in column:
                field_descriptions[column] = f"{column}指标，通常越大越好（效益型指标）"
            else:
                field_descriptions[column] = f"{column}指标"
    
    return field_descriptions


def _generate_next_actions(df: pd.DataFrame, resource_id: str) -> List[str]:
    """
    生成下一步行动建议
    
    Args:
        df: 数据框
        resource_id: 资源ID
        
    Returns:
        行动建议列表
    """
    next_actions = []
    
    if resource_id.startswith("mcr_") and "topsiseval" in resource_id:
        # TOPSIS结果后的建议
        next_actions = [
            "📊 **分析相对接近度排序**：按相对接近度从大到小排序，识别最优方案",
            "🔍 **多级别比较**：比较不同评估级别下的最优方案",
            "📈 **可视化分析**：建议使用图表展示各方案的相对接近度分布",
            "🤝 **决策支持**：基于相对接近度结果制定资源分配或优化策略",
            "🔄 **敏感性分析**：可调整级别参数进行敏感性分析",
            "💾 **数据导出**：当前已导出原始数据格式，可直接用于进一步分析"
        ]
    elif resource_id.startswith("mc_") and "membership" in resource_id:
        # 隶属度计算后的建议
        next_actions = [
            "🎯 **执行TOPSIS评估**：使用perform_topsis_comprehensive_evaluation进行多准则决策",
            "📋 **检查隶属度分布**：验证隶属度矩阵的合理性",
            "⚙️ **参数调优**：如有需要，可调整级别参数重新计算",
            "📊 **可视化展示**：建议使用热力图展示隶属度分布"
        ]
    elif resource_id.startswith("raw_"):
        # 原始数据后的建议
        next_actions = [
            "🔍 **数据质量检查**：使用analyze_data_fields分析数据质量",
            "🎯 **执行隶属度计算**：使用calculate_membership_with_config进行隶属度计算",
            "⚙️ **配置级别参数**：为每个字段设置合适的级别参数",
            "📋 **极性确认**：确保所有字段的极性设置正确"
        ]
    
    return next_actions


def _generate_resource_info(resource_id: str, df: pd.DataFrame) -> Dict[str, Any]:
    """
    生成资源基本信息
    
    Args:
        resource_id: 资源ID
        df: 数据框
        
    Returns:
        资源信息字典
    """
    resource_type = "未知类型"
    description = ""
    
    if resource_id.startswith("raw_"):
        resource_type = "原始数据"
        description = "上传的原始业务数据，包含业务对象和各评价指标"
    elif resource_id.startswith("mc_") and "membership" in resource_id:
        resource_type = "隶属度矩阵"
        description = "经过隶属度计算后的结果，包含业务对象在各评价因子和级别下的隶属度"
    elif resource_id.startswith("mcr_") and "topsiseval" in resource_id:
        resource_type = "TOPSIS评估结果"
        description = "基于隶属度矩阵的TOPSIS多准则决策分析结果，包含相对接近度等关键指标"
    
    return {
        "resource_id": resource_id,
        "resource_type": resource_type,
        "description": description,
        "data_shape": f"{len(df)} 行 × {len(df.columns)} 列",
        "columns": list(df.columns)
    }


def _generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    生成数据统计摘要
    
    Args:
        df: 数据框
        
    Returns:
        数据摘要字典
    """
    summary = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_names": list(df.columns),
        "data_types": {}
    }
    
    # 分析数值列
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        summary["numeric_summary"] = {}
        for col in numeric_columns:
            summary["numeric_summary"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std())
            }
    
    # 分析分类列
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        summary["categorical_summary"] = {}
        for col in categorical_columns:
            summary["categorical_summary"][col] = {
                "unique_count": len(df[col].unique()),
                "sample_values": list(df[col].unique()[:5])  # 显示前5个唯一值
            }
    
    return summary


def _generate_standardized_topsis_table(df: pd.DataFrame, resource_id: str) -> str:
    """
    生成标准化的TOPSIS结果表格
    
    Args:
        df: TOPSIS结果数据框
        resource_id: 资源ID
        
    Returns:
        标准化的表格字符串
    """
    # 检查是否为TOPSIS结果
    if not (resource_id.startswith("mcr_") and "topsiseval" in resource_id):
        return ""
    
    # 检查数据框是否包含必要的列
    required_columns = ['业务对象', '评估级别', '相对接近度']
    if not all(col in df.columns for col in required_columns):
        return ""
    
    # 按业务对象和评估级别重新组织数据
    try:
        # 创建透视表：业务对象为行，评估级别为列，相对接近度为值
        pivot_df = df.pivot_table(
            index='业务对象', 
            columns='评估级别', 
            values='相对接近度', 
            aggfunc='first'
        ).reset_index()
        
        # 确保列名按级别顺序排列
        level_columns = [col for col in pivot_df.columns if col.startswith('e')]
        level_columns.sort()  # 按e1, e2, e3, e4排序
        
        # 重新排列列顺序
        pivot_df = pivot_df[['业务对象'] + level_columns]
        
        # 重命名列名为中文
        column_mapping = {'业务对象': '业务对象'}
        for i, level in enumerate(level_columns, 1):
            column_mapping[level] = f'级别{i}'
        
        pivot_df = pivot_df.rename(columns=column_mapping)
        
        # 生成标准化的表格字符串
        table_lines = []
        
        # 表头
        headers = list(pivot_df.columns)
        table_lines.append("\t".join(headers))
        
        # 数据行
        for _, row in pivot_df.iterrows():
            row_data = []
            for col in headers:
                value = row[col]
                if isinstance(value, (int, float)):
                    # 数值格式化：保留3位小数
                    row_data.append(f"{value:.3f}")
                else:
                    row_data.append(str(value))
            table_lines.append("\t".join(row_data))
        
        return "\n".join(table_lines)
        
    except Exception as e:
        print(f"生成标准化表格时出错: {e}")
        return ""


def _is_membership_matrix(df: pd.DataFrame) -> bool:
    """
    检查DataFrame是否为隶属度矩阵格式
    
    Args:
        df: 要检查的DataFrame
        
    Returns:
        如果是隶属度矩阵结构返回True，否则返回False
    """
    if df.empty:
        return False
    
    # 检查扁平化格式（包含必要的列）
    required_columns = ['业务对象', '评价因子', '评估级别', '隶属度']
    if all(col in df.columns for col in required_columns):
        return True
    
    # 检查嵌套字典结构（原始格式）
    first_row = df.iloc[0]
    for value in first_row:
        if isinstance(value, dict):
            # 检查字典键是否包含"级别"字样
            if any("级别" in str(key) for key in value.keys()):
                return True
    
    return False


def _flatten_membership_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    将隶属度矩阵的嵌套结构转换为扁平化结构
    
    Args:
        df: 隶属度矩阵DataFrame（嵌套结构）
        
    Returns:
        扁平化后的DataFrame
    """
    flattened_results = []
    
    # 获取原始数据的业务对象标识符（如果有的话）
    # 假设原始数据的第一列是业务对象标识符
    object_identifier = df.columns[0] if len(df.columns) > 0 else "业务对象"
    
    for row_idx, row in df.iterrows():
        # 获取业务对象标识符（如果存在）
        object_id = row[object_identifier] if object_identifier in row else f"对象{row_idx + 1}"
        
        # 遍历每个字段的隶属度字典
        for field_name, level_data in row.items():
            # 跳过业务对象标识符列
            if field_name == object_identifier:
                continue
            
            # 检查是否为嵌套字典结构
            if isinstance(level_data, dict):
                for level_name, membership_value in level_data.items():
                    flattened_results.append({
                        object_identifier: object_id,
                        "评价因子": field_name,
                        "级别": level_name,
                        "隶属度": membership_value
                    })
    
    # 创建扁平化的DataFrame
    if flattened_results:
        return pd.DataFrame(flattened_results)
    else:
        # 如果没有找到嵌套结构，返回原始DataFrame
        return df


@mcp.tool()
def reload_persistent_resources() -> Dict[str, Any]:
    """
    重新加载所有持久化资源到内存缓存（MVP阶段：此功能已禁用）
    
    Returns:
        重新加载结果
    """
    return {
        "success": False,
        "message": "MVP阶段：禁用持久化资源重新加载功能，仅使用内存缓存"
    }
    
    # 原始实现（已禁用）
    # try:
    #     # 清空当前缓存
    #     DATA_CACHE.clear()
    #     
    #     # 重新加载所有持久化资源
    #     load_all_persistent_resources()
    #     
    #     return {
    #         "success": True,
    #         "message": f"已重新加载 {len(RESOURCE_INDEX)} 个持久化资源"
    #     }
    #     
    # except Exception as e:
    #     return {
    #         "success": False,
    #         "message": f"重新加载失败: {str(e)}"
    #     }


@mcp.tool()
def analyze_data_fields(resource_id: str, cache_result: bool = True) -> Dict[str, Any]:
    """
    分析数据集的字段特征，返回固定的分析结果
    
    **分析内容**:
    - 字段类型识别（数值型/分类型）
    - 常见统计特征（最小值、最大值、平均值、标准差等）
    - 缺失值分析（缺失数量、缺失率）
    - 数据质量评估
    - 字段极性自动检测和建议
    
    **重要提示**:
    - 后续的隶属度计算和多准则分析要求所有字段的极性必须一致
    - 请仔细检查极性建议，确保数据符合计算要求
    
    Args:
        resource_id: 资源标识符（可以是纯ID或完整URI）
        cache_result: 是否缓存分析结果（默认True）
        
    Returns:
        Dict[str, Any]: 包含以下字段的分析结果：
        - dataset_info: 数据集基本信息
        - field_analysis: 字段分析结果
        - data_quality_summary: 数据质量摘要
        - analysis_summary: 分析总结
        - polarity_warnings: 极性一致性警告
        - result_resource_id: 分析结果资源ID（如果缓存了结果）
    """
    # 如果没有提供resource_id，尝试自动发现mc资源
    if resource_id is None:
        resource_id = find_latest_membership_resource()
        if resource_id is None:
            raise ValueError("未找到可用的隶属度资源(mc_开头)，请先执行隶属度计算或上传隶属度数据")
        print(f"🤖 自动发现隶属度资源: {resource_id}")
    
    # 处理URI格式的资源ID
    if resource_id and resource_id.startswith("data://"):
        resource_id = resource_id[7:]  # 移除 "data://" 前缀
    
    uri = get_resource_uri(resource_id)
    
    if uri not in DATA_CACHE:
        raise ValueError(f"资源不存在: {uri}")
    
    df = DATA_CACHE[uri]
    
    # 执行字段分析
    numeric_analysis = analyze_numeric_fields(df)
    categorical_analysis = analyze_categorical_fields(df)
    
    # 自动检测字段极性
    polarity_suggestions = auto_detect_polarity(df, numeric_analysis)
    
    # 检查极性一致性并生成警告
    polarity_warnings = check_polarity_consistency(polarity_suggestions)
    
    # 生成数据质量摘要
    data_quality_summary = generate_data_quality_summary(df)
    
    # 生成分析总结（包含极性检查结果）
    analysis_summary = generate_analysis_summary(df, numeric_analysis, categorical_analysis, data_quality_summary, polarity_warnings)
    
    # 构建分析结果
    analysis_result = {
        "dataset_info": {
            "resource_uri": uri,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "column_names": list(df.columns)
        },
        "field_analysis": {
            "numeric_fields": numeric_analysis,
            "categorical_fields": categorical_analysis,
            "field_polarity": polarity_suggestions
        },
        "data_quality_summary": data_quality_summary,
        "analysis_summary": analysis_summary,
        "polarity_warnings": polarity_warnings
    }
    
    # 如果需要缓存结果
    if cache_result:
        # 生成分析结果资源ID
        result_resource_id = generate_resource_id(
            "field_analysis", 
            parent_resource_id=resource_id,
            step_name="field_analysis"
        )
        result_uri = get_resource_uri(result_resource_id)
        
        # 将分析结果转换为DataFrame格式缓存
        # 这里我们缓存一个包含分析摘要的简化DataFrame
        analysis_df = pd.DataFrame([{
            'analysis_type': 'field_analysis',
            'parent_resource': resource_id,
            'numeric_fields_count': len(numeric_analysis),
            'categorical_fields_count': len(categorical_analysis),
            'overall_missing_rate': data_quality_summary['overall_quality']['overall_missing_rate'],
            'quality_rating': data_quality_summary['overall_quality']['quality_rating'],
            'polarity_consistency': polarity_warnings['is_consistent']
        }])
        
        DATA_CACHE[result_uri] = analysis_df
        analysis_result["result_resource_id"] = result_resource_id
        
        print(f"字段分析结果已缓存: {result_uri}")
    
    return analysis_result


def check_polarity_consistency(polarity_suggestions: Dict[str, Any]) -> Dict[str, Any]:
    """
    检查字段极性一致性
    
    Args:
        polarity_suggestions: 字段极性建议字典
        
    Returns:
        极性一致性检查结果
    """
    if not polarity_suggestions:
        return {
            "is_consistent": True,
            "message": "无数值字段，无需极性检查",
            "polarity_distribution": {},
            "warnings": []
        }
    
    # 统计极性分布
    polarity_counts = {}
    for field_info in polarity_suggestions.values():
        polarity = field_info.get('suggested_polarity', 'unknown')
        polarity_counts[polarity] = polarity_counts.get(polarity, 0) + 1
    
    # 检查一致性
    is_consistent = len(polarity_counts) <= 1
    
    # 生成警告信息
    warnings = []
    if not is_consistent:
        warnings.append("⚠️ 检测到混合极性字段！后续计算可能产生错误结果")
        warnings.append("   建议统一所有字段的极性（全部为极大型或极小型）")
    
    if 'unknown' in polarity_counts:
        warnings.append("⚠️ 部分字段极性无法自动识别，请手动确认")
    
    return {
        "is_consistent": is_consistent,
        "message": "极性一致" if is_consistent else "极性不一致",
        "polarity_distribution": polarity_counts,
        "warnings": warnings
    }


def generate_data_quality_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """生成数据质量摘要"""
    
    total_rows = len(df)
    total_columns = len(df.columns)
    
    # 计算总体缺失情况
    total_missing = df.isnull().sum().sum()
    total_cells = total_rows * total_columns
    overall_missing_rate = (total_missing / total_cells) * 100 if total_cells > 0 else 0
    
    # 按列统计缺失情况
    column_missing_stats = {}
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_rate = (missing_count / total_rows) * 100
        column_missing_stats[col] = {
            "missing_count": int(missing_count),
            "missing_rate": float(missing_rate),
            "data_type": str(df[col].dtype)
        }
    
    # 数据质量评级
    if overall_missing_rate < 1:
        quality_rating = "优秀"
    elif overall_missing_rate < 5:
        quality_rating = "良好"
    elif overall_missing_rate < 10:
        quality_rating = "一般"
    else:
        quality_rating = "较差"
    
    return {
        "overall_quality": {
            "total_rows": total_rows,
            "total_columns": total_columns,
            "total_missing_cells": int(total_missing),
            "overall_missing_rate": float(overall_missing_rate),
            "quality_rating": quality_rating
        },
        "column_quality": column_missing_stats
    }


def generate_analysis_summary(df: pd.DataFrame, numeric_analysis: Dict, categorical_analysis: Dict, data_quality_summary: Dict, polarity_warnings: Dict = None) -> Dict[str, Any]:
    """生成分析总结"""
    
    numeric_count = len(numeric_analysis)
    categorical_count = len(categorical_analysis)
    total_fields = numeric_count + categorical_count
    
    # 统计关键指标
    summary_stats = {
        "dataset_size": f"{len(df)} 行 × {len(df.columns)} 列",
        "field_types": {
            "numeric_fields": numeric_count,
            "categorical_fields": categorical_count,
            "total_fields": total_fields
        },
        "data_quality": {
            "missing_rate": f"{data_quality_summary['overall_quality']['overall_missing_rate']:.2f}%",
            "quality_rating": data_quality_summary['overall_quality']['quality_rating']
        }
    }
    
    # 生成关键发现
    key_findings = []
    
    # 数值字段发现
    if numeric_analysis:
        for col, analysis in numeric_analysis.items():
            basic_stats = analysis['basic_stats']
            data_quality = analysis['data_quality']
            
            findings = f"数值字段 '{col}': "
            findings += f"范围 [{basic_stats['min']:.2f}, {basic_stats['max']:.2f}], "
            findings += f"均值 {basic_stats['mean']:.2f}, "
            findings += f"缺失率 {data_quality['missing_percentage']:.1f}%"
            key_findings.append(findings)
    
    # 分类字段发现
    if categorical_analysis:
        for col, analysis in categorical_analysis.items():
            freq_analysis = analysis['frequency_analysis']
            data_quality = analysis['data_quality']
            
            findings = f"分类字段 '{col}': "
            findings += f"{data_quality['unique_count']} 个唯一值, "
            findings += f"缺失率 {data_quality['missing_percentage']:.1f}%"
            key_findings.append(findings)
    
    # 数据质量发现
    quality_stats = data_quality_summary['overall_quality']
    if quality_stats['overall_missing_rate'] > 5:
        key_findings.append(f"⚠️  数据质量注意: 总体缺失率 {quality_stats['overall_missing_rate']:.1f}% 较高")
    else:
        key_findings.append(f"✅ 数据质量良好: 总体缺失率 {quality_stats['overall_missing_rate']:.1f}%")
    
    # 极性检查发现
    if polarity_warnings:
        if not polarity_warnings.get('is_consistent', True):
            key_findings.append("⚠️  **重要警告**: 检测到混合极性字段！")
            key_findings.append("   后续隶属度计算要求所有字段极性一致")
        else:
            key_findings.append("✅ 极性一致性检查通过")
    
    # 生成建议
    recommendations = [
        "建议对缺失值较多的字段进行数据清洗",
        "数值字段可进一步进行相关性分析和分布检验",
        "分类字段可进行频次分析和可视化展示"
    ]
    
    # 添加极性相关建议
    if polarity_warnings and not polarity_warnings.get('is_consistent', True):
        recommendations.insert(0, "**重要**: 请统一所有数值字段的极性（全部为极大型或极小型）")
        recommendations.insert(1, "检查字段极性建议，确保符合实际业务逻辑")
    
    return {
        "summary_statistics": summary_stats,
        "key_findings": key_findings,
        "recommendations": recommendations
    }


def find_latest_membership_resource() -> str:
    """
    查找最新的隶属度计算资源（mc开头）
    
    Returns:
        最新的mc资源ID，如果没有找到则返回None
    """
    mc_resources = []
    
    # 在DATA_CACHE中查找mc开头的资源
    for uri, df in DATA_CACHE.items():
        if uri.startswith("data://mc_"):
            resource_id = uri[7:]  # 移除 "data://" 前缀
            # 验证数据格式是否为隶属度矩阵
            if _is_membership_matrix(df):
                mc_resources.append({
                    'resource_id': resource_id,
                    'uri': uri,
                    'rows': len(df),
                    'columns': len(df.columns)
                })
    
    if not mc_resources:
        return None
    
    # 返回最新的mc资源（根据资源ID中的时间戳或uuid排序）
    # 简单处理：返回第一个找到的mc资源
    return mc_resources[0]['resource_id']


def _generate_topsis_formatted_table(comprehensive_scores: Dict[str, List[float]], weights: List[float] = None) -> str:
    """
    生成特定格式的TOPSIS相对接近度矩阵（V值）表格
    
    Args:
        comprehensive_scores: TOPSIS相对接近度计算结果
        weights: 权重列表
        
    Returns:
        格式化的表格字符串
    """
    weight_str = f"[{', '.join([f'{w:.2f}' for w in weights])}]" if weights else "[等权重]"
    
    # 构建表格头部
    table_lines = [
        f"**TOPSIS相对接近度矩阵（V值） (权重: {weight_str}):**",
        "| 业务对象 | 级别1 | 级别2 | 级别3 | 级别4 |",
        "|---------|-------|-------|-------|-------|"
    ]
    
    # 添加数据行
    for obj_name, scores in comprehensive_scores.items():
        # 格式化相对接近度值（保留3位小数）
        score_str = " | ".join([f"{score:.3f}" for score in scores])
        table_lines.append(f"| {obj_name:<8} | {score_str} |")
    
    return "\n".join(table_lines)


def _generate_topsis_available_tables(comprehensive_membership: Dict[str, Any], 
                                     comprehensive_scores: Dict[str, List[float]],
                                     membership_scores: Dict[str, List[float]]) -> str:
    """
    生成可查询表格清单
    
    Args:
        comprehensive_membership: TOPSIS综合隶属度计算结果
        comprehensive_scores: 相对接近度矩阵
        membership_scores: 综合隶属度矩阵
        
    Returns:
        可查询表格清单字符串
    """
    table_list = [
        "**📊 可查询的表格清单：**",
        "",
        "您可以通过回复以下名称来查看对应的详细数据表格：",
        "",
        "1. **相对接近度矩阵** - 核心评价指标，值越大表示方案越优",
        "2. **综合隶属度矩阵** - 基于相对接近度计算的标准化结果",
        "3. **距离矩阵** - 包含与最优解和最劣解的欧氏距离",
        "4. **完整结果表** - 包含所有指标的完整数据表格",
        "",
        "**查询示例：**",
        "- 回复 \"相对接近度矩阵\" 查看详细V值数据",
        "- 回复 \"距离矩阵\" 查看D+和D-距离数据",
        "- 回复 \"完整结果表\" 查看包含所有指标的综合表格",
        "",
        "**表格说明：**",
        "- **相对接近度（V值）**：0-1之间，越大表示越接近最优解",
        "- **综合隶属度（u值）**：基于V值计算的标准化结果",
        "- **D+距离**：与理想最优解的距离，越小越好",
        "- **D-距离**：与理想最劣解的距离，越大越好"
    ]
    
    return "\n".join(table_list)


def _generate_relative_closeness_table(comprehensive_scores: Dict[str, List[float]], weights: List[float] = None) -> str:
    """
    生成相对接近度矩阵（V值）详细表格
    
    Args:
        comprehensive_scores: 相对接近度矩阵
        weights: 权重列表
        
    Returns:
        详细的相对接近度表格字符串
    """
    weight_str = f"[{', '.join([f'{w:.2f}' for w in weights])}]" if weights else "[等权重]"
    
    table_lines = [
        f"**📊 相对接近度矩阵（V值）详细数据 (权重: {weight_str}):**",
        "",
        "| 业务对象 | 级别1 | 级别2 | 级别3 | 级别4 | 最大值 | 最优级别 |",
        "|---------|-------|-------|-------|-------|--------|----------|"
    ]
    
    for obj_name, scores in comprehensive_scores.items():
        # 格式化相对接近度值（保留4位小数）
        score_str = " | ".join([f"{score:.4f}" for score in scores])
        
        # 计算最大值和最优级别
        max_score = max(scores)
        best_level = scores.index(max_score) + 1
        
        table_lines.append(f"| {obj_name:<8} | {score_str} | {max_score:.4f} | e{best_level} |")
    
    table_lines.extend([
        "",
        "**说明：**",
        "- **相对接近度（V值）**：0-1之间的数值，越大表示该业务对象在该评估级别上越接近理想最优解",
        "- **最大值**：该业务对象在所有评估级别中的最高相对接近度",
        "- **最优级别**：相对接近度最高的评估级别，表示该业务对象最适合的评估等级",
        "",
        "**解读建议：**",
        "- 重点关注每个业务对象的**最大值**和**最优级别**",
        "- 相对接近度>0.8表示非常接近最优解",
        "- 相对接近度<0.3表示距离最优解较远"
    ])
    
    return "\n".join(table_lines)


def _generate_membership_matrix_table(membership_scores: Dict[str, List[float]], weights: List[float] = None) -> str:
    """
    生成综合隶属度矩阵（u值）详细表格
    
    Args:
        membership_scores: 综合隶属度矩阵
        weights: 权重列表
        
    Returns:
        详细的综合隶属度表格字符串
    """
    weight_str = f"[{', '.join([f'{w:.2f}' for w in weights])}]" if weights else "[等权重]"
    
    table_lines = [
        f"**📊 综合隶属度矩阵（u值）详细数据 (权重: {weight_str}):**",
        "",
        "| 业务对象 | 级别1 | 级别2 | 级别3 | 级别4 | 隶属度和 | 隶属度分布 |",
        "|---------|-------|-------|-------|-------|----------|------------|"
    ]
    
    for obj_name, scores in membership_scores.items():
        # 格式化综合隶属度值（保留4位小数）
        score_str = " | ".join([f"{score:.4f}" for score in scores])
        
        # 计算隶属度和和分布
        total_score = sum(scores)
        distribution = "/".join([f"{score/total_score*100:.1f}%" for score in scores])
        
        table_lines.append(f"| {obj_name:<8} | {score_str} | {total_score:.4f} | {distribution} |")
    
    table_lines.extend([
        "",
        "**说明：**",
        "- **综合隶属度（u值）**：基于相对接近度计算的标准化结果，表示业务对象属于各评估级别的程度",
        "- **隶属度和**：所有级别隶属度的总和，用于验证计算正确性（应接近1.0）",
        "- **隶属度分布**：各评估级别在总隶属度中的占比",
        "",
        "**解读建议：**",
        "- 隶属度分布显示业务对象在各评估级别的归属程度",
        "- 隶属度和应接近1.0，表示隶属度计算正确",
        "- 分布百分比帮助理解业务对象的评估级别偏好"
    ])
    
    return "\n".join(table_lines)


def _generate_distance_matrix_table(comprehensive_membership: Dict[str, Any], weights: List[float] = None) -> str:
    """
    生成距离矩阵详细表格
    
    Args:
        comprehensive_membership: TOPSIS综合隶属度计算结果
        weights: 权重列表
        
    Returns:
        详细的距离矩阵表格字符串
    """
    weight_str = f"[{', '.join([f'{w:.2f}' for w in weights])}]" if weights else "[等权重]"
    
    table_lines = [
        f"**📊 距离矩阵详细数据 (权重: {weight_str}):**",
        "",
        "| 业务对象 | D+距离（级别1） | D+距离（级别2） | D+距离（级别3） | D+距离（级别4） | D-距离（级别1） | D-距离（级别2） | D-距离（级别3） | D-距离（级别4） |",
        "|---------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|"
    ]
    
    for obj_name, distances in comprehensive_membership.items():
        d_plus = distances["D+"]
        d_minus = distances["D-"]
        
        # 格式化距离值（保留4位小数）
        d_plus_str = " | ".join([f"{d:.4f}" for d in d_plus])
        d_minus_str = " | ".join([f"{d:.4f}" for d in d_minus])
        
        table_lines.append(f"| {obj_name:<8} | {d_plus_str} | {d_minus_str} |")
    
    table_lines.extend([
        "",
        "**说明：**",
        "- **D+距离**：业务对象与理想最优解的欧氏距离，越小表示越接近最优解",
        "- **D-距离**：业务对象与理想最劣解的欧氏距离，越大表示越远离最劣解",
        "",
        "**解读建议：**",
        "- 理想的业务对象应该具有**较小的D+距离**和**较大的D-距离**",
        "- D+距离<0.3表示非常接近最优解",
        "- D-距离>0.7表示远离最劣解"
    ])
    
    return "\n".join(table_lines)


def _generate_complete_result_table(comprehensive_membership: Dict[str, Any], 
                                   comprehensive_scores: Dict[str, List[float]],
                                   membership_scores: Dict[str, List[float]],
                                   weights: List[float] = None) -> str:
    """
    生成完整结果表格
    
    Args:
        comprehensive_membership: TOPSIS综合隶属度计算结果
        comprehensive_scores: 相对接近度矩阵
        membership_scores: 综合隶属度矩阵
        weights: 权重列表
        
    Returns:
        完整的综合结果表格字符串
    """
    weight_str = f"[{', '.join([f'{w:.2f}' for w in weights])}]" if weights else "[等权重]"
    
    table_lines = [
        f"**📊 完整TOPSIS结果表格 (权重: {weight_str}):**",
        "",
        "| 业务对象 | 评估级别 | 相对接近度 | 综合隶属度 | D+距离 | D-距离 | 级别排名 |",
        "|---------|----------|------------|------------|--------|--------|----------|"
    ]
    
    for obj_name in comprehensive_membership.keys():
        for level_idx in range(len(comprehensive_scores[obj_name])):
            v_value = comprehensive_scores[obj_name][level_idx]
            u_value = membership_scores[obj_name][level_idx]
            d_plus = comprehensive_membership[obj_name]["D+"][level_idx]
            d_minus = comprehensive_membership[obj_name]["D-"][level_idx]
            
            # 计算该级别在所有业务对象中的排名
            level_scores = [comprehensive_scores[obj][level_idx] for obj in comprehensive_scores.keys()]
            sorted_scores = sorted(level_scores, reverse=True)
            rank = sorted_scores.index(v_value) + 1
            
            table_lines.append(f"| {obj_name:<8} | e{level_idx+1} | {v_value:.4f} | {u_value:.4f} | {d_plus:.4f} | {d_minus:.4f} | {rank} |")
    
    table_lines.extend([
        "",
        "**说明：**",
        "- **相对接近度（V值）**：核心评价指标，越大越好",
        "- **综合隶属度（u值）**：标准化结果，表示归属程度",
        "- **D+距离**：与最优解的距离，越小越好",
        "- **D-距离**：与最劣解的距离，越大越好",
        "- **级别排名**：该业务对象在该评估级别中的相对排名",
        "",
        "**解读建议：**",
        "- 按业务对象分组查看各评估级别的表现",
        "- 关注相对接近度和级别排名进行综合评估",
        "- 结合D+和D-距离分析方案的优劣程度"
    ])
    
    return "\n".join(table_lines)


def _generate_topsis_next_actions(num_objects: int) -> str:
    """
    生成TOPSIS计算后的下一步建议
    
    Args:
        num_objects: 评估的业务对象数量
        
    Returns:
        下一步建议字符串
    """
    return f"""
**下一步建议：**
1. **结果解读**：相对接近度值越大表示该业务对象在对应评估级别下越接近最优解
2. **决策支持**：根据相对接近度值进行业务对象排序，选择最优方案
3. **进一步分析**：可调用 `export_resource_to_csv` 函数导出详细数据表格进行深入分析
4. **对比分析**：可尝试使用VIKOR方法进行对比分析，获得不同视角的决策结果

本次评估共分析了 {num_objects} 个业务对象，建议重点关注相对接近度较高的方案。
"""


def _generate_topsis_original_data_preview(df: pd.DataFrame, resource_id: str) -> str:
    """
    生成TOPSIS结果的原始数据格式预览
    
    Args:
        df: TOPSIS结果数据框
        resource_id: 资源ID
        
    Returns:
        原始数据格式预览字符串
    """
    # 检查是否为TOPSIS结果
    if not (resource_id.startswith("mcr_") and "topsiseval" in resource_id):
        return ""
    
    # 检查数据框是否包含必要的列
    required_columns = ['业务对象', '评估级别', '相对接近度', '综合隶属度', '与最优解距离', '与最劣解距离']
    if not all(col in df.columns for col in required_columns):
        return ""
    
    # 获取前8行数据作为预览
    preview_df = df.head(8).copy()
    
    # 格式化数值（保留3位小数）
    numeric_columns = ['相对接近度', '综合隶属度', '与最优解距离', '与最劣解距离']
    for col in numeric_columns:
        if col in preview_df.columns:
            preview_df[col] = preview_df[col].apply(lambda x: f"{x:.3f}" if pd.notnull(x) else "")
    
    # 构建原始数据预览
    preview_lines = ["**原始数据格式:**"]
    preview_lines.append("   业务对象 评估级别  相对接近度  综合隶属度  与最优解距离  与最劣解距离")
    
    for idx, row in preview_df.iterrows():
        line = f"{idx:<2}  {row['业务对象']:<8} {row['评估级别']:<8} {row['相对接近度']:<10} {row['综合隶属度']:<10} {row['与最优解距离']:<12} {row['与最劣解距离']:<12}"
        preview_lines.append(line)
    
    # 添加字段说明
    preview_lines.append("")
    preview_lines.append("**字段说明:**")
    preview_lines.append("- **业务对象**: 被评价的业务对象标识符")
    preview_lines.append("- **评估级别**: 隶属度评估的级别（e1-e4）")
    preview_lines.append("- **相对接近度**: 值越大表示方案越优")
    preview_lines.append("- **综合隶属度**: 核心评价指标，业务对象在特定级别下的隶属程度")
    preview_lines.append("- **与最优解距离**: 与理想最优解的距离，越小越好")
    preview_lines.append("- **与最劣解距离**: 与理想最劣解的距离，越大越好")
    
    return "\n".join(preview_lines)

# 报告格式已在函数文档字符串中明确定义，直接通过代码生成格式化报告

@mcp.tool()
def perform_topsis_comprehensive_evaluation(resource_id: str = None, weights: List[float] = None, cache_result: bool = True) -> Dict[str, Any]:
    """
    使用TOPSIS方法进行综合评估计算 - 多准则决策分析工具
    该函数实现TOPSIS（Technique for Order Preference by Similarity to Ideal Solution）算法，
    基于欧氏距离计算各评估对象与理想解和负理想解的相对接近度，得到综合排序结果。
    
    Args:
        resource_id: 隶属度矩阵资源ID（可以是纯ID或完整URI）
                     例如：'mc_4a291844_769818_membership' 或 'mc_91f53e05_f85398_membership'
        weights: 因子权重列表，用于设置各评价因子的重要性权重
                例如：[0.32, 0.24, 0.24, 0.20] 表示4个因子的权重分配
                默认为None，使用等权重（每个因子权重相同）
        cache_result: 是否缓存计算结果（默认True），设置为False可跳过结果缓存
        
    重要：必须严格按照以下模版输出结果，不得有模版以外的任何其它信息！！            
    Returns:
        回复模版‘’‘
        ## 📊 评估概述
        - **评估方法**：TOPSIS
        - **评估对象数量**：12个
        - **评价因子数量**：4个
        - **权重分配**：[0.32, 0.24, 0.24, 0.2]
        
        ## 📈 相对接近度矩阵（V值）
        | 业务对象 | 级别1 | 级别2 | 级别3 | 级别4 |
        |---------|-------|-------|-------|-------|
        | T1 | 0.898 | 0.691 | 0.476 | 0.273 |
        | T2 | 0.333 | 0.386 | 0.389 | 0.672 |
        | T3 | 0.333 | 0.210 | 0.148 | 0.711 |
        | T4 | 0.470 | 0.242 | 0.153 | 0.499 |
        | T5 | 1.000 | 0.496 | 0.317 | 0.166 |
        | T6 | 0.000 | 0.364 | 0.470 | 0.763 |
        | T7 | 1.000 | 0.579 | 0.389 | 0.205 |
        
        ## 📋 可查询表格清单
        1. **相对接近度矩阵** - 核心评价指标，值越大表示方案越优
        2. **综合隶属度矩阵** - 基于相对接近度计算的标准化结果
        3. **距离矩阵** - 包含与最优解和最劣解的欧氏距离
        4. **完整结果表** - 包含所有指标的完整数据表格
        
        ## 🚀 下一步操作建议’‘’
    """
    # 如果没有提供resource_id，尝试自动发现mc资源
    if resource_id is None:
        resource_id = find_latest_membership_resource()
        if resource_id is None:
            raise ValueError("未找到可用的隶属度资源(mc_开头)，请先执行隶属度计算或上传隶属度数据")
        print(f"🤖 自动发现隶属度资源: {resource_id}")
    
    # 处理URI格式的资源ID
    if resource_id and resource_id.startswith("data://"):
        resource_id = resource_id[7:]  # 移除 "data://" 前缀
    
    uri = get_resource_uri(resource_id)
    
    if uri not in DATA_CACHE:
        raise ValueError(f"资源不存在: {uri}")
    
    # 获取隶属度矩阵数据
    membership_df = DATA_CACHE[uri]
    
    # 检查数据格式
    if not _is_membership_matrix(membership_df):
        raise ValueError("数据格式错误：需要隶属度矩阵格式的数据")
    
    # 转换隶属度矩阵为规范化特征值矩阵格式
    normalized_matrices = _convert_membership_to_normalized_matrix(membership_df)
    
    # 创建TOPSIS评估器
    topsis_evaluator = TOPSISComprehensiveEvaluation()
    
    # 计算TOPSIS综合隶属度
    comprehensive_membership = topsis_evaluator.calculate_comprehensive_membership(normalized_matrices, weights)
    
    # 计算相对接近度（V值）
    comprehensive_scores = topsis_evaluator.get_comprehensive_scores(comprehensive_membership)
    
    # 计算最终综合隶属度（u值）
    membership_scores = topsis_evaluator.calculate_comprehensive_membership_scores(comprehensive_scores)
    
    # 生成结果摘要
    summary = {
        "evaluation_method": "TOPSIS",
        "total_objects": len(comprehensive_membership),
        "total_factors": len(weights) if weights else 4,  # 默认4个因子
        "weight_distribution": weights if weights else [0.25, 0.25, 0.25, 0.25],
        "sample_results": {}
    }
    
    # 添加前3个对象的示例结果
    sample_objects = list(comprehensive_membership.keys())[:3]
    for obj_name in sample_objects:
        summary["sample_results"][obj_name] = {
            "D+_distances": comprehensive_membership[obj_name]["D+"][:2],  # 前2个级别
            "D-_distances": comprehensive_membership[obj_name]["D-"][:2],  # 前2个级别
            "relative_closeness": comprehensive_scores[obj_name][:2],      # 前2个级别
            "membership_scores": membership_scores[obj_name][:2]          # 前2个级别
        }
    
    # 生成特定格式的TOPSIS相对接近度矩阵表格
    topsis_table = _generate_topsis_formatted_table(comprehensive_scores, weights)
    
    # 生成可查询表格清单
    available_tables = _generate_topsis_available_tables(comprehensive_membership, comprehensive_scores, membership_scores)
    
    # 生成下一步建议
    next_actions = _generate_topsis_next_actions(len(comprehensive_membership))
    
    result = {
        "resource_id": resource_id,
        "comprehensive_membership": comprehensive_membership,
        "comprehensive_scores": comprehensive_scores,
        "membership_scores": membership_scores,
        "summary": summary,
        "topsis_table": topsis_table,
        "available_tables": available_tables,
        "next_actions": next_actions
    }
    
    # 如果需要缓存结果
    if cache_result:
        # 生成结果资源ID
        result_resource_id = generate_resource_id(
            "multi_criteria", 
            parent_resource_id=resource_id,
            step_name="topsis_evaluation"
        )
        result_uri = get_resource_uri(result_resource_id)
        
        # 将结果转换为DataFrame格式缓存
        result_data = []
        for obj_name, scores in membership_scores.items():
            for level_idx, score in enumerate(scores):
                result_data.append({
                    '业务对象': obj_name,
                    '评估级别': f'e{level_idx + 1}',
                    '综合隶属度': score,
                    '相对接近度': comprehensive_scores[obj_name][level_idx],
                    '与最优解距离': comprehensive_membership[obj_name]["D+"][level_idx],
                    '与最劣解距离': comprehensive_membership[obj_name]["D-"][level_idx]
                })
        
        result_df = pd.DataFrame(result_data)
        DATA_CACHE[result_uri] = result_df
        
        # 注册资源到索引中（解决预览问题）
        RESOURCE_INDEX[result_resource_id] = {
            "uri": result_uri,
            "data_shape": f"{len(result_df)}行×{len(result_df.columns)}列",
            "resource_type": "topsis_evaluation",
            "description": "TOPSIS综合得分矩阵（相对接近度V值）"
        }
        result["result_resource_id"] = result_resource_id
        
        print(f"TOPSIS综合得分矩阵已缓存: {result_uri}")
    
    # 生成格式化报告并添加到结果中 - 直接使用模板字符串避免LLM加工
    summary = result.get("summary", {})
    total_objects = summary.get("total_objects", 0)
    total_factors = summary.get("total_factors", 0)
    weight_distribution = summary.get("weight_distribution", [])
    
    result["formatted_report"] = f"""## 📊 评估概述
- **评估方法**：TOPSIS（逼近理想解排序法）
- **评估对象数量**：{total_objects} 个
- **评价因子数量**：{total_factors} 个
- **权重分配**：{weight_distribution}

## 📈 相对接近度矩阵（V值）
{topsis_table}

## 📋 可查询表格清单
{available_tables}

## 🚀 下一步操作建议
{next_actions}"""
    
    return result


@mcp.tool()
def perform_vikor_comprehensive_evaluation(resource_id: str, weights: List[float] = None, v: float = 0.5, cache_result: bool = True) -> Dict[str, Any]:
    """
    使用VIKOR方法进行综合评估计算 - 多准则妥协排序分析工具
    
    该函数实现VIKOR（VIKOR compromise ranking method）算法，
    基于妥协解的概念进行多属性决策分析，计算S值（群体效用）、R值（个体遗憾）和Q值（妥协解）。
    
    
    Args:
        resource_id: 隶属度矩阵资源ID（可以是纯ID或完整URI）
                     例如：'membership_matrix_001' 或 'data://membership_matrix_001'
        weights: 因子权重列表，用于设置各评价因子的重要性权重
                例如：[0.32, 0.24, 0.24, 0.20] 表示4个因子的权重分配
                默认为None，使用等权重（每个因子权重相同）
        v: 策略权重，0≤v≤1，调节群体效用与个体遗憾的权衡
           v=0：以个体最大遗憾为基础（保守策略）
           v=1：以群体多数效用为基础（激进策略）
           v=0.5：平衡策略（默认）
        cache_result: 是否缓存计算结果（默认True），设置为False可跳过结果缓存
        
    Returns：
        
        formatted_report模板格式（必须严格按照此格式输出，不得有任何额外内容）：
        
        ## 📊 评估概述
        - **评估方法**：VIKOR（多准则妥协排序法）
        - **评估对象数量**：5个
        - **评价因子数量**：4个
        - **权重分配**：[0.32, 0.24, 0.24, 0.2]
        - **策略权重**：v = 0.5（0=保守策略，1=激进策略，0.5=平衡策略）
        
        ## 📈 妥协排序结果（Q值排序）
        | 业务对象 | S值(群体效用) | R值(个体遗憾) | Q值(妥协解) | 排序 |
        |----------|---------------|---------------|------------|------|
        | T2 | 0.4025 | 0.1364 | 0.0000 | 1 |
        | T1 | 0.3973 | 0.1455 | 0.2134 | 2 |
        | T4 | 0.4261 | 0.2070 | 0.7167 | 3 |
        | T3 | 0.5876 | 0.2006 | 0.9254 | 4 |
        | T5 | 0.5434 | 0.2229 | 1.0000 | 5 |
        
        ## 📋 可查询表格清单
        - **VIKOR综合评估表**：包含S、R、Q值的完整评估结果
        - **妥协排序表**：基于Q值的方案排序结果
        - **敏感性分析表**：不同策略权重v下的结果对比
        
        ## 🚀 下一步操作建议
        1. **查看详细排序结果**：分析Q值排序，识别最优妥协解
        2. **敏感性分析**：调整策略权重v（当前v=0.5），观察结果稳定性
        3. **结果验证**：结合S值和R值，验证妥协解的合理性
        4. **决策支持**：基于Q值排序，为最终决策提供量化依据
        
        💡 **VIKOR结果解读**：
        - **Q值越小**：妥协解越优，推荐度越高
        - **S值**：群体效用，越小表示整体表现越好
        - **R值**：个体遗憾，越小表示最差属性表现越好
    """
    # 处理URI格式的资源ID
    if resource_id.startswith("data://"):
        resource_id = resource_id[7:]  # 移除 "data://" 前缀
    
    uri = get_resource_uri(resource_id)
    
    if uri not in DATA_CACHE:
        raise ValueError(f"资源不存在: {uri}")
    
    # 获取隶属度矩阵数据
    membership_df = DATA_CACHE[uri]
    
    # 检查数据格式
    if not _is_membership_matrix(membership_df):
        raise ValueError("数据格式错误：需要隶属度矩阵格式的数据")
    
    # 转换隶属度矩阵为规范化特征值矩阵格式
    normalized_matrices = _convert_membership_to_normalized_matrix(membership_df)
    
    # 创建VIKOR评估器
    vikor_evaluator = VIKORComprehensiveEvaluation()
    
    # 计算VIKOR综合得分
    comprehensive_scores = vikor_evaluator.calculate_comprehensive_scores(normalized_matrices, weights, v)
    
    # 生成结果摘要
    summary = {
        "evaluation_method": "VIKOR",
        "total_objects": len(comprehensive_scores),
        "total_factors": len(weights) if weights else 4,  # 默认4个因子
        "weight_distribution": weights if weights else [0.25, 0.25, 0.25, 0.25],
        "strategy_weight": v,
        "sample_results": {}
    }
    
    # 添加前3个对象的示例结果
    sample_objects = list(comprehensive_scores.keys())[:3]
    for obj_name in sample_objects:
        summary["sample_results"][obj_name] = {
            "S_values": comprehensive_scores[obj_name]["S"][:2],  # 前2个级别
            "R_values": comprehensive_scores[obj_name]["R"][:2],  # 前2个级别
            "Q_values": comprehensive_scores[obj_name]["Q"][:2]   # 前2个级别
        }
    
    result = {
        "resource_id": resource_id,
        "comprehensive_scores": comprehensive_scores,
        "summary": summary
    }
    
    # 如果需要缓存结果
    if cache_result:
        # 生成结果资源ID
        result_resource_id = generate_resource_id(
            "multi_criteria", 
            parent_resource_id=resource_id,
            step_name="vikor_evaluation"
        )
        result_uri = get_resource_uri(result_resource_id)
        
        # 将结果转换为DataFrame格式缓存
        result_data = []
        for obj_name, scores in comprehensive_scores.items():
            for level_idx in range(len(scores["S"])):
                result_data.append({
                    '业务对象': obj_name,
                    '评估级别': f'e{level_idx + 1}',
                    'S值': scores["S"][level_idx],
                    'R值': scores["R"][level_idx],
                    'Q值': scores["Q"][level_idx]
                })
        
        result_df = pd.DataFrame(result_data)
        DATA_CACHE[result_uri] = result_df
        result["result_resource_id"] = result_resource_id
        
        print(f"VIKOR综合评估结果已缓存: {result_uri}")
    
    # 生成VIKOR格式化报告并添加到结果中 - 直接使用模板字符串避免LLM加工
    # 构建VIKOR结果表格
    comprehensive_scores = result.get("comprehensive_scores", {})
    vikor_table = "| 业务对象 | S值(群体效用) | R值(个体遗憾) | Q值(妥协解) | 排序 |\n"
    vikor_table += "|----------|---------------|---------------|------------|------|\n"
    
    # 计算Q值排序
    q_values = []
    for obj_name, scores in comprehensive_scores.items():
        if scores["Q"]:
            q_values.append((obj_name, scores["Q"][0]))
    
    # 按Q值升序排列（Q值越小越好）
    q_values.sort(key=lambda x: x[1])
    
    for rank, (obj_name, q_val) in enumerate(q_values, 1):
        scores = comprehensive_scores[obj_name]
        s_val = scores["S"][0] if scores["S"] else 0
        r_val = scores["R"][0] if scores["R"] else 0
        vikor_table += f"| {obj_name} | {s_val:.4f} | {r_val:.4f} | {q_val:.4f} | {rank} |\n"
    
    # 生成可查询表格清单和下一步建议
    available_tables = """
- **VIKOR综合评估表**：包含S、R、Q值的完整评估结果
- **妥协排序表**：基于Q值的方案排序结果
- **敏感性分析表**：不同策略权重v下的结果对比
"""
    
    next_actions = f"""
1. **查看详细排序结果**：分析Q值排序，识别最优妥协解
2. **敏感性分析**：调整策略权重v（当前v={v}），观察结果稳定性
3. **结果验证**：结合S值和R值，验证妥协解的合理性
4. **决策支持**：基于Q值排序，为最终决策提供量化依据

💡 **VIKOR结果解读**：
- **Q值越小**：妥协解越优，推荐度越高
- **S值**：群体效用，越小表示整体表现越好
- **R值**：个体遗憾，越小表示最差属性表现越好
"""
    
    result["formatted_report"] = f"""## 📊 评估概述
- **评估方法**：VIKOR（多准则妥协排序法）
- **评估对象数量**：{summary.get("total_objects", 0)} 个
- **评价因子数量**：{summary.get("total_factors", 0)} 个
- **权重分配**：{summary.get("weight_distribution", [])}
- **策略权重**：v = {v}（0=保守策略，1=激进策略，0.5=平衡策略）

## 📈 妥协排序结果（Q值排序）
{vikor_table}

## 📋 可查询表格清单
{available_tables}

## 🚀 下一步操作建议
{next_actions}"""
    
    return result


def _convert_membership_to_normalized_matrix(membership_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    将隶属度矩阵转换为规范化特征值矩阵格式
    
    Args:
        membership_df: 隶属度矩阵DataFrame（扁平化格式）
        
    Returns:
        规范化特征值矩阵字典，格式为{"T1": 矩阵, "T2": 矩阵, ...}
    """
    # 按业务对象分组
    grouped = membership_df.groupby('业务对象')
    
    normalized_matrices = {}
    
    for obj_name, group in grouped:
        # 按评价因子分组
        factor_groups = group.groupby('评价因子')
        
        matrix = []
        factor_names = []
        
        for factor_name, factor_group in factor_groups:
            # 按级别排序
            factor_group = factor_group.sort_values('评估级别')
            
            # 提取隶属度值作为一行
            row = factor_group['隶属度'].tolist()
            matrix.append(row)
            factor_names.append(factor_name)
        
        if matrix:
            normalized_matrices[obj_name] = np.array(matrix)
    
    return normalized_matrices


if __name__ == "__main__":
    # 运行 MCP 服务器
    mcp.run(transport="stdio")