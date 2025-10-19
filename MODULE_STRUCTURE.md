# 📋 智能数据分析工具 - 模块结构说明

本说明文档描述了基于MCP框架的智能数据分析工具的模块结构和功能划分。

## 🏗️ 整体架构

```
main.py
├── 📦 依赖导入与基础配置
├── 🗄️ 数据存储与缓存管理
├── 🔧 资源管理 - 资源ID生成与解析
├── 📊 数据上传与解析模块
├── 🔍 字段分析与极性检测模块
├── 📈 隶属度计算模块
├── 🎯 TOPSIS综合评估模块
├── ⚖️ VIKOR综合评估模块
├── 📋 辅助工具与实用函数
├── 🔧 内部辅助函数
└── 🚀 主程序入口
```

## 📑 各模块详细说明

### 📦 依赖导入与基础配置
- **功能**：导入所有必需的第三方库和自定义模块
- **包含**：FastMCP服务器创建、全局变量定义、模块导入
- **关键组件**：
  - `mcp`: FastMCP服务器实例
  - `DATA_CACHE`: 内存数据缓存（MVP阶段）
  - `RESOURCE_INDEX`: 资源索引管理
  - `RESOURCE_TYPES`: 标准化的资源类型映射
  - 模块导入：`field_analyzer`, `membership_functions`, `topsis_comprehensive_evaluation`, `vikor_comprehensive_evaluation`

### 🗄️ 数据存储与缓存管理
- **功能**：管理数据缓存和持久化存储（MVP阶段：仅内存存储）
- **关键变量**：
  - `DATA_CACHE`: 会话级内存缓存
  - `PERSISTENT_STORAGE_DIR`: 持久化存储目录（MVP阶段未使用）
  - `RESOURCE_INDEX_FILE`: 资源索引文件路径（MVP阶段未使用）
- **核心函数**：
  - `load_resource_from_persistent_storage()`: 从内存加载资源
  - `list_persistent_resources()`: 列出存储中的所有资源
  - `delete_persistent_resource()`: 删除指定资源
  - `save_resource_to_persistent_storage()`: 保存资源到内存缓存

### 🔧 资源管理 - 资源ID生成与解析
- **功能**：资源ID的生成、解析和管理
- **核心函数**：
  - `generate_resource_id()`: 生成标准化资源ID（格式：{type}_{uuid}_{parent_hash}_{step}）
  - `get_resource_uri()`: 生成资源URI（格式：data://{resource_id}）
  - `parse_resource_id()`: 解析资源ID信息
  - `get_resource_dependency_chain()`: 获取资源的依赖链信息

### 📊 数据上传与解析模块
- **功能**：处理数据文件的上传和解析，自动触发字段极性智能检测
- **MCP工具**：
  - `@mcp.tool() upload_csv()`: 上传CSV文本数据并缓存为资源
  - `@mcp.tool() upload_excel()`: 上传Excel文件数据并缓存为资源
  - `@mcp.resource() get_data_resource()`: 通过URI访问已缓存的数据资源
  - `@mcp.tool() list_data_resources()`: 列出当前缓存的所有数据资源
  - `@mcp.tool() list_resources_by_type()`: 按类型列出资源
  - `@mcp.tool() clear_data_cache()`: 清空数据缓存
- **支持格式**：CSV、Excel
- **特性**：
  - 自动触发字段极性智能检测
  - 标准化的资源命名和URI生成
  - MVP阶段仅使用内存缓存

### 🔍 字段分析与极性检测模块
- **功能**：分析数据字段特征和极性检测
- **MCP工具**：
  - `@mcp.tool() analyze_data_fields()`: 分析数据集的字段特征
  - `@mcp.tool() apply_polarity_adjustment_tool()`: 根据极性配置调整数据极性
- **分析内容**：
  - 字段类型识别（数值型/分类型）
  - 常见统计特征分析
  - 缺失值分析和数据质量评估
  - 字段极性自动检测和建议

### 📈 隶属度计算模块
- **功能**：基于下界型函数的隶属度计算
- **MCP工具**：
  - `@mcp.tool() generate_membership_config_template()`: 生成下界型隶属度计算配置模板
  - `@mcp.tool() validate_membership_config()`: 验证下界型隶属度计算配置
  - `@mcp.tool() calculate_membership_with_config()`: 使用下界型函数配置执行隶属度计算
- **算法特点**：
  - 使用下界型级别隶属函数
  - 支持多级别模糊综合评价
  - 基于数学公式的严格验证

### 🎯 TOPSIS综合评估模块
- **功能**：TOPSIS多准则决策分析
- **MCP工具**：
  - `@mcp.tool() perform_topsis_comprehensive_evaluation()`: 实现TOPSIS多准则决策分析算法
- **算法原理**：
  - 逼近理想解排序法
  - 基于欧氏距离计算相对接近度
  - 支持自定义权重分配的多属性决策分析

### 🎯 等级综合评定模块
- **功能**：基于TOPSIS结果的等级综合评定，包括级别特征值计算和二元语义评定
- **MCP工具**：
  - `@mcp.tool() perform_grade_comprehensive_assessment()`: 执行等级综合评定算法
- **算法原理**：
  - **级别特征值计算**：基于公式4-32，计算业务对象的级别特征值
    \[ v_j = \sum_{k=1}^{h} k u_{jk} \quad (j = 1,2,\cdots,n) \]
  - **二元语义评定**：基于公式4-33，将级别特征值表示为有序组(e_k, α_{jk})
    \[ k = \text{Round}(v_j),\ \alpha_{jk} = v_j - k \]
- **输入要求**：需要TOPSIS结果作为输入（相对接近度矩阵或综合隶属度向量）
- **输出结果**：级别特征值、二元语义、最终等级评定

### ⚖️ VIKOR综合评估模块
- **功能**：VIKOR妥协排序分析
- **MCP工具**：
  - `@mcp.tool() perform_vikor_comprehensive_evaluation()`: 使用VIKOR方法进行综合评估计算
- **算法特点**：
  - 多准则妥协排序
  - 计算S值（群体效用）、R值（个体遗憾）、Q值（妥协解）
  - 实现多准则妥协排序分析

### 📋 辅助工具与实用函数
- **功能**：提供通用辅助功能和工具函数
- **MCP工具**：
  - `@mcp.tool() get_resource_dependency_chain()`: 获取资源依赖关系链
  - `@mcp.tool() list_persistent_resources()`: 列出持久化存储中的资源
  - `@mcp.tool() delete_persistent_resource()`: 删除持久化存储中的指定资源
- **其他函数**：
  - 数据验证和格式化工具
  - 错误处理和日志记录功能
  - 配置管理和状态检查工具

### 🔧 内部辅助函数
- **功能**：模块内部使用的辅助函数
- **主要函数**：
  - `save_resource_to_persistent_storage()`: 将资源保存到持久化存储
  - `generate_resource_id()`: 生成唯一的资源标识符
  - `parse_resource_id()`: 解析资源标识符获取详细信息
  - `get_resource_metadata()`: 获取资源的元数据信息
- **特点**：
  - 不直接暴露为MCP工具
  - 支持模块间功能复用
  - 提供底层数据操作能力

### 🚀 主程序入口
- **文件**：`main.py`
- **功能**：
  - 初始化FastMCP服务器实例
  - 注册所有MCP工具和资源
  - 启动服务监听
- **配置**：
  - 端口配置：默认8000
  - 日志级别：INFO
  - 数据缓存目录：`./data_cache`
  - 持久化存储目录：`./persistent_storage`
  - 资源索引文件：`./resource_index.json`
- **实现**：`if __name__ == "__main__": mcp.run()`

## 🔄 典型工作流程

### 完整数据分析流程
```
1. 📊 数据上传 → upload_and_parse_data()
2. 🔍 字段分析 → analyze_data_fields()
3. ⚡ 极性调整 → auto_detect_and_adjust_polarity()
4. 📈 隶属度计算 → calculate_membership_with_config()
5. 🎯 综合评估 → perform_topsis_comprehensive_evaluation() 或 perform_vikor_comprehensive_evaluation()
6. 🎯 等级评定 → perform_grade_comprehensive_assessment()
```

### 快速配置流程
```
1. 📊 数据上传 → upload_and_parse_data()
2. 📋 配置模板 → generate_membership_config_template()
3. ✔️ 配置验证 → validate_membership_config()
4. 📈 隶属度计算 → calculate_membership_with_config()
5. 🎯 综合评估 → TOPSIS/VIKOR评估
```

## 📊 资源类型说明

| 资源类型 | 前缀 | 说明 | 示例 |
|---------|------|------|------|
| raw_data | raw | 原始上传数据 | raw_abc123 |
| field_analysis | fa | 字段分析结果 | fa_def456 |
| membership_calc | mc | 隶属度计算结果 | mc_ghi789 |
| multi_criteria | mcr | 多准则评估结果 | mcr_jkl012 |
| binary_semantic | bs | 二元语义结果 | bs_mno345 |
| other | other | 其他类型 | other_pqr678 |

## 🎯 使用建议

1. **模块化使用**：根据具体需求选择相应模块，不必执行完整流程
2. **资源管理**：利用资源ID和URI进行数据追踪和管理
3. **配置验证**：在执行计算前务必验证配置的正确性
4. **结果缓存**：合理使用缓存功能提高重复操作效率
5. **错误处理**：关注各函数的异常处理和错误提示

## ⚠️ 注意事项

- MVP阶段禁用磁盘持久化，仅使用内存缓存
- 所有资源ID支持纯ID和完整URI两种格式
- 隶属度计算必须使用下界型函数
- 极性调整确保所有字段极性一致
- TOPSIS和VIKOR评估需要隶属度矩阵作为输入