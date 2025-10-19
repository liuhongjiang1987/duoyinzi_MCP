# 🚀 智能数据分析工具 - 快速参考指南

## 📋 代码结构速览

整理后的代码已按照功能模块进行清晰分块，每个模块都有明确的职责和边界：

```
main_organized.py
├── 📦 依赖导入与基础配置          # 所有导入和基础设置
├── 🗄️ 数据存储与缓存管理          # 缓存和存储相关变量
├── 🔧 工具类方法 - 资源ID管理      # 资源ID生成和解析
├── 🗃️ 数据存储与缓存管理 - 持久化操作  # 资源保存加载操作
├── 📊 数据上传与解析模块          # 数据上传功能
├── 🔍 字段分析与极性检测模块      # 字段分析和极性处理
├── 📈 隶属度计算模块              # 隶属度计算相关
├── 🎯 TOPSIS综合评估模块          # TOPSIS算法
├── ⚖️ VIKOR综合评估模块            # VIKOR算法
├── 📋 辅助工具与实用函数          # 各种辅助功能
├── 🔧 内部辅助函数                # 内部使用函数
└── 🚀 主程序入口                  # 程序启动
```

## 🎯 核心功能模块

### 1️⃣ 数据上传与解析模块
**位置**：`📊 数据上传与解析模块`
**主要函数**：`upload_and_parse_data()`
**功能**：上传CSV/Excel文件并解析为标准化数据格式

### 2️⃣ 字段分析与极性检测模块
**位置**：`🔍 字段分析与极性检测模块`
**主要函数**：
- `analyze_data_fields()` - 字段特征分析
- `auto_detect_and_adjust_polarity()` - 自动极性调整
- `generate_polarity_adjustment_report()` - 极性调整报告

### 3️⃣ 隶属度计算模块
**位置**：`📈 隶属度计算模块`
**主要函数**：
- `generate_membership_config_template()` - 生成配置模板
- `validate_membership_config()` - 验证配置
- `calculate_membership_with_config()` - 执行隶属度计算

### 4️⃣ TOPSIS/VIKOR综合评估模块
**位置**：
- `🎯 TOPSIS综合评估模块`
- `⚖️ VIKOR综合评估模块`
**主要函数**：
- `perform_topsis_comprehensive_evaluation()`
- `perform_vikor_comprehensive_evaluation()`

## 🔧 工具类方法

### 资源ID管理
**位置**：`🔧 工具类方法 - 资源ID管理`
**函数**：
- `generate_resource_id()` - 生成资源ID
- `get_resource_uri()` - 获取资源URI
- `parse_resource_id()` - 解析资源ID

### 数据存储管理
**位置**：`🗃️ 数据存储与缓存管理 - 持久化操作`
**函数**：
- `save_resource_to_persistent_storage()` - 保存资源
- `load_resource_from_persistent_storage()` - 加载资源

### 辅助工具
**位置**：`📋 辅助工具与实用函数`
**函数**：
- `get_resource_dependency_chain()` - 依赖链分析
- `export_resource_to_csv()` - 导出CSV
- `list_all_resources()` - 资源列表
- `find_latest_membership_resource()` - 查找最新隶属度资源

## 📝 使用流程示例

### 完整流程
```python
# 1. 上传数据
result = upload_and_parse_data(file_content, "data.csv")
resource_id = result["resource_id"]

# 2. 字段分析
analysis = analyze_data_fields(resource_id)

# 3. 极性调整
adjusted = auto_detect_and_adjust_polarity(resource_id, "positive")

# 4. 隶属度计算
template = generate_membership_config_template(resource_id)
# ... 配置验证和计算

# 5. 综合评估
topsis_result = perform_topsis_comprehensive_evaluation(membership_resource_id)
```

### 快速配置流程
```python
# 1. 上传数据
result = upload_and_parse_data(file_content, "data.csv")

# 2. 生成配置模板
template = generate_membership_config_template(result["resource_id"])

# 3. 验证配置
validation = validate_membership_config(result["resource_id"], config)

# 4. 计算隶属度
membership = calculate_membership_with_config(result["resource_id"], config)

# 5. 执行评估
evaluation = perform_topsis_comprehensive_evaluation(membership["result_resource_id"])
```

## 🔍 快速定位函数

| 功能需求 | 模块位置 | 主要函数 |
|---------|----------|----------|
| 上传数据文件 | 📊 数据上传与解析模块 | `upload_and_parse_data()` |
| 分析字段特征 | 🔍 字段分析与极性检测模块 | `analyze_data_fields()` |
| 调整字段极性 | 🔍 字段分析与极性检测模块 | `auto_detect_and_adjust_polarity()` |
| 生成隶属度配置 | 📈 隶属度计算模块 | `generate_membership_config_template()` |
| 验证配置 | 📈 隶属度计算模块 | `validate_membership_config()` |
| 计算隶属度 | 📈 隶属度计算模块 | `calculate_membership_with_config()` |
| TOPSIS评估 | 🎯 TOPSIS综合评估模块 | `perform_topsis_comprehensive_evaluation()` |
| VIKOR评估 | ⚖️ VIKOR综合评估模块 | `perform_vikor_comprehensive_evaluation()` |
| 导出数据 | 📋 辅助工具与实用函数 | `export_resource_to_csv()` |
| 查看资源 | 📋 辅助工具与实用函数 | `list_all_resources()` |

## ⚡ 快速调试技巧

### 1. 资源ID问题
- 使用 `parse_resource_id()` 解析资源ID结构
- 使用 `get_resource_dependency_chain()` 查看资源依赖关系
- 使用 `find_latest_membership_resource()` 自动发现最新隶属度资源

### 2. 配置问题
- 使用 `validate_membership_config()` 验证配置有效性
- 使用 `generate_membership_config_template()` 获取推荐配置

### 3. 数据问题
- 使用 `analyze_data_fields()` 检查数据质量
- 使用 `export_resource_to_csv()` 导出数据查看
- 使用 `list_all_resources()` 查看可用资源

## 🎯 注意事项

1. **代码未修改**：整理后的代码完全保持原有功能，仅优化了结构和注释
2. **MVP限制**：当前为MVP阶段，禁用磁盘持久化，仅使用内存缓存
3. **资源格式**：支持纯ID和完整URI两种资源标识格式
4. **错误处理**：各函数都有完善的错误处理和验证机制
5. **模块独立**：各模块可独立使用，不必执行完整流程

## 📚 相关文档

- `MODULE_STRUCTURE.md` - 详细模块结构说明
- `main_organized.py` - 整理后的主程序文件
- 原 `main.py` - 保持不变的原始文件