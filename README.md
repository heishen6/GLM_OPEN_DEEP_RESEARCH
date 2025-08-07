# Open Deep Research - 智谱 GLM-4.5 增强版

[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-blue)](https://github.com/langchain-ai/langgraph)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green)](https://www.python.org/)
[![GLM-4.5](https://img.shields.io/badge/Supports-GLM--4.5-orange)](https://open.bigmodel.cn/)

## 📌 项目简介

Open Deep Research 是一个基于 LangGraph 的可配置深度研究代理，能够自动进行深入的网络研究并生成全面的报告。本增强版特别优化支持了**智谱 GLM-4.5** 系列模型，包括：

- **GLM-4.5** - 完整版模型，适用于复杂研究任务
- **GLM-4.5-Air** - 轻量版模型，适用于快速研究和低成本场景

## 🆕 GLM-4.5 支持改动

### 核心改动

1. **模型配置支持**
   - ✅ 支持通过环境变量配置智谱 API 密钥 (`ZHIPU_API_KEY`)
   - ✅ 支持自定义 API Base URL 配置
   - ✅ 添加 GLM-4.5 系列模型的 token 限制配置

2. **兼容性优化**
   - ✅ 修复结构化输出兼容性问题（GLM-4.5 部分支持结构化输出）
   - ✅ 添加错误处理机制，当结构化输出失败时自动降级处理
   - ✅ 优化流式响应处理，避免断言错误

3. **功能增强**
   - ✅ 支持通过 LangChain 的 `init_chat_model` 统一接口调用
   - ✅ 保留原有的多模型支持（OpenAI、Anthropic、Google、Groq 等）
   - ✅ 支持混合使用不同模型（研究、压缩、报告生成可使用不同模型）

### 文件改动列表

```
修改的文件：
├── .env                          # 添加智谱 API 配置
├── src/open_deep_research/
│   ├── utils.py                  # 添加 GLM-4.5 模型支持和 base_url 处理
│   └── deep_researcher.py        # 添加结构化输出错误处理
└── test_glm.py                   # GLM-4.5 测试脚本（新增）
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/langchain-ai/open_deep_research.git
cd open_deep_research

# 创建 conda 环境（推荐）
conda create -n deepresearch python=3.11
conda activate deepresearch

# 安装依赖
pip install -e .
```

### 2. 配置智谱 API

在 `.env` 文件中配置：

```env
# 智谱 AI 配置
ZHIPU_API_KEY=your_zhipu_api_key_here
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/  # 可选，默认值

# 搜索 API 配置（至少配置一个）
TAVILY_API_KEY=your_tavily_api_key_here  # 推荐
# 或使用其他搜索 API
# LINKUP_API_KEY=your_linkup_api_key
# EXA_API_KEY=your_exa_api_key
```

### 3. 启动服务

```bash
# 启动 LangGraph 开发服务器
langgraph dev --port 2024

# 服务启动后访问:
# - API: http://127.0.0.1:2024
# - Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
# - API 文档: http://127.0.0.1:2024/docs
```

## 💡 使用示例

### Python API 调用

```python
import asyncio
from open_deep_research import deep_researcher

async def research_with_glm():
    """使用 GLM-4.5 进行深度研究"""
    
    config = {
        "configurable": {
            # 使用 GLM-4.5 作为研究模型
            "research_model": "glm-4.5:latest",
            "research_model_max_tokens": 4000,
            
            # 使用 GLM-4.5-Air 作为压缩模型（更快速）
            "compression_model": "glm-4.5-air:latest",
            "compression_model_max_tokens": 2000,
            
            # 使用 GLM-4.5 生成最终报告
            "final_report_model": "glm-4.5:latest",
            "final_report_model_max_tokens": 8000,
            
            # 其他配置
            "allow_clarification": False,  # GLM-4.5 结构化输出支持有限
            "max_concurrent_research_units": 3,
            "max_researcher_iterations": 5,
        }
    }
    
    # 执行研究
    result = await deep_researcher.ainvoke({
        "messages": [{"role": "human", "content": "研究一下2024年人工智能的最新进展"}]
    }, config)
    
    print(result["final_report"])

# 运行研究
asyncio.run(research_with_glm())
```

### 混合模型配置

```python
config = {
    "configurable": {
        # 使用 GLM-4.5 进行研究（强大的理解能力）
        "research_model": "glm-4.5:latest",
        
        # 使用 GLM-4.5-Air 进行压缩（快速且经济）
        "compression_model": "glm-4.5-air:latest",
        
        # 使用 GPT-4 生成最终报告（如果需要更好的格式化）
        "final_report_model": "gpt-4o",
        
        # 搜索配置
        "search_api": "tavily",
        "tavily_max_results": 10,
    }
}
```

## 🔧 配置选项

### 模型配置

| 参数 | 说明 | GLM-4.5 推荐值 |
|------|------|---------------|
| `research_model` | 研究执行模型 | `glm-4.5:latest` |
| `research_model_max_tokens` | 研究模型最大 token | 4000 |
| `compression_model` | 信息压缩模型 | `glm-4.5-air:latest` |
| `compression_model_max_tokens` | 压缩模型最大 token | 2000 |
| `final_report_model` | 报告生成模型 | `glm-4.5:latest` |
| `final_report_model_max_tokens` | 报告模型最大 token | 8000 |

### GLM-4.5 模型限制

| 模型 | 上下文长度 | 适用场景 |
|------|------------|----------|
| GLM-4.5 | 128K tokens | 深度研究、复杂推理、长文本生成 |
| GLM-4.5-Air | 128K tokens | 快速研究、信息提取、低成本场景 |

### 研究参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `allow_clarification` | 是否启用用户澄清 | false（GLM-4.5 建议关闭） |
| `max_concurrent_research_units` | 并发研究单元数 | 3 |
| `max_researcher_iterations` | 研究迭代次数 | 5 |
| `max_react_tool_calls` | 工具调用次数限制 | 10 |

## ⚠️ 已知限制

1. **结构化输出支持有限**
   - GLM-4.5 对 LangChain 的结构化输出支持不完整
   - 建议关闭 `allow_clarification` 选项
   - 系统会自动降级处理结构化输出失败的情况

2. **流式响应兼容性**
   - 某些情况下流式响应可能出现断言错误
   - 已添加错误处理机制自动恢复

3. **Token 计算**
   - GLM-4.5 的 token 计算方式与 OpenAI 略有不同
   - 建议适当调整 max_tokens 参数

## 🛠️ 故障排除

### 常见问题

**Q: 遇到 "AssertionError" 错误**
- A: 这通常是结构化输出不兼容导致的。确保 `allow_clarification` 设置为 `false`

**Q: API 调用失败**
- A: 检查 `.env` 中的 `ZHIPU_API_KEY` 是否正确配置

**Q: 研究结果不完整**
- A: 尝试增加 `max_researcher_iterations` 和 `max_react_tool_calls` 的值

### 测试脚本

运行测试脚本验证 GLM-4.5 配置：

```bash
python test_glm.py
```

## 📊 性能对比

| 指标 | GLM-4.5 | GLM-4.5-Air | GPT-4 |
|------|---------|-------------|-------|
| 研究深度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 响应速度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 成本效益 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 中文支持 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！特别欢迎：

- 改进 GLM-4.5 的结构化输出支持
- 优化 token 使用效率
- 添加更多中文研究场景的示例

## 📜 许可证

本项目基于原始 [open_deep_research](https://github.com/langchain-ai/open_deep_research) 项目，遵循相同的开源许可证。

## 🙏 致谢

- 感谢 [LangChain](https://github.com/langchain-ai) 团队的原始项目
- 感谢 [智谱 AI](https://www.zhipuai.cn/) 提供强大的 GLM-4.5 模型
- 感谢所有贡献者的支持

---

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 [GitHub Issue](https://github.com/your-username/open_deep_research/issues)
- 发送邮件至：your-email@example.com

---

**注意**: 本项目仍在积极开发中，API 和功能可能会有变动。建议在生产环境使用前进行充分测试。
