# 🔬 Open Deep Research (智谱 GLM-4.5 增强版)

<img width="1388" height="298" alt="full_diagram" src="https://github.com/user-attachments/assets/12a2371b-8be2-4219-9b48-90503eb43c69" />

深度研究已成为最受欢迎的智能代理应用之一。这是一个简单、可配置、完全开源的深度研究代理，支持多种模型提供商、搜索工具和 MCP 服务器。其性能与许多流行的深度研究代理相当（[查看 Deep Research Bench 排行榜](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)）。

<img width="817" height="666" alt="Screenshot 2025-07-13 at 11 21 12 PM" src="https://github.com/user-attachments/assets/052f2ed3-c664-4a4f-8ec2-074349dcaa3f" />

### 🔥 最新更新

**2025年1月7日**：新增对智谱 GLM-4.5 系列模型（GLM-4.5 和 GLM-4.5-Air）的支持，并增强了结构化输出的错误处理。

### 🚀 快速开始

1. 克隆仓库并激活虚拟环境：
```bash
git clone https://github.com/heishen6/GLM_OPEN_DEEP_RESEARCH.git
cd GLM_OPEN_DEEP_RESEARCH
# 使用 uv (推荐)
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 或使用 conda
conda create -n deepresearch python=3.11
conda activate deepresearch
```

2. 安装依赖：
```bash
# 使用 uv
uv sync

# 或使用 pip
pip install -e .
```

3. 设置 `.env` 文件配置环境变量（用于模型选择、搜索工具和其他配置）：
```bash
cp .env.example .env
```

**GLM-4.5 用户配置**，在 `.env` 中添加：
```env
# 智谱 AI 配置
ZHIPU_API_KEY=你的智谱API密钥
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/  # 可选，默认值

# 搜索 API 配置（至少配置一个）
TAVILY_API_KEY=你的Tavily密钥  # 推荐
```

4. 启动 LangGraph 服务器：

```bash
# 使用 uvx 启动（推荐）
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking

# 或使用 langgraph 命令
langgraph dev --port 2024
```

这将在浏览器中打开 LangGraph Studio UI。

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API 文档: http://127.0.0.1:2024/docs
```

在 `messages` 输入框中输入问题并点击 `Submit`。可以在 "Manage Assistants" 标签页中选择不同的配置。

### ⚙️ 配置说明

#### 大语言模型 :brain:

Open Deep Research 通过 [init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/) 支持多种 LLM 提供商。

**新增：GLM-4.5 支持**
- `glm-4.5:latest` - 完整版模型，适用于复杂研究任务（128K 上下文）
- `glm-4.5-air:latest` - 轻量版模型，适用于快速处理（128K 上下文）

系统在不同任务中使用 LLM。详见 [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) 文件中的模型字段。可通过 LangGraph Studio UI 访问配置。

- **摘要生成** (默认: `openai:gpt-4o-mini`): 总结搜索 API 结果
- **研究执行** (默认: `openai:gpt-4o`): 驱动搜索代理
- **内容压缩** (默认: `openai:gpt-4o`): 压缩研究发现
- **最终报告** (默认: `openai:gpt-4o`): 撰写最终报告

> 注意：所选模型需要支持[结构化输出](https://python.langchain.com/docs/integrations/chat/)和[工具调用](https://python.langchain.com/docs/how_to/tool_calling/)。

> **GLM-4.5 注意事项**：这些模型对结构化输出的支持有限。请在配置中设置 `allow_clarification=False` 以避免错误。

#### 搜索 API :mag:

Open Deep Research 支持多种搜索工具。默认使用 [Tavily](https://www.tavily.com/) 搜索 API。完全兼容 MCP，并支持 Anthropic 和 OpenAI 的原生网络搜索。详见 [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) 文件中的 `search_api` 和 `mcp_config` 字段。

#### 其他配置

查看 [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) 中的其他字段，了解各种自定义 Open Deep Research 行为的设置。

### 🎯 GLM-4.5 专项修改

#### 修改的文件

1. **`src/open_deep_research/utils.py`**
   - 添加 GLM-4.5 模型映射，支持 128K token 限制
   - 实现 `get_base_url_for_model()` 函数，支持自定义 API 端点
   - 添加 ZHIPU_API_KEY 环境变量支持

2. **`src/open_deep_research/deep_researcher.py`**
   - 为结构化输出失败添加 try-catch 块
   - 实现结构化输出失败时的自动降级机制
   - 修复流式响应断言错误

3. **`.env.example`**
   - 添加智谱 AI 配置示例

#### GLM-4.5 配置示例

```python
config = {
    "configurable": {
        # 使用 GLM-4.5 进行研究
        "research_model": "glm-4.5:latest",
        "research_model_max_tokens": 4000,
        
        # 使用 GLM-4.5-Air 进行压缩（更快）
        "compression_model": "glm-4.5-air:latest",
        "compression_model_max_tokens": 2000,
        
        # 使用 GLM-4.5 生成最终报告
        "final_report_model": "glm-4.5:latest",
        "final_report_model_max_tokens": 8000,
        
        # 重要：禁用 GLM 模型的澄清功能
        "allow_clarification": False,
        
        # 其他设置
        "max_concurrent_research_units": 3,  # 并发研究单元数
        "max_researcher_iterations": 5,      # 研究迭代次数
    }
}
```

### 💡 使用示例

#### Python API 调用

```python
import asyncio
from open_deep_research import deep_researcher

async def research_with_glm():
    """使用 GLM-4.5 进行深度研究"""
    
    config = {
        "configurable": {
            # 模型配置
            "research_model": "glm-4.5:latest",
            "compression_model": "glm-4.5-air:latest",
            "final_report_model": "glm-4.5:latest",
            
            # 重要配置
            "allow_clarification": False,
            
            # 搜索配置
            "search_api": "tavily",
            "tavily_max_results": 10,
        }
    }
    
    # 执行研究
    result = await deep_researcher.ainvoke({
        "messages": [{"role": "human", "content": "研究2024年人工智能的最新进展"}]
    }, config)
    
    print(result["final_report"])

# 运行
asyncio.run(research_with_glm())
```

#### 混合模型配置（优化成本）

```python
config = {
    "configurable": {
        # GLM-4.5 用于主要研究（中文能力强）
        "research_model": "glm-4.5:latest",
        
        # GLM-4.5-Air 用于压缩（速度快、成本低）
        "compression_model": "glm-4.5-air:latest",
        
        # 可以混用其他模型
        "final_report_model": "gpt-4o",  # 如需更好的格式化
    }
}
```

### ⚠️ GLM-4.5 已知限制

1. **结构化输出**：GLM-4.5 对 LangChain 的结构化输出支持有限。系统会在错误发生时自动降级到非结构化模式。

2. **澄清功能**：使用 GLM 模型时应禁用（`allow_clarification=False`）以避免断言错误。

3. **Token 计算**：GLM 模型的 token 计算方式与 OpenAI 不同，请相应调整 max_tokens 参数。

### 🛠️ 故障排除

**问：遇到 "AssertionError" 错误**
答：这通常是结构化输出不兼容导致的。确保 `allow_clarification` 设置为 `False`。

**问：API 调用失败**
答：检查 `.env` 中的 `ZHIPU_API_KEY` 是否正确配置。

**问：研究结果不完整**
答：尝试增加 `max_researcher_iterations` 和 `max_react_tool_calls` 的值。

**问：如何优化成本？**
答：使用 GLM-4.5-Air 进行压缩和摘要任务，GLM-4.5 仅用于核心研究。

### 📊 性能对比

| 指标 | GLM-4.5 | GLM-4.5-Air | GPT-4 |
|------|---------|-------------|-------|
| 研究深度 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 响应速度 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 成本效益 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 中文支持 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 上下文长度 | 128K | 128K | 128K |

### 📈 评估

Open Deep Research 配置用于 [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard) 评估。该基准包含 100 个博士级研究任务（50 个英文，50 个中文），由 22 个领域的专家精心设计，反映真实的深度研究需求。

#### 使用方法

> 警告：运行全部 100 个示例可能花费 ~$20-$100，具体取决于模型选择。

数据集可通过 [LangSmith 链接](https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d) 获取。运行评估：

```bash
# 在 LangSmith 数据集上运行综合评估
python tests/run_evaluate.py
```

### 🚀 部署和使用

#### LangGraph Studio

按照[快速开始](#-快速开始)在本地启动 LangGraph 服务器，并在 LangGraph Studio 上测试代理。

#### 云端部署
 
可以轻松部署到 [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options)。

### 🏛️ 旧版实现

`src/legacy/` 文件夹包含两个早期实现，提供了自动化研究的替代方法。虽然性能不如当前实现，但提供了理解深度研究不同方法的替代思路。

## 📜 许可证

本项目基于 LangChain 的原始 [open_deep_research](https://github.com/langchain-ai/open_deep_research) 项目，并保持与原始许可证的兼容性。

## 🙏 致谢

- 原始项目：[LangChain](https://github.com/langchain-ai)
- GLM-4.5 模型：[智谱 AI](https://www.zhipuai.cn/)
- 中文社区的支持和贡献

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 [GitHub Issue](https://github.com/heishen6/GLM_OPEN_DEEP_RESEARCH/issues)
- 加入讨论组了解更多使用技巧
