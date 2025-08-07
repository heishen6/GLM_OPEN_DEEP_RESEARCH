# 🔬 Open Deep Research

<img width="1388" height="298" alt="full_diagram" src="https://github.com/user-attachments/assets/12a2371b-8be2-4219-9b48-90503eb43c69" />

Deep research has broken out as one of the most popular agent applications. This is a simple, configurable, fully open source deep research agent that works across many model providers, search tools, and MCP servers. It's performance is on par with many popular deep research agents ([see Deep Research Bench leaderboard](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard)).

<img width="817" height="666" alt="Screenshot 2025-07-13 at 11 21 12 PM" src="https://github.com/user-attachments/assets/052f2ed3-c664-4a4f-8ec2-074349dcaa3f" />

### 🔥 Recent Updates

**January 7, 2025**: Added support for Zhipu GLM-4.5 series models (GLM-4.5 and GLM-4.5-Air) with enhanced error handling for structured outputs.

### 🚀 Quickstart

1. Clone the repository and activate a virtual environment:
```bash
git clone https://github.com/heishen6/GLM_OPEN_DEEP_RESEARCH.git
cd GLM_OPEN_DEEP_RESEARCH
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
uv sync
# or
pip install -e .
```

3. Set up your `.env` file to customize the environment variables (for model selection, search tools, and other configuration settings):
```bash
cp .env.example .env
```

**For GLM-4.5 users**, add these to your `.env`:
```env
# Zhipu AI Configuration
ZHIPU_API_KEY=your_zhipu_api_key_here
ZHIPU_BASE_URL=https://open.bigmodel.cn/api/paas/v4/  # Optional
```

4. Launch agent with the LangGraph server locally:

```bash
# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --allow-blocking
```

This will open the LangGraph Studio UI in your browser.

```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Ask a question in the `messages` input field and click `Submit`. Select different configuration in the "Manage Assistants" tab.

### ⚙️ Configurations

#### LLM :brain:

Open Deep Research supports a wide range of LLM providers via the [init_chat_model() API](https://python.langchain.com/docs/how_to/chat_models_universal_init/). 

**NEW: GLM-4.5 Support**
- `glm-4.5:latest` - Full model for complex research tasks (128K context)
- `glm-4.5-air:latest` - Lightweight model for fast processing (128K context)

It uses LLMs for a few different tasks. See the below model fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) file for more details. This can be accessed via the LangGraph Studio UI. 

- **Summarization** (default: `openai:gpt-4o-mini`): Summarizes search API results
- **Research** (default: `openai:gpt-4o`): Power the search agent
- **Compression** (default: `openai:gpt-4o`): Compresses research findings
- **Final Report Model** (default: `openai:gpt-4o`): Write the final report

> Note: the selected model will need to support [structured outputs](https://python.langchain.com/docs/integrations/chat/) and [tool calling](https://python.langchain.com/docs/how_to/tool_calling/).

> **GLM-4.5 Note**: These models have limited structured output support. Set `allow_clarification=False` in configuration to avoid errors.

#### Search API :mag:

Open Deep Research supports a wide range of search tools. By default it uses the [Tavily](https://www.tavily.com/) search API. Has full MCP compatibility and work native web search for Anthropic and OpenAI. See the `search_api` and `mcp_config` fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) file for more details. This can be accessed via the LangGraph Studio UI. 

#### Other 

See the fields in the [configuration.py](https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/configuration.py) for various other settings to customize the behavior of Open Deep Research. 

### GLM-4.5 Specific Modifications

#### Files Modified

1. **`src/open_deep_research/utils.py`**
   - Added GLM-4.5 model mappings with 128K token limits
   - Implemented `get_base_url_for_model()` function for custom API endpoints
   - Added ZHIPU_API_KEY environment variable support

2. **`src/open_deep_research/deep_researcher.py`**
   - Added try-catch blocks for structured output failures
   - Implemented automatic fallback mechanism when structured output fails
   - Fixed streaming response assertion errors

3. **`.env.example`**
   - Added Zhipu AI configuration examples

#### Configuration Example for GLM-4.5

```python
config = {
    "configurable": {
        # Use GLM-4.5 for research
        "research_model": "glm-4.5:latest",
        "research_model_max_tokens": 4000,
        
        # Use GLM-4.5-Air for compression (faster)
        "compression_model": "glm-4.5-air:latest",
        "compression_model_max_tokens": 2000,
        
        # Use GLM-4.5 for final report
        "final_report_model": "glm-4.5:latest",
        "final_report_model_max_tokens": 8000,
        
        # Important: Disable clarification for GLM models
        "allow_clarification": False,
        
        # Other settings
        "max_concurrent_research_units": 3,
        "max_researcher_iterations": 5,
    }
}
```

### Known Limitations with GLM-4.5

1. **Structured Output**: GLM-4.5 has limited support for LangChain's structured output. The system will automatically fallback to non-structured mode if errors occur.

2. **Clarification Feature**: Should be disabled (`allow_clarification=False`) when using GLM models to avoid assertion errors.

3. **Token Calculation**: GLM models calculate tokens differently than OpenAI. Adjust max_tokens parameters accordingly.

### 📊 Evaluation

Open Deep Research is configured for evaluation with [Deep Research Bench](https://huggingface.co/spaces/Ayanami0730/DeepResearch-Leaderboard). This benchmark has 100 PhD-level research tasks (50 English, 50 Chinese), crafted by domain experts across 22 fields (e.g., Science & Tech, Business & Finance) to mirror real-world deep-research needs.

#### Usage

> Warning: Running across the 100 examples can cost ~$20-$100 depending on the model selection.

The dataset is available on [LangSmith via this link](https://smith.langchain.com/public/c5e7a6ad-fdba-478c-88e6-3a388459ce8b/d). To kick off evaluation, run the following command:

```bash
# Run comprehensive evaluation on LangSmith datasets
python tests/run_evaluate.py
```

### 🚀 Deployments and Usage

#### LangGraph Studio

Follow the [quickstart](#-quickstart) to start LangGraph server locally and test the agent out on LangGraph Studio.

#### Hosted deployment
 
You can easily deploy to [LangGraph Platform](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

### Legacy Implementations 🏛️

The `src/legacy/` folder contains two earlier implementations that provide alternative approaches to automated research. They are less performant than the current implementation, but provide alternative ideas understanding the different approaches to deep research.

## License

This project is based on the original [open_deep_research](https://github.com/langchain-ai/open_deep_research) by LangChain and maintains compatibility with the original license.

## Acknowledgments

- Original project by [LangChain](https://github.com/langchain-ai)
- GLM-4.5 models by [Zhipu AI](https://www.zhipuai.cn/)
