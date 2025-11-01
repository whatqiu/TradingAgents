import os
from openai import OpenAI
try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = None
from .config import get_config


def get_client():
    """获取配置好的客户端"""
    config = get_config()

    # 根据provider选择不同的API key
    if config["llm_provider"] == "zhipuai" or config["llm_provider"] == "智谱ai (glm)":
        api_key = os.getenv("GLM_API_KEY")
        if not api_key:
            raise ValueError("GLM_API_KEY环境变量未设置，请在.env文件中设置GLM_API_KEY")

        if ZhipuAI is not None:
            # 使用官方ZhipuAI SDK
            return ZhipuAI(api_key=api_key)
        else:
            # 如果官方SDK不可用，回退到OpenAI兼容客户端
            print("Warning: ZhipuAI SDK not available, falling back to OpenAI client")
            return OpenAI(base_url=config["backend_url"], api_key=api_key)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY环境变量未设置")
        return OpenAI(base_url=config["backend_url"], api_key=api_key)


def create_chat_completion(client, model, messages, temperature=0.7, max_tokens=4096, top_p=1):
    """创建聊天完成请求，支持不同的API格式"""
    config = get_config()
    provider = config["llm_provider"].lower()

    # 检查客户端类型
    client_type = type(client).__name__

    if (provider == "zhipuai" or provider == "智谱ai (glm)") and client_type == "ZhipuAI":
        # 使用官方ZhipuAI SDK
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message
    else:
        # 使用OpenAI兼容格式
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return response.choices[0].message


def get_stock_news_openai(query, start_date, end_date):
    config = get_config()
    client = get_client()

    system_prompt = f"请搜索{query}在{start_date}到{end_date}期间的社交媒体讨论。请确保只获取该时间段内发布的数据。"

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"请提供关于{query}从{start_date}到{end_date}期间的社交媒体讨论和分析。"
        }
    ]

    response = create_chat_completion(
        client=client,
        model=config["quick_think_llm"],
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        top_p=1,
    )

    return response.content


def get_global_news_openai(curr_date, look_back_days=7, limit=5):
    config = get_config()
    client = get_client()

    system_prompt = f"请搜索从{look_back_days}天前{curr_date}到{curr_date}期间对交易有信息价值的全球或宏观经济新闻。请确保只获取该时间段内发布的数据，限制结果为{limit}篇文章。"

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"请提供从{look_back_days}天前到{curr_date}期间对交易有重要影响的全球宏观经济新闻，最多{limit}篇。"
        }
    ]

    response = create_chat_completion(
        client=client,
        model=config["quick_think_llm"],
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        top_p=1,
    )

    return response.content


def get_fundamentals_openai(ticker, curr_date):
    config = get_config()
    client = get_client()

    system_prompt = f"请搜索{ticker}在{curr_date}所在月份及前一个月的基本面分析讨论。请确保只获取该时间段内发布的数据。请以表格形式列出，包含PE/PS/现金流等指标。"

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"请提供{ticker}在{curr_date}期间的基本面分析数据，包括PE比率、PS比率、现金流等关键财务指标。"
        }
    ]

    response = create_chat_completion(
        client=client,
        model=config["quick_think_llm"],
        messages=messages,
        temperature=0.7,
        max_tokens=4096,
        top_p=1,
    )

    return response.content