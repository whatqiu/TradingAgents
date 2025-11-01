from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.logging import setup_logging_from_env, get_logger

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logging system
setup_logging_from_env()
logger = get_logger(__name__)
logger.info("TradingAgents application starting")

# Create a custom config
config = DEFAULT_CONFIG.copy()
# 使用默认配置中的GLM-4.6模型
config["max_debate_rounds"] = 1  # Increase debate rounds

# Configure data vendors (default uses yfinance and alpha_vantage)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: yfinance, alpha_vantage, local
    "technical_indicators": "yfinance",      # Options: yfinance, alpha_vantage, local
    "fundamental_data": "alpha_vantage",     # Options: openai, alpha_vantage, local
    "news_data": "alpha_vantage",            # Options: openai, alpha_vantage, google, local
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
