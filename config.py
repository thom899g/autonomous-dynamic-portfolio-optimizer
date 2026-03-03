"""
Configuration manager for the Autonomous Dynamic Portfolio Optimizer.
Centralizes all configurable parameters with environment variable overrides.
"""
import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

@dataclass
class DataConfig:
    """Data collection and preprocessing configuration."""
    # Data sources
    DATA_SOURCES: List[str] = os.getenv("DATA_SOURCES", "yfinance,ccxt").split(",")
    
    # Asset universe
    EQUITY_TICKERS: List[str] = os.getenv("EQUITY_TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
    CRYPTO_SYMBOLS: List[str] = os.getenv("CRYPTO_SYMBOLS", "BTC/USDT,ETH/USDT,BNB/USDT").split(",")
    
    # Time periods
    TRAIN_START: str = os.getenv("TRAIN_START", "2020-01-01")
    TRAIN_END: str = os.getenv("TRAIN_END", "2023-12-31")
    VALIDATION_START: str = os.getenv("VALIDATION_START", "2023-01-01")
    VALIDATION_END: str = os.getenv("VALIDATION_END", "2023-12-31")
    
    # Feature engineering
    LOOKBACK_WINDOW: int = int(os.getenv("LOOKBACK_WINDOW", "30"))
    TECHNICAL_INDICATORS: List[str] = os.getenv(
        "TECHNICAL_INDICATORS", 
        "RSI,MACD,BBANDS,ATR,ADX"
    ).split(",")

@dataclass
class ModelConfig:
    """Neural network and RL model configuration."""
    # Neural network architecture
    HIDDEN_LAYERS: List[int] = eval(os.getenv("HIDDEN_LAYERS", "[128, 64, 32]"))
    DROPOUT_RATE: float = float(os.getenv("DROPOUT_RATE", "0.2"))
    ACTIVATION: str = os.getenv("ACTIVATION", "relu")
    
    # Reinforcement learning
    LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", "0.001"))
    GAMMA: float = float(os.getenv("GAMMA", "0.99"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "32"))
    BUFFER_SIZE: int = int(os.getenv("BUFFER_SIZE", "10000"))
    
    # Training parameters
    EPOCHS: int = int(os.getenv("EPOCHS", "100"))
    VALIDATION_FREQ: int = int(os.getenv("VALIDATION_FREQ", "10"))
    EARLY_STOPPING_PATIENCE: int = int(os.getenv("EARLY_STOPPING_PATIENCE", "20"))

@dataclass
class PortfolioConfig:
    """Portfolio management and risk configuration."""
    # Initial capital
    INITIAL_CAPITAL: float = float(os.getenv("INITIAL_CAPITAL", "100000.0"))
    
    # Risk limits
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.2"))
    MAX_LEVERAGE: float = float(os.getenv("MAX_LEVERAGE", "1.0"))
    STOP_LOSS_PCT: float = float(os.getenv("STOP_LOSS_PCT", "0.05"))
    
    # Transaction costs
    COMMISSION_RATE: float = float(os.getenv("COMMISSION_RATE", "0.001"))
    SLIPPAGE_RATE: float = float(os.getenv("SLIPPAGE_RATE", "0.0005"))
    
    # Rebalancing
    REBALANCE_FREQUENCY: str = os.getenv("REBALANCE_FREQUENCY", "1d")
    LOOKAHEAD_WINDOW: int = int(os.getenv("LOOKAHEAD_WINDOW", "5"))

@dataclass
class FirebaseConfig:
    """Firebase configuration for state management."""
    CREDENTIALS_PATH: str = os.getenv("FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json")
    PROJECT_ID: str = os.getenv("FIREBASE_PROJECT_ID", "")
    DATABASE_URL: str = os.getenv("FIREBASE_DATABASE_URL", "")
    
    # Collections
    PORTFOLIO_COLLECTION: str = os.getenv("PORTFOLIO_COLLECTION", "portfolio_states")
    TRADES_COLLECTION: str = os.getenv("TRADES_COLLECTION", "trade_history")
    MODEL_COLLECTION: str = os.getenv("MODEL_COLLECTION",