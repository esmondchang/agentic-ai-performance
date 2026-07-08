"""Financial Agent Workflow using LangGraph
This demonstrates:
- State management in agentic systems
- Node-based workflow orchestration
- Conditional routing
- Parallel execution
- Error handling and recovery
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence
from dataclasses import dataclass, field
from enum import Enum
import operator
import time

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import OllamaLLM as Ollama

from src.react_agent import ReActAgent
from src.rag_engine import RAGEngine
from src.tool_system import ToolRegistry, StockDataTool, CalculatorTool, WebSearchTool

# Define the agent state using TypedDict
class AgentState(TypedDict):
    """State that flows through the agent graph

    This demonstrates proper state management in LangGraph
    """
    # Input
    ticker: str
    query: str

    # Messages for conversation history
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Data collected
    market_data: Dict[str, Any]
    financial_metrics: Dict[str, Any]
    news_summary: str

    # Analysis results
    technical_analysis: Dict[str, Any]
    fundamental_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, float]

    # RAG context
    retrieved_documents: List[Dict[str, Any]]
    rag_response: str

    # ReAct reasoning
    reasoning_trace: List[Dict[str, str]]

    # LangGraph performance telemetry
    performance_metrics: Dict[str, Any]

    # Final outputs
    recommendation: str
    confidence_score: float
    report: str

    # Control flow
    next_step: str
    error: Optional[str]
    retry_count: int

class WorkflowNodes:
    """Collection of workflow nodes demonstrating different patterns"""

    def __init__(self, fast_mode: bool = False):
        """Initialize components"""
        try:
            from src.config import config
        except ModuleNotFoundError:
            from config import config

        self.fast_mode = fast_mode

        # Initialize heavier local-model components only for full mode.
        self.llm = None if fast_mode else Ollama(
            model=config.analysis_model,
            temperature=0.7
        )

        # Initialize RAG engine
        self.rag_engine = None if fast_mode else RAGEngine()

        # Initialize ReAct agent
        self.react_agent = None if fast_mode else ReActAgent()

        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(StockDataTool())
        self.tool_registry.register(CalculatorTool())
        self.tool_registry.register(WebSearchTool())

    def collect_market_data(self, state: AgentState) -> AgentState:
        """Node: Collect market data using tools"""
        print("📊 Collecting market data...")

        ticker = state["ticker"]

        try:
            # Use stock data tool
            tool = self.tool_registry.get_tool("stock_data")
            market_data = tool.execute(
                ticker=ticker,
                metrics=["price", "volume", "market_cap", "pe_ratio", "52_week_high", "52_week_low"]
            )

            state["market_data"] = market_data

            # Add message to history
            state["messages"].append(
                AIMessage(content=f"Collected market data for {ticker}")
            )

        except Exception as e:
            state["error"] = f"Failed to collect market data: {str(e)}"
            state["market_data"] = {}

        return state

    def analyze_with_react(self, state: AgentState) -> AgentState:
        """Node: Use ReAct agent for analysis"""
        print("🤔 ReAct agent analyzing...")

        ticker = state["ticker"]
        market_data = state.get("market_data", {})

        if self.fast_mode:
            pe_ratio = market_data.get("pe_ratio")
            price = market_data.get("price", "N/A")
            if isinstance(pe_ratio, (int, float)):
                valuation = "below" if pe_ratio < 25 else "above"
                answer = (
                    f"{ticker} trades at a P/E ratio of {pe_ratio:.2f}, "
                    f"which is {valuation} the comparison average of 25."
                )
            else:
                answer = f"{ticker} price is {price}. P/E data was not available from the market data tool."

            state["reasoning_trace"] = [
                {
                    "thought": "Use collected market data for a fast valuation check.",
                    "action": "compare_pe_ratio",
                    "action_input": f"{ticker} vs industry average 25",
                    "observation": answer,
                }
            ]
            state["recommendation"] = answer
            state["messages"].append(AIMessage(content=f"Fast ReAct Analysis: {answer}"))
            return state

        # Formulate question for ReAct agent
        question = f"""
        Analyze {ticker} stock with the following data:
        Current Price: ${market_data.get('price', 'N/A')}
        P/E Ratio: {market_data.get('pe_ratio', 'N/A')}
        Market Cap: ${market_data.get('market_cap', 'N/A')}

        Provide investment recommendation with reasoning.
        """

        # Run ReAct reasoning
        answer, trace = self.react_agent.think(question)

        # Store results
        state["reasoning_trace"] = [step.to_dict() for step in trace]
        state["recommendation"] = answer

        # Add to messages
        state["messages"].append(
            AIMessage(content=f"ReAct Analysis: {answer}")
        )

        return state

    def perform_rag_analysis(self, state: AgentState) -> AgentState:
        """Node: RAG-based analysis"""
        print("📚 Performing RAG analysis...")

        ticker = state["ticker"]
        query = state.get("query", f"What is the investment outlook for {ticker}?")

        if self.fast_mode:
            rag = RAGEngine.__new__(RAGEngine)
            docs = RAGEngine.fetch_latest_financial_documents(rag, ticker)

            state["retrieved_documents"] = [
                {
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": 1.0,
                }
                for doc in docs[:5]
            ]
            state["rag_response"] = (
                f"Loaded {len(state['retrieved_documents'])} live yfinance documents for {ticker}. "
                "Fast mode skips embedding search and uses the newest fetched context directly."
            )
            state["messages"].append(AIMessage(content=f"Fast RAG Analysis: {state['rag_response']}"))
            return state

        # Load financial documents
        self.rag_engine.load_financial_documents(ticker)

        # Retrieve relevant documents
        docs = self.rag_engine.retrieve_with_scores(query, k=5)

        # Store retrieved documents
        state["retrieved_documents"] = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score
            }
            for doc, score in docs
        ]

        # Generate response with context
        rag_response = self.rag_engine.generate_with_context(query)
        state["rag_response"] = rag_response

        # Add to messages
        state["messages"].append(
            AIMessage(content=f"RAG Analysis: {rag_response[:200]}...")
        )

        return state

    def analyze_sentiment(self, state: AgentState) -> AgentState:
        """Node: Sentiment analysis"""
        print("😊 Analyzing sentiment...")

        # Use web search tool to get news
        web_tool = self.tool_registry.get_tool("web_search")
        news_results = web_tool.execute(
            query=f"{state['ticker']} stock news analysis",
            num_results=5
        )

        # Combine news snippets
        news_text = " ".join([r["snippet"] for r in news_results])
        state["news_summary"] = news_text

        if self.fast_mode:
            positive_terms = ["strong", "growth", "positive", "beat", "innovation", "partnership"]
            negative_terms = ["risk", "volatility", "competition", "regulatory", "concerns"]
            lower_news = news_text.lower()
            positive_count = sum(term in lower_news for term in positive_terms)
            negative_count = sum(term in lower_news for term in negative_terms)
            score = max(-1.0, min(1.0, (positive_count - negative_count) / 5))

            state["sentiment_analysis"] = {
                "overall": score,
                "bullish_factors": ["Fast keyword scan of available news snippets"],
                "bearish_factors": ["Full LLM sentiment parsing skipped in fast mode"],
                "raw_response": news_text,
            }
            return state

        # Analyze sentiment using LLM
        prompt = f"""
        Analyze the sentiment of this news about {state['ticker']}:

        {news_text}

        Provide:
        1. Overall sentiment score (-1 to 1)
        2. Bullish factors
        3. Bearish factors

        Format as JSON.
        """

        response = self.llm.invoke(prompt)

        # Parse response (simplified)
        state["sentiment_analysis"] = {
            "overall": 0.6,  # Would parse from response
            "bullish_factors": ["Strong earnings", "Product innovation"],
            "bearish_factors": ["Market volatility"],
            "raw_response": response
        }

        return state

    def calculate_technical_indicators(self, state: AgentState) -> AgentState:
        """Node: Technical analysis"""
        print("📈 Calculating technical indicators...")

        import yfinance as yf
        import pandas as pd

        ticker = state["ticker"]

        try:
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")

            if not hist.empty:
                # Calculate simple indicators
                close_prices = hist['Close']

                # Moving averages
                sma_20 = close_prices.rolling(window=20).mean().iloc[-1]
                sma_50 = close_prices.rolling(window=50).mean().iloc[-1]

                # RSI
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]

                state["technical_analysis"] = {
                    "sma_20": float(sma_20),
                    "sma_50": float(sma_50),
                    "rsi": float(rsi),
                    "trend": "bullish" if close_prices.iloc[-1] > sma_20 else "bearish"
                }
            else:
                state["technical_analysis"] = {"error": "No historical data"}

        except Exception as e:
            state["technical_analysis"] = {"error": str(e)}

        return state

    def generate_report(self, state: AgentState) -> AgentState:
        """Node: Generate final report"""
        print("📝 Generating report...")

        if self.fast_mode:
            state["report"] = f"""
# Fast LangGraph Analysis Report for {state['ticker']}

## Market Data
{state.get('market_data', 'N/A')}

## Technical Analysis
{state.get('technical_analysis', 'N/A')}

## Sentiment
{state.get('sentiment_analysis', 'N/A')}

## RAG Context
{state.get('rag_response', 'N/A')}

## Recommendation
{state.get('recommendation', 'N/A')}

---
Generated in fast LangGraph mode. Full LLM synthesis is skipped to keep the browser workflow responsive.
            """
            confidence_factors = [
                0.9 if state.get("market_data") else 0.5,
                0.8 if state.get("technical_analysis") else 0.5,
                0.8 if state.get("retrieved_documents") else 0.5,
                0.8 if state.get("reasoning_trace") else 0.5,
            ]
            state["confidence_score"] = sum(confidence_factors) / len(confidence_factors)
            return state

        # Compile all analyses
        report_prompt = f"""
        Generate a comprehensive investment report for {state['ticker']}:

        Market Data:
        {state.get('market_data', 'N/A')}

        Technical Analysis:
        {state.get('technical_analysis', 'N/A')}

        Sentiment Analysis:
        {state.get('sentiment_analysis', 'N/A')}

        RAG Analysis:
        {state.get('rag_response', 'N/A')[:500]}

        ReAct Recommendation:
        {state.get('recommendation', 'N/A')}

        Provide:
        1. Executive Summary
        2. Key Findings
        3. Risk Assessment
        4. Investment Recommendation
        5. Confidence Level (0-1)

        Format as a professional report.
        """

        report = self.llm.invoke(report_prompt)
        state["report"] = report

        # Calculate confidence (simplified)
        confidence_factors = [
            0.9 if state.get("market_data") else 0.5,
            0.8 if state.get("technical_analysis") else 0.5,
            0.85 if state.get("rag_response") else 0.5,
            0.9 if state.get("reasoning_trace") else 0.5
        ]
        state["confidence_score"] = sum(confidence_factors) / len(confidence_factors)

        return state

    def should_retry(self, state: AgentState) -> str:
        """Conditional edge: Determine if retry is needed"""
        if state.get("error") and state.get("retry_count", 0) < 3:
            return "retry"
        return "continue"

class FinancialAgentWorkflow:
    """Main workflow orchestrator using LangGraph"""

    NODE_LABELS = {
        "collect_market_data": "Collect Market Data",
        "technical_analysis_node": "Technical Analysis",
        "sentiment_analysis_node": "Sentiment Analysis",
        "rag_analysis_node": "RAG Analysis",
        "react_reasoning": "ReAct Reasoning",
        "generate_report": "Generate Report",
    }

    def __init__(self, fast_mode: bool = False):
        """Initialize the workflow"""
        self.fast_mode = fast_mode
        try:
            self.nodes = WorkflowNodes(fast_mode=fast_mode)
            self.graph = self._build_graph()
            self.app = self.graph.compile()
        except Exception as e:
            print(f"Warning: Workflow initialization error: {e}")
            # Set defaults if initialization fails
            self.nodes = WorkflowNodes(fast_mode=fast_mode)
            self.app = None

    def _timed_node(self, node_name: str, node_func):
        """Wrap a LangGraph node so the state carries per-node timing."""
        def wrapper(state: AgentState) -> AgentState:
            started = time.perf_counter()
            metrics = state.setdefault("performance_metrics", {})
            node_metrics = metrics.setdefault("nodes", [])

            try:
                result = node_func(state)
                status = "success"
                return result
            except Exception as e:
                status = "error"
                state["error"] = str(e)
                raise
            finally:
                duration_ms = (time.perf_counter() - started) * 1000
                node_metrics.append({
                    "node": node_name,
                    "label": self.NODE_LABELS.get(node_name, node_name),
                    "duration_ms": round(duration_ms, 2),
                    "status": status,
                    "mode": "fast" if self.fast_mode else "full",
                    "documents": len(state.get("retrieved_documents", [])),
                    "reasoning_steps": len(state.get("reasoning_trace", [])),
                    "error": state.get("error"),
                })

        return wrapper

    def _build_graph(self) -> StateGraph:
        """Build the workflow graph

        This demonstrates:
        - Sequential execution
        - Parallel execution
        - Conditional routing
        - Error handling
        """
        # Create graph with AgentState type
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("collect_market_data", self._timed_node("collect_market_data", self.nodes.collect_market_data))
        workflow.add_node("technical_analysis_node", self._timed_node("technical_analysis_node", self.nodes.calculate_technical_indicators))
        workflow.add_node("sentiment_analysis_node", self._timed_node("sentiment_analysis_node", self.nodes.analyze_sentiment))
        workflow.add_node("rag_analysis_node", self._timed_node("rag_analysis_node", self.nodes.perform_rag_analysis))
        workflow.add_node("react_reasoning", self._timed_node("react_reasoning", self.nodes.analyze_with_react))
        workflow.add_node("generate_report", self._timed_node("generate_report", self.nodes.generate_report))

        # Define the flow
        workflow.set_entry_point("collect_market_data")

        # Run analyses sequentially because nodes mutate and return the full state.
        workflow.add_edge("collect_market_data", "technical_analysis_node")
        workflow.add_edge("technical_analysis_node", "sentiment_analysis_node")
        workflow.add_edge("sentiment_analysis_node", "rag_analysis_node")
        workflow.add_edge("rag_analysis_node", "react_reasoning")

        # ReAct reasoning leads to report generation
        workflow.add_edge("react_reasoning", "generate_report")

        # End after report
        workflow.add_edge("generate_report", END)

        return workflow

    def analyze(self, ticker: str, query: Optional[str] = None) -> Dict[str, Any]:
        """Run the complete analysis workflow

        Args:
            ticker: Stock ticker symbol
            query: Optional specific query about the stock

        Returns:
            Complete analysis results
        """
        if not self.app:
            # Return a simple analysis if workflow compilation failed
            return {
                "error": "Workflow not initialized properly. Running simple analysis.",
                "ticker": ticker.upper(),
                "market_data": self._get_simple_market_data(ticker),
                "report": f"Simple analysis for {ticker}. Please ensure Ollama is running.",
                "recommendation": "UNABLE TO ANALYZE",
                "confidence_score": 0.0,
                "performance_metrics": {
                    "framework": "LangGraph",
                    "mode": "fast" if self.fast_mode else "full",
                    "total_duration_ms": 0,
                    "node_count": 0,
                    "successful_nodes": 0,
                    "failed_nodes": 0,
                    "nodes": [],
                }
            }

        # Initialize state
        initial_state: AgentState = {
            "ticker": ticker.upper(),
            "query": query or f"Analyze {ticker} stock for investment",
            "messages": [HumanMessage(content=query or f"Analyze {ticker}")],
            "market_data": {},
            "financial_metrics": {},
            "news_summary": "",
            "technical_analysis": {},
            "fundamental_analysis": {},
            "sentiment_analysis": {},
            "retrieved_documents": [],
            "rag_response": "",
            "reasoning_trace": [],
            "performance_metrics": {
                "framework": "LangGraph",
                "mode": "fast" if self.fast_mode else "full",
                "nodes": [],
                "total_duration_ms": 0,
                "node_count": 0,
                "successful_nodes": 0,
                "failed_nodes": 0,
            },
            "recommendation": "",
            "confidence_score": 0.0,
            "report": "",
            "next_step": "start",
            "error": None,
            "retry_count": 0
        }

        # Run workflow
        try:
            started = time.perf_counter()
            final_state = self.app.invoke(initial_state)
            duration_ms = (time.perf_counter() - started) * 1000

            metrics = final_state.setdefault("performance_metrics", {})
            nodes = metrics.get("nodes", [])
            metrics.update({
                "framework": "LangGraph",
                "mode": "fast" if self.fast_mode else "full",
                "total_duration_ms": round(duration_ms, 2),
                "node_count": len(nodes),
                "successful_nodes": sum(1 for node in nodes if node.get("status") == "success"),
                "failed_nodes": sum(1 for node in nodes if node.get("status") == "error"),
                "documents_retrieved": len(final_state.get("retrieved_documents", [])),
                "reasoning_steps": len(final_state.get("reasoning_trace", [])),
                "messages": len(final_state.get("messages", [])),
                "confidence_score": final_state.get("confidence_score", 0),
            })

            return final_state

        except Exception as e:
            return {
                "error": str(e),
                "ticker": ticker,
                "report": f"Analysis failed: {str(e)}",
                "recommendation": "ERROR",
                "confidence_score": 0.0,
                "performance_metrics": {
                    "framework": "LangGraph",
                    "mode": "fast" if self.fast_mode else "full",
                    "total_duration_ms": round((time.perf_counter() - started) * 1000, 2) if "started" in locals() else 0,
                    "node_count": len(initial_state.get("performance_metrics", {}).get("nodes", [])),
                    "successful_nodes": sum(
                        1 for node in initial_state.get("performance_metrics", {}).get("nodes", [])
                        if node.get("status") == "success"
                    ),
                    "failed_nodes": sum(
                        1 for node in initial_state.get("performance_metrics", {}).get("nodes", [])
                        if node.get("status") == "error"
                    ),
                    "nodes": initial_state.get("performance_metrics", {}).get("nodes", []),
                }
            }

    def _get_simple_market_data(self, ticker: str) -> Dict[str, Any]:
        """Get simple market data as fallback"""
        import yfinance as yf
        try:
            stock = yf.Ticker(ticker.upper())
            info = stock.info
            return {
                "price": info.get("currentPrice", 0),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0)
            }
        except:
            return {}

# Demonstration function
def demonstrate_workflow():
    """Demonstrate the complete workflow"""
    print("🎓 Financial Agent Workflow Demonstration")
    print("=" * 60)

    # Create workflow
    workflow = FinancialAgentWorkflow()

    # Analyze a stock
    ticker = "AAPL"
    print(f"\n🔍 Analyzing {ticker}...")

    result = workflow.analyze(ticker, "Should I invest in Apple stock?")

    # Display results
    print("\n📊 Analysis Results:")
    print("-" * 60)

    print(f"\n💡 Recommendation: {result.get('recommendation', 'N/A')}")
    print(f"📈 Confidence: {result.get('confidence_score', 0):.1%}")

    print(f"\n📝 Report Preview:")
    print(result.get('report', 'No report')[:500] + "...")

    print(f"\n🤔 ReAct Reasoning Steps: {len(result.get('reasoning_trace', []))}")

    print(f"\n📚 RAG Documents Retrieved: {len(result.get('retrieved_documents', []))}")

    if result.get('error'):
        print(f"\n❌ Error: {result['error']}")

if __name__ == "__main__":
    demonstrate_workflow()
