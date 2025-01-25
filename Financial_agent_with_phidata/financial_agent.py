# Building a financial agent/analyst : a chatbot
# Like if someone asks, "hey, can you summarise and give recommandation on the nvidia stock?"
# What this chatbot would do is:
#  1. go to the agents, there are multiple agent to do different tasks.
#  2. like, one ai agent get the details of the stock and other ai agent get the information regarding the stock from web and news.
#  3. Then, we feed this data to a llm model.
#  4. lastly, produce result ( recommandation ) for the user.

from phi.agent import Agent
from phi.model.groq import Groq
from dotenv import load_dotenv
import os
from phi.tools.yfinance import YFinanceTools

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

"""
 
"""

financial_agent = Agent(
    name="Financial Analyst",
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    tools=[
        YFinanceTools(
            stock_price=True,
            company_info=True,
            stock_fundamentals=True,
            analyst_recommendations=True,
            key_financial_ratios=True,
        )
    ],
    instructions=["Use tables to display data."],
    show_tool_calls=True,
    markdown=True,
)

financial_agent.print_response(
    "comprehensive analyst recommendations for NVDA", stream=True
)
