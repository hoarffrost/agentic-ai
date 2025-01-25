# refer: https://docs.phidata.com/agent-ui
# to understand how agent ui ( playground ) works provided by phidata.
# This will convert the agents we build into a chatbot for user to interact with.

from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.duckduckgo import DuckDuckGo
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os

load_dotenv()

# get the api key to use llm ( Gemini from google )
# gemini model is being able to streaming ( chat completion )
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# so using groq instead
# here you can do streaming with groq models: https://console.groq.com/docs/text-chat
Groq_API_KEY = os.getenv("GROQ_API_KEY")

web_agent = Agent(
    name="Web Agent",
    role="a web crawler fetching the data we need of any topic. you are an expert at research.",
    model=Groq(id="llama-3.3-70b-versatile", api_key=Groq_API_KEY, stream=True),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    storage=SqlAgentStorage(table_name="web_agent_session", db_file="tmp/web_data.db"),
    # if "add_history_to_messages=true" then adds the chat history to the messages sent to the Model (llm).
    add_history_to_messages=True,
    markdown=True,
)

financial_agent = Agent(
    name="Financial Analyst",
    role="a seasoned financial analyst. give your recommandation and financial advise.",
    model=Groq(id="llama-3.3-70b-versatile", api_key=Groq_API_KEY, stream=True),
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
    storage=SqlAgentStorage(
        table_name="financial_agent_session", db_file="tmp/finance_data.db"
    ),
    # tool calls doesnot make sense as we are not seeing the output of agent on cli, but on a chatbot way.
    # show_tool_calls=True,
    add_history_to_messages=True,
    markdown=True,
)

# building a playground for our agents
from phi.playground import Playground, serve_playground_app

"""
Got the playground working on the chrome, not on brave.
Read for more details: https://docs.phidata.com/faq/phi-auth
"""

app = Playground(agents=[financial_agent, web_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app(app="playground:app", host="127.0.0.1", reload=True)

# Yayy, got this app working.
