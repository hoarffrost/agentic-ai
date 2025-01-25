# here, we are going to build team of agents.
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo

# refer to the phidata storage section to understand why storage for agent: https://docs.phidata.com/storage/introduction
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os

load_dotenv()

# get the api key to use llm
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

web_agent = Agent(
    name="Web Agent",
    role="a web crawler fetching the data we need of any topic. you are an expert at research.",
    model=Gemini(id="gemini-1.5-flash-8b", api_key=GEMINI_API_KEY),
    tools=[DuckDuckGo()],
    instructions=["Always include the sources"],
    storage=SqlAgentStorage(table_name="web_agent_session", db_file="tmp/web_data.db"),
    show_tool_calls=True,
    add_history_to_messages=True,
    markdown=True,
)

financial_agent = Agent(
    name="Financial Analyst",
    role="a seasoned financial analyst. give your recommandation and financial advise.",
    model=Gemini(id="gemini-1.5-flash-8b", api_key=GEMINI_API_KEY),
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
    show_tool_calls=True,
    add_history_to_messages=True,
    markdown=True,
)

agents_team = Agent(
    team=[web_agent, financial_agent],
    # this model needs to be added here for the team of agents to work.
    # but above info is not given in the example of phidata i am following.
    # openai is the default model in phidata agents.
    model=Gemini(id="gemini-1.5-flash-8b", api_key=GEMINI_API_KEY),
    instructions=["Always include sources", "Use tables to display data"],
    show_tool_calls=True,
    storage=SqlAgentStorage(
        table_name="agent_team_session", db_file="tmp/team_data.db"
    ),
    add_history_to_messages=True,
    markdown=True,
)


agents_team.print_response(
    "tell me about the movie the pianist and also all the finance related to this movie, like production and how much it earned.",
    stream=True,
)

# done, works.
