# going to create agent to understand the image.

from phi.agent import Agent

# from phi.model.groq import Groq
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from dotenv import load_dotenv
import os

load_dotenv()

# groq is not working with image, cause we cannot pass both prompt and image at the same time.
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

img_agent = Agent(
    name="Image Describe",
    model=Gemini(id="gemini-1.5-flash", api_key=GEMINI_API_KEY),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
)

img_agent.print_response(
    "Tell me about this image of john fruciante and give me the latest news about it.",
    images=[
        "https://upload.wikimedia.org/wikipedia/commons/6/67/John_Frusciante_%2852277957957%29_cropped.jpg"
    ],
    stream=True,
)
