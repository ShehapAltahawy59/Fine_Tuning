import logging
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import os
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

gemini_api_key = os.environ["GEMINI_API_KEY"]


class Qustion(BaseModel):
    question: str
    answer : str

class output(BaseModel):
    output : List[Qustion]


def run():
    gemini_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash",
            api_key=gemini_api_key,
            model_info=ModelInfo(vision=True, function_calling=True, json_output=True, family="unknown", structured_output=True)
            )

    history_Specialist = AssistantAgent(
        name="history_Speciallist_agent",
        model_client=gemini_client,
        system_message="""You are a history expert assistant. From the paragraph , generate 2–3 factual question–answer pairs.
        -only make qustions relate to history.
        """
        ,
        output_content_type=output
        )
    
    return history_Specialist
