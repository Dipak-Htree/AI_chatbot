from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from rasa_sdk import Tracker, Action
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from transformers.pipelines.conversational import Conversation
from rasa.core.nlg.generator import NaturalLanguageGenerator

nlg = NaturalLanguageGenerator()

class Body(BaseModel):
    response: str
    arguments: Optional[Dict[str, Any]]
    # tracker: Dict[str, Any]
    tracker: Dict[str, Any]
    channel: Dict[str, str] = None


@dataclass
class Response:
    text: str
    buttons: List[Any] = field(default_factory=lambda: [])
    image: str = None
    elements: List[Any] = field(default_factory=lambda: [])
    attachments: List[Any] = field(default_factory=lambda: [])
    custom: Dict = field(default_factory=lambda: {})


async def build_conversation(tracker: Dict[str, Any]) -> Conversation:
    print("____________________________________")
    # print(tracker)
    print("____________________________________")
    events = tracker["events"]
    print(tracker['latest_action'], "\n\n\n\n")
    user_msgs = [e["text"] for e in events if e["event"] == "user"]
    bot_msgs = [e["text"] for e in events if e["event"] == "bot"]
    

    latest_user_msg = user_msgs[-1]
    user_hist = user_msgs[:-1]

    print("latest user msg:", latest_user_msg)
    print("user hist", user_hist)
    print("bot msgs", bot_msgs)
    print("-----------", await nlg.generate(utter_action="utter_ask_any_questions", tracker = tracker, output_channel = None) )    
    
    return Conversation(
        text=latest_user_msg,  generated_responses=bot_msgs
    )
    # return await nlg.generate(utter_action="utter_ask_any_questions", tracker = tracker, output_channel = None) 

app = FastAPI()
model_pipeline = pipeline(
    task="conversational",
    framework="pt",
    model = 'facebook/blenderbot_small-90M' 
)


@app.post("/nlg")
async def generate_response(body: Body = None):
    conv = await build_conversation(body.tracker)
    print("------------------------Conversation-----------------\n")
    bot_response_text = model_pipeline(conv).generated_responses
    print("------------------------Response-----------------\n", bot_response_text)
    return Response(text=bot_response_text[-1])

