from dataclasses import dataclass, field
from lib2to3.pgen2.tokenize import tokenize
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from transformers.pipelines.conversational import Conversation
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Body(BaseModel):
    response: str
    arguments: Optional[Dict[str, Any]]
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


def build_conversation(tracker: Dict[str, Any]) -> Conversation:
    events = tracker["events"]
    user_msgs = [e["text"] for e in events if e["event"] == "user"]
    bot_msgs = [e["text"] for e in events if e["event"] == "bot"]

    latest_user_msg = user_msgs[-1]
    user_hist = user_msgs[:-1]

    return Conversation(
        text=latest_user_msg, past_user_inputs=user_hist, generated_responses=bot_msgs
    )


app = FastAPI()
# model = AutoModelForSeq2SeqLM.from_pretrained("blenderbot-3B")
# tokenizer = AutoTokenizer.from_pretrained("blenderbot-3B")
# model_pipeline = pipeline(
#     task="conversational", framework="pt", model=model
# )

nlp = pipeline(task="question-answering",model = "deepset/roberta-base-squad2", tokenizer="deepset/roberta-base-squad2")
params = {"do_sample":True, "num_beams":4, "no_repeat_ngram_size":3, "early_stopping":True,}  

@app.post("/nlg")
def generate_response(body: Body = None):
#     QA_input = {
#     'question': 'Why is model conversion important?',
#     'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
# }
#     res = nlp(QA_input)
    print(body.tracker)
    
    # conv = build_conversation(body.tracker)
    # bot_response_text = model_pipeline(conv).generated_responses[-1]
    # return Response(text=bot_response_text)