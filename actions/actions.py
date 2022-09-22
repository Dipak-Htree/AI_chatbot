from cgitb import text
from typing import Dict, Text, Any, List
import csv
from rasa_sdk import Tracker, Action
from rasa_sdk.executor import CollectingDispatcher
import pandas as pd

class GetAnswer(Action):
    def name(self) -> Text:
        return "action_find_properties_for_agencies"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = pd.read_excel('/home/fx/Dev/AI_chatbot/Excel_File/Agent_Information.ods', engine='odf')
        # df = data[data["Agent"] == {'agency_list'}]



        agency = tracker.get_slot("agency_list")
        # dispatcher.utter_message(text = f"hgh {Tracker.get_latest_entity_values('name')}")
        dispatcher.utter_message(text=f"GetAnswer class is working properly {agency}")
        return []


# class ActionService(Action):
#     def name(self) -> Text:
#         return "action_service"
    
#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

#         data=[{"title":"A] 20-40 Lakhs","payload":"/informbudget{'budget':'AA'}"},
#             {"title":"B] 40-60 Lakhs","payload":"/informbudget{'budget':'BB'}"},
#             {"title":"C] 60-80 Lakhs","payload":"/informbudget{'budget':'CC'}"},
#             # {"title":"D] More than 80 Lakhs","payload":"/informbudget{'budget':'DD'}"},
#             ]

#         message={"payload":"dropDown","data":data}
        
#         dispatcher.utter_message(text="Please select a option",json_message=message)

 
