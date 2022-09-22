from argparse import Action
from typing import Dict, Text, Any, List
import csv
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher


def name(self) -> Text:

    return "action_findengr"

def run(self,

        dispatcher: CollectingDispatcher,

        tracker: Tracker,

        domain: Dict[Text, Any]

    ) -> List[Dict[Text, Any]]:

    # get the location slot

    location = tracker.get_slot('location')

    # read the CSV file

    with open('C:\\Users\\om sai infotech\\OneDrive\\Desktop\\bot\\actions\\financial.csv','r',encoding = "utf-8") as file:

        reader = csv.DictReader(file)

        # get a list of universities in the desired location

        output = [row for row in reader if row['Location'] == location]

    if output:

        reply  = f"Output: {location}:"

        reply += "\n- " + "\n- ".join([item['S.No.','date','firm','Ticker','Research Development','Income Before Tax','Net Income','Selling General Administrative','Gross Profit','Ebit','Operating Income','Interest Expense','Income Tax Expense','Total Revenue','Total Operating Expenses','Cost Of Revenue','Total Other Income Expense Net','Net Income From Continuing Ops','Net Income Applicable To Common Shares'] for item in output])

        # utter the message

        dispatcher.utter_message(reply)

    else: # the list is empty

        dispatcher.utter_message(f"I could not this in {location}")



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

 
