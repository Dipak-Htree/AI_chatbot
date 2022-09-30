from cmath import cos
from typing import Dict, Text, Any, List, Optional
import csv
from rasa_sdk import Tracker, Action
from rasa_sdk.executor import CollectingDispatcher
from rasa.core.nlg import NaturalLanguageGenerator as nlg
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import json
import nltk
from nltk import tokenize
from operator import itemgetter
import math
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

data=pd.read_json('/home/fx/Dev/AI_chatbot/Excel_File/rightmove.json').T

property_data = pd.read_excel('Excel_File/Properties_Information.ods', engine='odf')
agency_data =  pd.read_excel('Excel_File/Agent_Information.ods', engine='odf')

class Information(Action):
    def name(self) -> Text:
        return "action_information"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        chosen_property = tracker.get_slot('chosen_property')
        self.property_detail = data.iloc[int(chosen_property)]
        return self.property_detail

    def search_query(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any], q, context) -> List[Dict[Text, Any]]:
        model_name = "deepset/roberta-base-squad2"

        nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
        QA_input = {
            'question': f'{q}',
            'context': f"""{context}"""
        }
        res = nlp(QA_input)
        return [res]

    def extract_keywords(doc):
        def check_sent(word, sentences): 
            final = [all([w in x for w in word]) for x in sentences] 
            sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
            return int(len(sent_len))
        def get_top_n(dict_elem, n):
            result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
            return result
        stop_words = set(stopwords.words('english'))
        total_sentences = tokenize.sent_tokenize(doc)
        total_sent_len = len(total_sentences)
        total_words = doc.split()
        total_word_length = len(total_words)
        tf_score = {}
        for each_word in total_words:
            each_word = each_word.replace('.','')
            if each_word not in stop_words:
                if each_word in tf_score:
                    tf_score[each_word] += 1
                else:
                    tf_score[each_word] = 1

        # Dividing by total_word_length for each dictionary element
        tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
        idf_score = {}
        for each_word in total_words:
            each_word = each_word.replace('.','')
            if each_word not in stop_words:
                if each_word in idf_score:
                    idf_score[each_word] = check_sent(each_word, total_sentences)
                else:
                    idf_score[each_word] = 1

        # Performing a log and divide
        idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())
        tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}
        keywords_dict = get_top_n(tf_idf_score, 5)
        keywords = keywords_dict.keys()

        return list(keywords)



class GetAnswer(Action):
    def name(self) -> Text:
        return "action_find_properties_for_agencies"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        agency = tracker.get_slot("agency_list")
        try:
            df = agency_data[agency_data["Agent"] == f'{agency}']['Property Information']
            dispatcher.utter_message(text=f"You have following property for agency {agency}. \n{df.T}")
        except:
            dispatcher.utter_message(text=f"Sorry, there is no information for this agency.")
            df = None
        return []


class ShowProperty(Action):
    def __init__(self) -> None:
        pass 

    def name(self) -> Text:
        return "action_showing_property"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        chosen_property = next(tracker.get_latest_entity_values('chosen_property'), None)
        try:
            self.property_detail = property_data[property_data['Properties'] == f'{chosen_property}']
            dispatcher.utter_message(text=f"Here is the informatioon. \n{self.property_detail.T}.")
        except:
            dispatcher.utter_message(text=f"Sorry, there is no information for this property.")
        return []


class CheckVillaType(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_house_type"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        house_type = self.property_detail['Property']['PROPERTY TYPE'].lower()
        q_house_type = tracker.get_slot('house_type')
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings1 = model.encode(house_type, convert_to_tensor=True)
        embeddings2 = model.encode(q_house_type, convert_to_tensor=True)
        
        #Compute cosine-similarities
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        print(cosine_scores)
        if cosine_scores[0][0].item() > 0.50:
            dispatcher.utter_message(text=f"Yes, This property is {house_type}")
        elif cosine_scores[0][0].item() < 0.50:
            dispatcher.utter_message(text=f"No, This property is {house_type}")
            

        return []


class CheckFloorPlan(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_property_floorplan"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        print(self.property_detail['Floorplan'].iloc[0])
        if self.property_detail['Floorplan'].iloc[0]: 
            dispatcher.utter_message(text=f"Yes, this property has floorplan. Here is the link for it.\n {self.property_detail['Floorplan'].iloc[0]}")
        else:
            dispatcher.utter_message(text=f"No, this property has no floorplan.")
        return []


class CheckBedroomNumber(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_bedroom"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        if self.property_detail['Property'] is not None:
             
            dispatcher.utter_message(text=f"There are total of {self.property_detail['Property']['BEDROOMS']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckLocation(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_show_address"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        print(self.property_detail['Address'])
        if self.property_detail['Address'] is not None: 
            dispatcher.utter_message(text=f"Here is the address for this property.\n{self.property_detail['Address']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckParking(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_show_parking"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        print(self.property_detail['Parking'].iloc[0])
        if "yes" in self.property_detail['Parking'].iloc[0].lower(): 
            dispatcher.utter_message(text=f"Yes, it includes parking.")
        elif "no" in self.property_detail['Parking'].iloc[0].lower():
            dispatcher.utter_message(text=f"No, it does not includes parking.")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckGarden(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_show_garden"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        print(self.property_detail['Garden'].iloc[0])
        if "yes" in self.property_detail['Garden'].iloc[0].lower(): 
            dispatcher.utter_message(text=f"Yes, it includes garden.")
        elif "no" in self.property_detail['Garden'].iloc[0].lower():
            dispatcher.utter_message(text=f"No, it does not includes garden.")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckWebsiteLink(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_show_website_link"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        print(self.property_detail['Website Link'].iloc[0])
        if self.property_detail['Website Link'].iloc[0].lower(): 
            dispatcher.utter_message(text=f"Yes, Here is the link for property, \n{self.property_detail['Website Link'].iloc[0]}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckPetInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_show_pet_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        print(self.property_detail['Pets'].iloc[0])
        if "no" in self.property_detail['Pets'].iloc[0].lower(): 
            dispatcher.utter_message(text=f"No, this porperty is not pet friendly.")
        elif "yes" in self.property_detail['Pets'].iloc[0].lower():
            dispatcher.utter_message(text=f"yes, this property is pet friendly.")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckReceptionInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_reception_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        print(self.property_detail['Receptions'].iloc[0])
        if "no" in self.property_detail['Receptions'].iloc[0].lower(): 
            dispatcher.utter_message(text=f"No, this porperty includes reception.")
        elif "yes" in self.property_detail['Receptions'].iloc[0].lower():
            dispatcher.utter_message(text=f"yes, this property includes reception.")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckBathroomInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_bathroom_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        print(self.property_detail['Bathroom'].iloc[0])
        if "bathroom" in self.property_detail['Bathroom'].iloc[0].lower(): 
            dispatcher.utter_message(text=f"Yes, This property includes bathroom \n{self.property_detail['Bathroom'].iloc[0]}")
        elif "no" in self.property_detail['Bathroom'].iloc[0].lower():
            dispatcher.utter_message(text=f"no, this property does not includes bathroom.")
        return []



class CheckFeaturesInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_features"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return []

class CheckFeaturesInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_property_rent_or_sale"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        return []


class CheckLiftInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        
        return "action_check_lift"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        return []




# class CheckHouseInfo(Information):
#     def __init__(self):
#         super().__init__()

#     def name(self) -> Text:
#         return "action_check_house_type"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         super().run(dispatcher, tracker, domain)
#         print(self.property_detail['Type'].iloc[0])
#         if self.property_detail['Type'].iloc[0].lower(): 
#             dispatcher.utter_message(text=f"House type for this property is {self.property_detail['Type'].iloc[0]}")
#         else: 
#             dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
#         return []





 
