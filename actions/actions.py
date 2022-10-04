from cmath import cos, nan
from email import contentmanager
from lib2to3.pgen2 import driver
from typing import Dict, Text, Any, List, Optional
import csv
from rasa_sdk import Tracker, Action
from rasa_sdk.executor import CollectingDispatcher
from rasa.core.nlg import NaturalLanguageGenerator as nlg
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk import tokenize
from operator import itemgetter
import math
from bs4 import BeautifulSoup
import requests
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
data = pd.read_csv('Excel_File/mainData.csv')

class Information(Action):
    def name(self) -> Text:
        return "action_information"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_information')
        chosen_property = tracker.get_slot('chosen_property')
        self.property_detail = data.iloc[int(chosen_property)]
        return self.property_detail

    def scrapeData(self):
        link = self.property_detail['websiteLink']
        page = requests.get(link)
        soup = BeautifulSoup(page.content, "html.parser")
        for iter in soup(['style', 'script']):
            iter.decompose()
        self.scrape_data = ' '.join(soup.stripped_strings)
        return self.scrape_data

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
    
       


class CheckVillaType(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_house_type"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_house_type')
        super().run(dispatcher, tracker, domain)
        house_type = self.property_detail['houseType'].lower()
        q_house_type = tracker.get_slot('house_type')
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings1 = model.encode(house_type, convert_to_tensor=True)
        embeddings2 = model.encode(q_house_type, convert_to_tensor=True)
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
        print('action_check_property_floorplan')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['floorplanLayout'] == 'yes': 
            dispatcher.utter_message(text=f"{self.property_detail['floorplanLayout']}")
        elif self.property_detail['floorplanLayout'] == 'no':
            dispatcher.utter_message(text=f"No, ")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckBedroomNumber(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_bedroom"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_bedroom')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['bedroom'] is not None:
            dispatcher.utter_message(text=f"There are total of {self.property_detail['bedroom']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckLocation(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_show_property_address"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_show_property_address')
        super().run(dispatcher, tracker, domain)
        print(self.property_detail['propertyAddress'])
        if self.property_detail['propertyAddress'] is not None: 
            dispatcher.utter_message(text=f"Here is the address for this property.\n{self.property_detail['propertyAddress']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckParking(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_show_parking"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_show_parking')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['parkingSpace'] == 'yes': 
            dispatcher.utter_message(text=f"Yes, it includes parking.")
        elif self.property_detail['parkingSpace'] == 'no':
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
        print('action_show_garden')
        super().run(dispatcher, tracker, domain)

        if ("garden" in self.property_detail['keyFeatures']) or ("garden" in self.property_detail['propertyDescriptions']) : 
            dispatcher.utter_message(text=f"Yes, it includes garden.")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckWebsiteLink(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_show_website_link"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_show_website_link')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['websiteLink']: 
            dispatcher.utter_message(text=f"Yes, Here is the link for property, \n{self.property_detail['websiteLink']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckPetInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_show_pet_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_show_pet_info')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['petsAllowed'] == 'yes': 
            dispatcher.utter_message(text=f"Yes, it includes pets.")
        elif self.property_detail['petsAllowed'] == 'no':
            dispatcher.utter_message(text=f"No, it does not includes pets.")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckReceptionInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_reception_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_reception_info')
        super().run(dispatcher, tracker, domain)
        if ("reception" in self.property_detail['keyFeatures']) or ("reception" in self.property_detail['propertyDescriptions']) : 
            dispatcher.utter_message(text=f"Yes, it includes reception.")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckBathroomInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_bathroom_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_bathroom_info')
        super().run(dispatcher, tracker, domain)
        if "bathroom" in self.property_detail['bathroom']: 
            dispatcher.utter_message(text=f"Yes, This property includes bathroom \n{self.property_detail['bathroom']}")
        elif "no" in self.property_detail['bathroom']:
            dispatcher.utter_message(text=f"no, this property does not includes bathroom.")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []



class CheckFeaturesInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_features"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_features')
        return []

class CheckRentOrSaleInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_property_rent_or_sale"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action check property rent or sale')
        return []

class ChecksmokingInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_smoking"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_smoking')
        super().run(dispatcher, tracker, domain)
        if ("smoking" in self.property_detail['keyFeatures'].lower()) or ("smoking" in self.property_detail['propertyDescriptions'].lower()) : 
            dispatcher.utter_message(text=f"Yes, smoking is allowed")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")

class CheckLiftInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_lift"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_lift')
        super().run(dispatcher, tracker, domain)
        if ("lift" in self.property_detail['keyFeatures'].lower()) or ("lift" in self.property_detail['propertyDescriptions'].lower()) : 
            dispatcher.utter_message(text=f"Yes, it includes lift.")
        else:
            super().scrapeData()
            if ("lift" in self.scrape_data.lower()):
                dispatcher.utter_message(text=f"Yes, it includes lift.")
            else:
                dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckMapLinkInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_map_link_for_property"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_map_link_for_property')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['mapLink'] is not None: 
            dispatcher.utter_message(text=f"Here is map link for this property.\n{self.property_detail['mapLink']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckTenureInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_property_tenure"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_property_tenure')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['tenure'] is not None: 
            dispatcher.utter_message(text=f"Here is the property tenure.\n{self.property_detail['tenure']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckLettingDetailInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_letting_details"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_letting_details')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['lettingDetails'] is not None: 
            dispatcher.utter_message(text=f"Here is the letting details for this property.\n{self.property_detail['lettingDetails']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckKeyFeatureInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_key_features"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_key_features')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['keyFeatures'] is not None: 
            dispatcher.utter_message(text=f"Here is key features for this property.\n{self.property_detail['keyFeatures']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckPropertyDescriptionInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_property_descriptions"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_property_descriptions')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['propertyDescriptions'] is not None: 
            dispatcher.utter_message(text=f"Here is the property descriptions.\n{self.property_detail['propertyDescriptions']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckCouncilTaxInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_council_tax"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_council_tax')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['councilTax'] is not None: 
            dispatcher.utter_message(text=f"Here is the council tax info for this property.\n{self.property_detail['councilTax']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []


class CheckAgentNameInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_agent_name"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_agent_name')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['agencyName'] is not None: 
            dispatcher.utter_message(text=f"Here is the agency name for this property.\n{self.property_detail['agencyName']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []



class CheckAgentAddressInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_agent_address"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_agent_address')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['agencyAddress'] is not None: 
            dispatcher.utter_message(text=f"Here is the agency address for this property.\n{self.property_detail['agencyAddress']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckAgentDescriptionInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_agent_decriptions"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_agent_decriptions')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['agencyDescription'] is not None: 
            dispatcher.utter_message(text=f"Here is the agency description for this property.\n{self.property_detail['agencyDescription']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []



class CheckPropertyAddedInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "check_about_how_long_property_been_on_market"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('check_about_how_long_property_been_on_market')
        super().run(dispatcher, tracker, domain)
        super().scrapeData()
        x = re.search("(added on \d{2}(\/|-)\d{2}(\/|-)\d{2,4})", self.scrape_data.lower())
        if x is not None:
            dispatcher.utter_message(text=f"The property is listed on the market for .\n{x.group(0)}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []




class CheckPropertyCost(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_property_cost"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_property_cost')
        super().run(dispatcher, tracker, domain)
        if self.property_detail['propertyPrice'] is not None: 
            dispatcher.utter_message(text=f"Here is the property cost.\n{self.property_detail['propertyPrice']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

 
# class FallBack(Information):
#     def __init__(self):
#         super().__init__()

#     def name(self) -> Text:
#         return "my_fallback_action"

#     def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#         print('my_fallback_action')
#         return []

# class ActionDeafultFallback(Action):

#     def name(self) -> Text:
#         return "Yohohohooho"

#     def run(self, dispatcher, tracker, domain):
#         message = "This is the end!!!!!Yohohoho"
#         dispatcher.utter_message(text=message)

#         return []
