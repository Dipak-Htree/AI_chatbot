from cmath import cos, nan
from email import contentmanager
from lib2to3.pgen2 import driver
from multiprocessing import context
from typing import Dict, Text, Any, List, Optional, final
import csv
from xml.sax.handler import feature_external_ges
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
import spacy
import math
from bs4 import BeautifulSoup
import requests
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

#Model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')


data = pd.read_csv('Excel_File/mainData.csv')

property_detail = ''
scrape_data = None




class Information(Action):
    
    def __init__(self, property_detail = '') -> None:
        property_detail = property_detail
        
    def name(self) -> Text:
        return "action_information"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        global property_detail
        global scrape_data
        print('action_information')
        chosen_property = tracker.get_slot('chosen_property')
        property_detail = data.iloc[int(chosen_property)]
        


        if scrape_data is None:
            
            print("scraping ...")
            link = property_detail['websiteLink']
            page = requests.get(link)
            soup = BeautifulSoup(page.content, "html.parser")
            for iter in soup(['style', 'script']):
                iter.decompose()
            scrape_data = ' '.join(soup.stripped_strings)
            
        
        return property_detail




    def search_query(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any], q, context) -> List[Dict[Text, Any]]:
        # model_name = "deepset/roberta-base-squad2"
        model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        # model_name = 'deepset/roberta-large-squad2'
        # model_name = 'bert-large-cased-whole-word-masking-finetuned-squad'
        # model_name = 'm3hrdadfi/gpt2-QA'
        # model_name = 'csarron/bert-base-uncased-squad-v1'
        nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
        QA_input = {
            'question': f'{q}',
            'context': f"""{context}"""
        }
        res = nlp(QA_input)
        return [res]

    def extract_keywords(self, doc):
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
        
        house_type = property_detail['houseType'].lower()
        q_house_type = tracker.get_slot('house_type').lower()
        try:
            if math.isnan(house_type) :
                dispatcher.utter_message(text=f"Sorry, we don't have this information.")
            
        except:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embeddings1 = model.encode(house_type, convert_to_tensor=True)
            embeddings2 = model.encode(q_house_type, convert_to_tensor=True)
            cosine_scores = util.cos_sim(embeddings1, embeddings2)
            print(cosine_scores)
            if cosine_scores[0][0].item() >= 0.40:
                dispatcher.utter_message(text=f"This property is {house_type}")
            elif cosine_scores[0][0].item() < 0.40:
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
        if property_detail['floorplanLayout'] == 'yes': 
            dispatcher.utter_message(text=f"{property_detail['floorplanLayout']}")
        elif property_detail['floorplanLayout'] == 'no':
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
        if property_detail['bedroom'] is not None:
            dispatcher.utter_message(text=f"There are total of {property_detail['bedroom']}")
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
        print(property_detail['propertyAddress'])
        if property_detail['propertyAddress'] is not None: 
            dispatcher.utter_message(text=f"Here is the address for this property.\n{property_detail['propertyAddress']}")
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
        if property_detail['parkingSpace'] == 'yes': 
            dispatcher.utter_message(text=f"Yes, it includes parking.")
        elif property_detail['parkingSpace'] == 'no':
            dispatcher.utter_message(text=f"No, it does not includes parking.")
        else:
            ()
            q = "Does this property has parking space?"
            res= super().search_query(dispatcher, tracker, domain, q, scrape_data)
            q = tracker.latest_message["text"]
            # final_res = super().search_query(dispatcher, tracker, domain, q, res[0]['answer'])
            # print(res[0], "\n", final_res[0])
            # if "parking" in final_res[0]['answer'].lower():
            #     keywords = super().extract_keywords(final_res[0]['answer'])
            #     print("Keywords", keywords)
            #     dispatcher.utter_message(text=f"{final_res[0]['answer']}")
            # else:
            #     dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
            print(res[0])
            encoding = tokenizer.encode_plus(text=q,text_pair=res[0]['answer'], padding=True, truncation=True, add_special_tokens=True)
            inputs = encoding['input_ids']  #Token embeddings
            sentence_embedding = encoding['token_type_ids']  #Segment embeddings
            tokens = tokenizer.convert_ids_to_tokens(inputs) 
            output = model(torch.tensor([inputs]),  token_type_ids=torch.tensor([sentence_embedding]))
            answer_start = torch.argmax(output.start_logits)
            answer_end = torch.argmax(output.end_logits)
            if answer_end >= answer_start:
                answer = " ".join(tokens[answer_start:answer_end+1])
                dispatcher.utter_message(text=f"{answer}")
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

        if ("garden" in property_detail['keyFeatures']) or ("garden" in property_detail['propertyDescriptions']) : 
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
        if property_detail['websiteLink']: 
            dispatcher.utter_message(text=f"Yes, Here is the link for property, \n{property_detail['websiteLink']}")
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
        if property_detail['petsAllowed'] == 'yes': 
            dispatcher.utter_message(text=f"Yes, it includes pets.")
        elif property_detail['petsAllowed'] == 'no':
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
        if ("reception" in property_detail['keyFeatures']) or ("reception" in property_detail['propertyDescriptions']) : 
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
        bathroom = property_detail['bathroom']
        try:
            if math.isnan(bathroom) :
                dispatcher.utter_message(text=f"Sorry, we don't have this information.")
        except:
            dispatcher.utter_message(text=f"Yes, This property includes {bathroom} bathroom")
        return []



class CheckFeaturesInfo(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_features"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        super().run(dispatcher, tracker, domain)
        print('action_check_features')
        feature = tracker.latest_message['entities'][0]['value']
        if feature == 'facilities':
            dispatcher.utter_message(text=f"{property_detail['keyFeatures']}")
        elif (feature in property_detail['keyFeatures']) or (feature in property_detail['propertyDescriptions']):
            dispatcher.utter_message(text=f"Yes, This property includes {feature}")
        else:
            dispatcher.utter_message(text=f"No, This property does not includes {feature}")
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
        if ("smoking" in property_detail['keyFeatures'].lower()) or ("smoking" in property_detail['propertyDescriptions'].lower()) : 
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
        if ("lift" in property_detail['keyFeatures'].lower()) or ("lift" in property_detail['propertyDescriptions'].lower()) : 
            dispatcher.utter_message(text=f"Yes, it includes lift.")
        else:
            ()
            if ("lift" in scrape_data.lower()):
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
        if property_detail['mapLink'] is not None: 
            dispatcher.utter_message(text=f"Here is map link for this property.\n{property_detail['mapLink']}")
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
        if property_detail['tenure'] is not None: 
            if 'ask agent' not in property_detail['tenure']:
                dispatcher.utter_message(text=f"Here is the property tenure.\n{property_detail['tenure']}")
            else:
                dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
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
        if property_detail['lettingDetails'] is not None: 
            dispatcher.utter_message(text=f"Here is the letting details for this property.\n{property_detail['lettingDetails']}")
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
        if property_detail['keyFeatures'] is not None: 
            dispatcher.utter_message(text=f"Here is key features for this property.\n{property_detail['keyFeatures']}")
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
        if property_detail['propertyDescriptions'] is not None: 
            dispatcher.utter_message(text=f"Here is the property descriptions.\n{property_detail['propertyDescriptions']}")
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
        if property_detail['councilTax'] is not None: 
            dispatcher.utter_message(text=f"Here is the council tax info for this property.\n{property_detail['councilTax']}")
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
        if property_detail['agencyName'] is not None: 
            dispatcher.utter_message(text=f"Here is the agency name for this property.\n{property_detail['agencyName']}")
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
        if property_detail['agencyAddress'] is not None: 
            dispatcher.utter_message(text=f"Here is the agency address for this property.\n{property_detail['agencyAddress']}")
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
        if property_detail['agencyDescription'] is not None: 
            dispatcher.utter_message(text=f"Here is the agency description for this property.\n{property_detail['agencyDescription']}")
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
        ()
        x = re.search("(added on \d{2}(\/|-)\d{2}(\/|-)\d{2,4})", scrape_data.lower())
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
        if property_detail['propertyPrice'] is not None: 
            dispatcher.utter_message(text=f"Here is the property cost.\n{property_detail['propertyPrice']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []

class CheckBedroomDescription(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_bedroom_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_bedroom_info')
        super().run(dispatcher, tracker, domain)
        ()
        
        



        q = "show me bedroom information?"
        res= super().search_query(dispatcher, tracker, domain, q, scrape_data)
        q = tracker.latest_message["text"]
        final_res = super().search_query(dispatcher, tracker, domain, q, scrape_data)
        print(res[0], "\n", final_res[0])
        if final_res[0]['score'] >= 0.01:
            dispatcher.utter_message(text=f"{final_res[0]['answer']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have that info.")
        return []


class BathroomDescription(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_bathroom"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_bathroom')
        super().run(dispatcher, tracker, domain)
        ()
        
        q = "show me bathroom information?"
        res= super().search_query(dispatcher, tracker, domain, q, scrape_data)
        q = tracker.latest_message["text"]
        final_res = super().search_query(dispatcher, tracker, domain, q, res[0]['answer'])
        print(res[0], "\n", final_res[0])


        lines = []
        nlp = spacy.load("en_core_web_md")
        doc = nlp(scrape_data)
        for sent in doc.sents:
            lines.append(sent.text)
        final_data = []
        for line in lines:
            if 'bedroom' in line.lower():
                final_data.append(line)
        final_data = " ".join(final_data)

        encoding = tokenizer.encode_plus(text=q,text_pair=final_data, padding=True, truncation=True, add_special_tokens=True)
        inputs = encoding['input_ids']  #Token embeddings
        sentence_embedding = encoding['token_type_ids']  #Segment embeddings
        tokens = tokenizer.convert_ids_to_tokens(inputs) 

        output = model(torch.tensor([inputs]),  token_type_ids=torch.tensor([sentence_embedding]))
        answer_start = torch.argmax(output.start_logits)
        answer_end = torch.argmax(output.end_logits)
        if answer_end >= answer_start:
            answer = " ".join(tokens[answer_start:answer_end+1])
            print("Answer: ", answer)
        else:
            print("Answer: I am unable to find the answer to this question. Can you please ask another question?")


        if final_res[0]['score'] >= 0.01:
            dispatcher.utter_message(text=f"{final_res[0]['answer']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have that info.")
        return []



class CheckKitchen(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_kitchen"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_kitchen')
        super().run(dispatcher, tracker, domain)
        ()
        q = "show me kitchen information?"
        res= super().search_query(dispatcher, tracker, domain, q, scrape_data)
        q = tracker.latest_message["text"]
        # print(res[0])
        final_res = super().search_query(dispatcher, tracker, domain, q, res[0]['answer'])
        print(res[0], "\n", final_res[0])
        if final_res[0]['score'] >= 0.05:
            # keywords = super().extract_keywords(final_res[0]['answer'])
            # print("Keywords", keywords)
            dispatcher.utter_message(text=f"{final_res[0]['answer']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        
        # encoding = tokenizer.encode_plus(text=q,text_pair=res[0]['answer'], padding=True, truncation=True, add_special_tokens=True)
        # inputs = encoding['input_ids']  #Token embeddings
        # sentence_embedding = encoding['token_type_ids']  #Segment embeddings
        # tokens = tokenizer.convert_ids_to_tokens(inputs) 
        # output = model(torch.tensor([inputs]),  token_type_ids=torch.tensor([sentence_embedding]))
        # print(output)
        # answer_start = torch.argmax(output.start_logits)
        # answer_end = torch.argmax(output.end_logits)
        # if answer_end >= answer_start:
        #     answer = " ".join(tokens[answer_start:answer_end+1])
        #     dispatcher.utter_message(text=f"{answer}")
        # else:
        #     dispatcher.utter_message(text=f"Sorry, we don't have this information, as soon as we get this information, we'll inform you.")
        return []



class CheckLivingRoomDescription(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_living_room"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_living_room')
        super().run(dispatcher, tracker, domain)
        ()
        
        q = "show me living room information?"
        res= super().search_query(dispatcher, tracker, domain, q, scrape_data)
        q = tracker.latest_message["text"]
        final_res = super().search_query(dispatcher, tracker, domain, q, res[0]['answer'])
        print(res[0], "\n", final_res[0])
        if final_res[0]['score'] >= 0.01:
            dispatcher.utter_message(text=f"{final_res[0]['answer']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have that info.")
        return []



class CheckEntranceHall(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_entrance_hall_info"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_entrance_hall_info')
        super().run(dispatcher, tracker, domain)
        ()
        
        q = "show me entrance hall information?"
        res= super().search_query(dispatcher, tracker, domain, q, scrape_data)
        q = tracker.latest_message["text"]
        final_res = super().search_query(dispatcher, tracker, domain, q, res[0]['answer'])
        print(res[0], "\n", final_res[0])
        if final_res[0]['score'] >= 0.01:
            dispatcher.utter_message(text=f"{final_res[0]['answer']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have that info.")
        return []




class CheckDiningRoom(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "action_check_about_dining_room"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('action_check_about_dining_room')
        super().run(dispatcher, tracker, domain)
        ()
        
        q = "Show me dining room information?"
        res= super().search_query(dispatcher, tracker, domain, q, scrape_data)
        q = tracker.latest_message["text"]
        final_res = super().search_query(dispatcher, tracker, domain, q, res[0]['answer'])
        print(res[0], "\n", final_res[0])
        if final_res[0]['score'] >= 0.01:
            dispatcher.utter_message(text=f"{final_res[0]['answer']}")
        else:
            dispatcher.utter_message(text=f"Sorry, we don't have that info.")
        return []










 
class FallBack(Information):
    def __init__(self):
        super().__init__()

    def name(self) -> Text:
        return "my_fallback_action"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        print('my_fallback_action')
        super().run(dispatcher, tracker, domain)
        ()

        q = tracker.latest_message["text"]
        res = super().search_query(dispatcher, tracker, domain,q, scrape_data)
        
        print(res[0])
        if res[0]['score'] > 0.10:
            dispatcher.utter_message(text=res[0]['answer'])
        else:
            dispatcher.utter_message(text="Sorry, we don't have that information. we will inform about this to the agency, they will shortly contact you.")
        
        return []

