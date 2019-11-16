# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 02:34:18 2019

@author: melania
"""

from google.cloud import vision
from translate import Translator
import pandas as pd
import numpy as np
import io
import os
from fuzzywuzzy import fuzz
import json
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="kaggle-57628-e5244a8ec4e6.json"

def get_annotations(client, picture_path):
    """Get labels annotated by Google vision. Choose only the best labels. """
    treshold = 0.95
    with io.open(picture_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    response = client.label_detection(image=image)
    labels = response.label_annotations
    descr = []
    for label in labels:
        if label.score > treshold:
            descr.append(label.description)
        
    return descr

def translate_to_fi(annotations): 
    translated = [] 
    translator= Translator(to_lang="fi")
    for label in annotations: 
        translation = translator.translate(label)
        translated.append(translation)

    return translated 

def get_ingredient_name_id(labels, ingredients):
    ###ingredient: {"id" : "name"}
    ### return with higher common tokens  
    ###add code here
    top_score = 0 
    top_choice = ""
    top_id = 0 
    for ing in ingredients.items():
        name = ing[1]
        ing_id = ing[0]
        for label in labels:
            ratio = fuzz.partial_ratio(name, label)
            #other
            if ratio > top_score: 
                top_score = ratio
                top_choice = name
                top_id = ing_id                
    
    return top_choice, top_id, top_score

def localize_objects(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    returns: dictionary with products and its amount. 
    """
    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)

    objects = client.object_localization(
        image=image).localized_object_annotations

    print('Number of objects found: {}'.format(len(objects)))
    products = dict()
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        if object_.name not in products.keys(): 
            products[object_.name] = 1 
        else: 
            products[object_.name] += 1 
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))
            
    return products

def translate_to_fi_dict(products): 
    products_fi = {}
    translator= Translator(to_lang="fi")
    for key, value in products.items(): 
        products_fi[translator.translate(key)] = value
        
    return products_fi

def get_matched_ingredients(products, ingredients):
    ###ingredient: {"id" : "name"}
    ### return with higher common tokens  
    ###add code here
    matched = []
    for product, amount in products.items():   
        top_score = 70 
        top_choice = ""
        top_id = 0 
        for ing in ingredients:
            name = ing['name']
            ing_id = ing['id']
            ratio = fuzz.ratio(name, product)
            if ratio > top_score: 
                top_score = ratio
                top_choice = name
                top_id = ing_id  
                
        matched.append((top_choice, top_id, amount))
        
    return matched 


if __name__ == "__main__": 
    client = vision.ImageAnnotatorClient()

    translator= Translator(to_lang="fi")
    picture_path = "egg_milk.jpg"
    json_ingredients = "ingredients.json"
    with open('ingredients.json', encoding = 'utf-8') as f:
        ingredients = json.load(f)
    products = localize_objects(picture_path)
    print(products)
    products_fi = translate_to_fi_dict(products)
    print(products_fi)
    
    matched = get_matched_ingredients(products_fi, ingredients)
    print(matched)
        
    
    

