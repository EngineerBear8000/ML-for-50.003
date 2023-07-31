from PIL import Image, ImageOps
import numpy as np
from copy import deepcopy
from difflib import SequenceMatcher
from paddleocr import PaddleOCR
import json
ocr_model = PaddleOCR(lang='en',use_angle_cls=True,show_log=False)

def run_ocr(image,type_of_document='invoice'):
    #Transpose the image accordingly to EXIF Orientation tag
    image = ImageOps.exif_transpose(image)
    #Image conversion to be used for paddleOCR
    img = np.asarray(image)
    #Implementing OCR for the entire image
    raw = ocr_model.ocr(img, cls=True)
    raw = order_by_tbyx(raw)[0]
    
    #Start and end identifiers determine the start and end rows of a table
    if type_of_document == 'invoice':
        start_identifiers = ['Description','Particular','Item']
        end_identifiers = ['Total']
    elif type_of_document == 'statement':
        start_identifiers = ['Document','Particulars']
        end_identifiers = ['Total']
    
    #Find startpoint accordingly to identifier
    startpoint = query_word_single(raw,start_identifiers,instance_from_front=True)
    if startpoint:
        #Since the start identifier selected may not be the first of the row, determine the actual startpoint
        for i in raw[raw.index(startpoint)::-1]:
            startpoint_xmin = min([i[0] for i in startpoint[0]])
            i_xmax = max([j[0] for j in i[0]])
            if i_xmax>startpoint_xmin:
                break
            startpoint = i
    else:
        print("ERROR: No initialier word can be found.")
        return
    
    #Find endpoint accordingly to identifier
    endpoint = query_word_single(raw[raw.index(startpoint)+5::],end_identifiers,instance_from_front=True)
    if not endpoint:
        endpoint = raw[-1]
    
    #Obtain 1D table data without structure, organize it into a 2D array so that has structure
    table_data = raw[raw.index(startpoint):raw.index(endpoint):]
    org_table_data = organise_table(table_data)
    #Some data may be seperated into 2 rows but are supposed to be the same datapoint, find and merge them
    merged_table = table_merger(org_table_data)
    
    #Organise data to be formatted and sent to backend
    row_data = []
    for i in merged_table[1::]:
        temp = []
        for j in i:
            if j!='-':
                temp.append(j[1][0])
            else:
                temp.append(None)
        row_data.append(temp)
    header_data = []
    for i in merged_table[0]:
        if i!='-':
            header_data.append(i[1][0])
        else:
            header_data.append(None)
    extracted_data = {"header":header_data,"row_data":row_data}
    return json.dumps(formater(extracted_data))

def run_ocr_template(image,template_data_loc,template_size):
    #Get position of table
    table_box = (template_data_loc["table_data"][0][0],template_data_loc["table_data"][0][1],
             template_data_loc["table_data"][1][0],template_data_loc["table_data"][2][1])
    
    #Transpose the image accordingly to EXIF Orientation tag
    image = ImageOps.exif_transpose(image)
    #Image conversion to be used for paddleOCR
    img = np.asarray(image)
    #Implementing OCR for the entire image
    raw = ocr_model.ocr(img, cls=True)
    raw = order_by_tbyx(raw)[0]
    
    #Perform image resizing
    image = image.resize(template_size)
    
    #Extract data requested from template 
    img = np.asarray(image)
    data = ocr_model.ocr(img)[0]
    table_data,extracted_data = extract_requested_data(template_data_loc,data,table_box)
    
    #Extract Table Data
    table_output = organise_table(table_data)
    merged_table = table_merger(table_output)
    
    row_data = []
    for i in merged_table[1::]:
        temp = []
        for j in i:
            if j!='-':
                temp.append(j[1][0])
            else:
                temp.append(None)
        row_data.append(temp)
    header_data = []
    for i in merged_table[0]:
        if i!='-':
            header_data.append(i[1][0])
        else:
            header_data.append(None)
    extracted_data = {"header":header_data,"row_data":row_data}
    return json.dumps(formater(extracted_data))

#Sorting of data in a normal reading order 
def order_by_tbyx(ocr_info):
    output = sorted(ocr_info,key=lambda r:(r[0][1],r[0][0]))
    for i in range(len(output)-1):
        for j in range(i,0,-1):
            if abs(output[j+1][0][1]-output[j][0][1])<20 and (output[j+1][0][0]<output[j][0][0]):
                temp = deepcopy(output[j])
                output[j] = deepcopy(output[j+1])
                output[j+1] = deepcopy(temp)
            else:
                break
    return output

#Search for a word in the data that is same r very similar
def query_word_single(data,words,instance_from_front):
    data_copy = deepcopy(data)
    if not instance_from_front:
        data_copy = data_copy[::-1]
    for word in words:
        for i in data_copy:
            if word.casefold() in i[1][0].casefold() or SequenceMatcher(None, word.casefold(), i[1][0].casefold()).ratio()>0.7:
                return i

#Using the template provided, search and extract the data requested
def extract_requested_data(template_data,data,table_box):
    table_box_xmax = table_box[2]
    table_box_xmin = table_box[0]
    table_box_ymax = table_box[3]
    table_box_ymin = table_box[1]
    
    table_data = []
    output = {}

    for k,v in template_data.items():
        template_xmax = max([n[0] for n in v])
        template_xmin = min([n[0] for n in v])
        template_ymax = max([n[1] for n in v])
        template_ymin = min([n[1] for n in v])
            
        for i in data:
            data_xmax = max([j[0] for j in i[0]])
            data_xmin = min([j[0] for j in i[0]])
            data_ymax = max([j[1] for j in i[0]])
            data_ymin = min([j[1] for j in i[0]])
            
            if k=='table_data' and not (data_xmax>table_box_xmax and data_xmin>table_box_xmax) and not (data_xmax<table_box_xmin and data_xmin<table_box_xmin) and not (data_ymax>table_box_ymax and data_ymin>table_box_ymax) and not (data_ymax<table_box_ymin and data_ymin<table_box_ymin):
                table_data.append(i)
            elif not (data_xmax>template_xmax and data_xmin>template_xmax) and not (data_xmax<template_xmin and data_xmin<template_xmin) and not (data_ymax>template_ymax and data_ymin>template_ymax) and not (data_ymax<template_ymin and data_ymin<template_ymin):
                if k in output.keys():
                    output[k] = output[k] + " " + i[1][0]
                else:
                    output[k] = i[1][0]
    return table_data, output

def organise_table(table_data):
    data_copy = deepcopy(table_data)
    
    #Cut data into seperate piece whenever the data is on a new row
    first_xmin = min([i[0] for i in table_data[0][0]])
    org_data = []
    temp = []
    for i in table_data:
        i_xmax = max([j[0] for j in i[0]])
        i_xmin = min([j[0] for j in i[0]])
        if i_xmax<first_xmin:
            org_data.append(temp)
            temp = []
        first_xmin = i_xmin
        temp.append(i)
    org_data.append(temp)
    
    #Create an output array that can fit all the data inside including the empty cells
    max_rows = len(org_data)
    all_column_len = max([(n,len(i)) for n,i in enumerate(org_data)], key = lambda x: x[1])
    row_of_max_column = all_column_len[0]
    max_columns = all_column_len[1]
    output = [['-']*max_columns for _ in range(max_rows)]
    
    #Using the row with the highest number of entries and the first element of each row as reference x and y values,
    #we check every cell generated from these x and y values if a data point resides within.
    #If it is within, add to the output cell. If the output cell already contains another datapoint, compare the y values
    #to determine which should be on top, merge the 2 cells and replace the original cell
    for col,i in enumerate(org_data[row_of_max_column]):
        i_xmax = max([k[0] for k in i[0]])
        i_xmin = min([k[0] for k in i[0]])
        for row,j in enumerate([j[0] for j in org_data]):
            j_ymax = max([k[1] for k in j[0]])
            j_ymin = min([k[1] for k in j[0]])
                
            for k in data_copy:
                k_xmax = max([n[0] for n in k[0]])
                k_xmin = min([n[0] for n in k[0]])
                k_ymax = max([n[1] for n in k[0]])
                k_ymin = min([n[1] for n in k[0]])
                
                if not (k_xmax>i_xmax and k_xmin>i_xmax) and not (k_xmax<i_xmin and k_xmin<i_xmin) and not (k_ymax>j_ymax and k_ymin>j_ymax) and not (k_ymax<j_ymin and k_ymin<j_ymin):
                    if output[row][col] == '-':
                        output[row][col] = k
                    else:
                        ymax = max([n[1] for n in output[row][col][0]])
                        if ymax<k_ymax:
                            new_entry = merge_words(output[row][col],k)
                        else:
                            new_entry = merge_words(k,output[row][col])
                        output[row][col] = new_entry
                    data_copy.remove(k)
                    break
                    
    #Clear any unused rows       
    for i in output[::-1]:
        if all([j == '-' for j in i]):
            output.remove(i)
    return output

#Merge datapoints
def merge_words(first,second):
    first_xmax = max([i[0] for i in first[0]])
    first_xmin = min([i[0] for i in first[0]])
    first_ymax = max([i[1] for i in first[0]])
    first_ymin = min([i[1] for i in first[0]])
    
    second_xmax = max([i[0] for i in second[0]])
    second_xmin = min([i[0] for i in second[0]])
    second_ymax = max([i[1] for i in second[0]])
    second_ymin = min([i[1] for i in second[0]])
    
    pos_arr = [[min(first_xmin,second_xmin),min(first_ymin,second_ymin)],[max(first_xmax,second_xmax),min(first_ymin,second_ymin)],[max(first_xmax,second_xmax),max(first_ymax,second_ymax)],[min(first_xmin,second_xmin),max(first_ymax,second_ymax)]]
    item_tuple = (first[1][0]+' '+second[1][0],(first[1][1]+second[1][1])*0.5)
    
    return [pos_arr,item_tuple]

def table_merger(table_data):
    data_copy = deepcopy(table_data)
    
    while(1):
        #Clear any unused rows 
        for i in data_copy[::-1]:
            if all([j == '-' for j in i]):
                data_copy.remove(i)
        #Merging is determined to have been completed when there are no rows left with just one element
        row_sum = []
        for i in data_copy:
            row_sum.append(sum([1 if j!='-' else 0 for j in i]))
        if all([i>1 for i in row_sum]):
            break
        
        #For all datapoints find 2 datapoint that are closest to each other and merge them
        min_dist = 1000000
        min_elem = []
        for i in data_copy[row_sum.index(min(row_sum))]:
            if i == '-':
                continue
            i_xave = sum([j[0] for j in i[0]])/4.0
            i_yave = sum([j[1] for j in i[0]])/4.0
            for j in data_copy[row_sum.index(min(row_sum))-1]:
                if j == '-':
                    continue
                j_xave = sum([k[0] for k in j[0]])/4.0
                j_yave = sum([k[1] for k in j[0]])/4.0
                
                dist = ((i_xave-j_xave)**2 + (i_yave-j_yave)**2)**0.5
                if dist<min_dist:
                    min_dist = dist
                    if i_yave<j_yave:
                        min_elem = [i,j]
                    else:
                        min_elem = [j,i]  
        new_entry = merge_words(min_elem[0],min_elem[1])
        for row,i in enumerate(data_copy):
            for col,j in enumerate(i):
                if j == min_elem[0]:
                    data_copy[row][col] = new_entry
                elif j == min_elem[1]:
                    data_copy[row][col] = '-'
#         print("running loop")        
    return data_copy

def formater(data):
    product_name_prompts = ['Description','Particular']
    product_code_prompts = ['Ref','Item','Code','Id','No']
    product_quantity_prompts = ['Quantity','Qty','Whole']
    product_unitprice_prompts = ['Cost','Price']
    product_cost_prompts = ['Amount','Amt','Nett']
    
    all_products = {}
    for col,i in enumerate(data['header']):
        for row,j in enumerate(data['row_data']):
            if row not in all_products:
                all_products[row] = {}
            if j==None or i==None:
                continue
            if any([similar(i,k) for k in product_name_prompts]) or any([k.casefold() in i.casefold() for k in product_name_prompts]):
                all_products[row]['product_name'] = j[col]
            elif any([similar(i,k) for k in product_code_prompts]) or any([k.casefold() in i.casefold() for k in product_code_prompts]):
                all_products[row]['product_code'] = j[col]
            elif any([similar(i,k) for k in product_quantity_prompts]) or any([k.casefold() in i.casefold() for k in product_quantity_prompts]):
                all_products[row]['product_quantity'] = j[col]
            elif any([similar(i,k) for k in product_unitprice_prompts]) or any([k.casefold() in i.casefold() for k in product_unitprice_prompts]):
                all_products[row]['product_unitprice'] = j[col]
            elif any([similar(i,k) for k in product_cost_prompts]) or any([k.casefold() in i.casefold() for k in product_cost_prompts]):
                all_products[row]['product_cost'] = j[col]
    output = []
    for i in all_products.values():
        output.append(i)
    return output

def similar(word_1,word_2):
    return SequenceMatcher(None, word_1.casefold(), word_2.casefold()).ratio()>0.6
