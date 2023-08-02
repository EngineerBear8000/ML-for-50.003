#!/usr/bin/env python
# coding: utf-8

# In[1]:


from PIL import Image, ImageOps
import numpy as np
import time
from copy import deepcopy
from difflib import SequenceMatcher
from paddleocr import PaddleOCR
import json 
ocr_model = PaddleOCR(lang='en',use_angle_cls=True,show_log=False)


# In[26]:


def run_ocr_merger(image,type_of_document='invoice'):
    image = ImageOps.exif_transpose(image)
    img = np.asarray(image)
    raw = ocr_model.ocr(img, cls=True)
    raw = order_by_tbyx(raw)[0]
    
    if type_of_document == 'invoice':
        start_identifiers = ['Description','Particular','Item']
        end_identifiers = ['Total']
    elif type_of_document == 'statement':
        start_identifiers = ['Document','Particulars']
        end_identifiers = ['Total']
    
    startpoint = query_word_single(raw,start_identifiers,instance_from_front=True)
    if startpoint:
        for i in raw[raw.index(startpoint)::-1]:
            startpoint_xmin = min([i[0] for i in startpoint[0]])
            i_xmax = max([j[0] for j in i[0]])
            if i_xmax>startpoint_xmin:
                break
            startpoint = i
    else:
        print("ERROR: No initialier word can be found.")
        return
    endpoint = query_word_single(raw[raw.index(startpoint)+5::],end_identifiers,instance_from_front=True)
    if not endpoint:
        endpoint = raw[-1]
    table_data = raw[raw.index(startpoint):raw.index(endpoint):]
    org_table_data = organise_table(table_data)
    merged_table = table_merger(org_table_data)
    
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


# In[14]:


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


# In[15]:


def query_word_single(data,words,instance_from_front):
    data_copy = deepcopy(data)
    if not instance_from_front:
        data_copy = data_copy[::-1]
    for word in words:
        for i in data_copy:
            if word.casefold() in i[1][0].casefold() or SequenceMatcher(None, word.casefold(), i[1][0].casefold()).ratio()>0.7:
                return i


# In[16]:


def organise_table(table_data):
    data_copy = deepcopy(table_data)
    
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
    
    max_rows = len(org_data)
    all_column_len = max([(n,len(i)) for n,i in enumerate(org_data)], key = lambda x: x[1])
    row_of_max_column = all_column_len[0]
    max_columns = all_column_len[1]
    output = [['-']*max_columns for _ in range(max_rows)]
            
    for col,i in enumerate(org_data[row_of_max_column]):
        i_xmax = max([k[0] for k in i[0]])
        i_xmin = min([k[0] for k in i[0]])
        for row,j in enumerate([j[0] for j in org_data]):
            j_ymax = max([k[1] for k in j[0]])
            j_ymin = min([k[1] for k in j[0]])
                
            for k in data_copy[::-1]:
                k_xmax = max([n[0] for n in k[0]])
                k_xmin = min([n[0] for n in k[0]])
                k_ymax = max([n[1] for n in k[0]])
                k_ymin = min([n[1] for n in k[0]])

                if not (k_xmax>i_xmax and k_xmin>i_xmax) and not (k_xmax<i_xmin and k_xmin<i_xmin) and not (k_ymax>j_ymax and k_ymin>j_ymax) and not (k_ymax<j_ymin and k_ymin<j_ymin):
                    if output[row][col] == '-':
                        output[row][col] = k
                        data_copy.remove(k)
                    else:
                        ymax = max([n[1] for n in output[row][col][0]])
                        if ymax<k_ymax:
                            new_entry = merge_words(output[row][col],k)
                        else:
                            new_entry = merge_words(k,output[row][col])
                        output[row][col] = new_entry
                        data_copy.remove(k)
    
    for i in output[::-1]:
        if all([j == '-' for j in i]):
            output.remove(i)
    
#     if len(org_data[0]) < sum([1 if i!='-' else 0 for i in output[0]]):
#         output[0] = org_data[0]
    return output


# In[17]:


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


# In[18]:


def table_merger(table_data):
    data_copy = deepcopy(table_data)
    
    while(1):
        for i in data_copy[::-1]:
            if all([j == '-' for j in i]):
                data_copy.remove(i)
        row_sum = []
        for i in data_copy:
            row_sum.append(sum([1 if j!='-' else 0 for j in i]))
        if all([i>1 for i in row_sum]):
            break
            
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


# In[19]:


def organise_data(data):
    vertical_output = {}
    horizontal_output = {}
    key_data = []
    for i in data:
        if any([j in i[1][0] for j in '1234567890']):
            vertical_output[i[1][0]] = ''
            horizontal_output[i[1][0]] = ''
            key_data.append(i)
    
    for i in key_data:
        possible_vertical = []
        i_xmin = min([j[0] for j in i[0]])
        i_xmax = max([j[0] for j in i[0]])
        i_ymin = min([j[1] for j in i[0]])
        for j in data:
            j_xmax = max([k[0] for k in j[0]])
            j_xmin = min([k[0] for k in j[0]])
            j_ymin = min([k[1] for k in j[0]])
            if i!=j and not (j_xmax<i_xmin and j_xmin<i_xmin) and not (j_xmax>i_xmax and j_xmin>i_xmax) and j_ymin<=i_ymin:
                possible_vertical.append(j) 
        if possible_vertical:
            closest_vertical = sorted(possible_vertical, key=lambda x: x[0][0][1])[-1]
            if len(vertical_output):
                vertical_output[i[1][0]] = closest_vertical[1][0] + " " + vertical_output[i[1][0]]
            else:
                vertical_output[i[1][0]] = closest_vertical[1][0]
    
    for i in key_data:
        possible_horizontal = []
        i_ymin = min([j[1] for j in i[0]])
        i_ymax = max([j[1] for j in i[0]])
        i_xmin = min([j[0] for j in i[0]])
        for j in data:
            j_ymax = max([k[1] for k in j[0]])
            j_ymin = min([k[1] for k in j[0]])
            j_xmin = min([k[0] for k in j[0]])
            if i!=j and not (j_ymax<i_ymin and j_ymin<i_ymin) and not (j_ymax>i_ymax and j_ymin>i_ymax) and j_xmin<=i_xmin:
                possible_horizontal.append(j) 
        if possible_horizontal:
            closest_horizontal = sorted(possible_horizontal, key=lambda x: x[0][0][0])[-1]
            if len(horizontal_output):
                horizontal_output[i[1][0]] = closest_horizontal[1][0] + " " + horizontal_output[i[1][0]]
            else:
                horizontal_output[i[1][0]] = closest_horizontal[1][0]
    
    output = {}
    for k,v in horizontal_output.items():
        if v=='':
            v = 'uncategorizable Data'
            if 'uncategorizable Data' in output:
                output[v] = output[v] + ' ' + k
            else:
                output[v] = k
        else:
            if not any([j in v for j in '1234567890']):
                output[v] = k
    return output


# In[31]:


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


# In[32]:


def similar(word_1,word_2):
    return SequenceMatcher(None, word_1.casefold(), word_2.casefold()).ratio()>0.6

