import tkinter as tk
from tkinter import filedialog, StringVar
from tkinter.filedialog import askopenfile

import argparse
from enum import Enum
import io, json

import argparse
from enum import Enum
import io, json

from google.cloud import vision
from PIL import Image, ImageDraw, ImageTk
from google.cloud.vision_v1 import AnnotateImageResponse

import time
from keras.models import load_model
from PIL import Image as im
import PIL.ImageOps
import numpy as np
import cv2

itemized = []
itemized_tip = []

parsed = {}
slider_val = 0
item_to_person = {}


def predict_digit(img):

  #load model

  model = load_model('final_model.h5')

  #resize image to 28x28 pixels

  result = np.full((img.size[1],img.size[1], 3), 255, dtype=np.uint8)
  center = (img.size[1]-img.size[0])//2
  result[:,center:center+img.size[0]]=img
  result = cv2.resize(result,(28,28))
  data = im.fromarray(result)

  #reshaping to support our model input and normalizing

  data = data.convert('L')
  data = PIL.ImageOps.invert(data)
  data = np.array(data)
  check1 = np.copy(data)
  data = data.reshape(1,28,28,1)
  data = data/255.0

  #predicting the class

  res = model.predict(data)[0]
  return np.argmax(res)

def find_tip_func(img, grayimg):

  #initialize arrays and set threshold

  tip=[]
  threshold=150
  grayimg = np.array(grayimg)
  img = np.array(img)

  #separate tip into individual digits

  start=0
  startnumflag=False
  end=0
  tophalf=0
  for col in range(len(grayimg[0])):
    hasnum=False
    for row in range(len(grayimg)):
      pixel=grayimg[row][col]
      if pixel<=threshold: #written pixel
        hasnum=True
        if row<=(len(grayimg)/2):  #check if in top half of image for period detection
          tophalf=1
        break
    if startnumflag==False and hasnum==False: #haven't found number yet
      start=col
    elif startnumflag==False and hasnum==True: #found beginning of number
      startnumflag=True
    elif startnumflag==True and hasnum==True: #haven't found end of number yet
      end=col
    else: #found end of number
      startnumflag=False
      if tophalf==0: #period detected
        tip.append(-1)
      else: #send cropped number to model
        tophalf=0
        data = im.fromarray(img[:, start:end])
        #data.save('out'+str(start)+'.png')
        tip.append(predict_digit(data))

  return tip

def find_tip(image):

  #read image

  try: 
    img = cv2.imread(image)
    contrast=1.2
    brightness=0
    img = cv2.addWeighted( img, contrast, img, 0, brightness) #adjust contrast and brightness
    grayimg = cv2.imread(image,0)
  except IOError:
    pass
  return find_tip_func(img, grayimg)

#start execution here

# print(find_tip('IMG_1970.jpg'))

def find_tip_box(bounds):
    #find index of subtotal text in itemized array
    tipIdx = 0;
    tipValIdx = 0;
    i = 0
    for item in itemized_tip:
        if(item["text"].lower() == "tip" or item["text"].lower() == "+tip" or item["text"].lower() == "+ tip"):
            tipIdx = i
            break;
        i+=1
    topTipY = itemized_tip[tipIdx]["bounding_box"].vertices[0].y
    print("FOUND TOP TIP: " + str(topTipY))
    #find index of subtotal value in itemized array
    i=0
    for item in itemized_tip:
        # avgItmY = (item["bounding_box"].vertices[0].y + item["bounding_box"].vertices[3].y)/2
        if(item["text"].lower() == "tip"):
            i+=1
            continue
        if(topTipY >= item["bounding_box"].vertices[0].y and topTipY <= item["bounding_box"].vertices[3].y and item["text"].__contains__(".") ):
            # print("BB 0Y:   " + str(item["bounding_box"].vertices[0].y))
            # print("BB 3Y:   " + str(item["bounding_box"].vertices[3].y))
            # print("avgSubY: " + str(avgSubY))
            # print(item)
            tipValIdx = i
            break;
        i+=1
    print("FOUND TOP TIP: " + str(topTipY))
    print("TIP VALUE: " + itemized_tip[tipValIdx]["text"])
    sub_tl = {}
    sub_tl['x'] = itemized_tip[tipValIdx]["bounding_box"].vertices[0].x
    sub_tl['y'] = itemized_tip[tipValIdx]["bounding_box"].vertices[0].y
    #top right 
    sub_tr = {}
    sub_tr['x'] = itemized_tip[tipValIdx]["bounding_box"].vertices[1].x
    sub_tr['y'] = itemized_tip[tipValIdx]["bounding_box"].vertices[1].y
    #bottom left 
    sub_bl = {}
    sub_bl['x'] = itemized_tip[tipValIdx]["bounding_box"].vertices[3].x
    sub_bl['y'] = itemized_tip[tipValIdx]["bounding_box"].vertices[3].y
    #top left 
    sub_br = {}
    sub_br['x'] = itemized_tip[tipValIdx]["bounding_box"].vertices[2].x
    sub_br['y'] = itemized_tip[tipValIdx]["bounding_box"].vertices[2].y
    vertices = [sub_tl, sub_tr, sub_br, sub_bl]

    bounds.append({})
    bi = 0
    bounds[bi]['vertices'] = vertices
    vertices = [sub_tl, sub_tr, sub_br, sub_bl]
    return bounds


def find_features(bounds):
    #find index of subtotal text in itemized array
    subIdx = 0;
    subValIdx = 0;
    i = 0
    for item in itemized:
        if(item["text"].lower() == "sub/ttl" or item["text"].lower() == "subtotal" or item["text"].lower() == "sub total"  or item["text"].lower() == "sub-total" or item["text"].lower() == "sub"):
            subIdx = i
            break;
        i+=1

    taxIdx = 0;
    taxValIdx = 0;
    i = 0
    for item in itemized:
        if(item["text"].lower() == "tax" or item["text"].lower() == "taxes"):
            taxIdx = i
            break;
        i+=1

    # average of the top of teh subtotal y and bottom of subtotal y
    avgSubY = (itemized[subIdx]["bounding_box"].vertices[0].y + itemized[subIdx]["bounding_box"].vertices[3].y)/2

    #find index of subtotal value in itemized array
    i=0
    for item in itemized:
        # avgItmY = (item["bounding_box"].vertices[0].y + item["bounding_box"].vertices[3].y)/2
        if(item["text"].lower() == "sub/ttl" or item["text"].lower() == "subtotal" or item["text"].lower() == "sub total"  or item["text"].lower() == "sub-total"or item["text"].lower() == "sub"):
            i+=1
            continue
        if(avgSubY >= item["bounding_box"].vertices[0].y and avgSubY <= item["bounding_box"].vertices[3].y and item["text"].__contains__(".") ):
            # print("BB 0Y:   " + str(item["bounding_box"].vertices[0].y))
            # print("BB 3Y:   " + str(item["bounding_box"].vertices[3].y))
            # print("avgSubY: " + str(avgSubY))
            # print(item)
            subValIdx = i
            break;
        i+=1


    # average of the top of teh tax y and bottom of subtotal y
    avgTaxY = (itemized[taxIdx]["bounding_box"].vertices[0].y + itemized[taxIdx]["bounding_box"].vertices[3].y)/2

    #find index of subtotal value in itemized array
    i=0
    for item in itemized:
        # avgItmY = (item["bounding_box"].vertices[0].y + item["bounding_box"].vertices[3].y)/2
        if(item["text"].lower() == "tax" or item["text"].lower() == "taxes"):
            i+=1
            continue
        if(avgTaxY >= item["bounding_box"].vertices[0].y and avgTaxY <= item["bounding_box"].vertices[3].y and item["text"].__contains__(".") ):
            # print("BB 0Y:   " + str(item["bounding_box"].vertices[0].y))
            # print("BB 3Y:   " + str(item["bounding_box"].vertices[3].y))
            # print("avgSubY: " + str(avgSubY))
            # print(item)
            taxValIdx = i
            break;
        i+=1

    #find the coordinates of each corner
    #top left 
    #find the coordinates of each corner
    #top left 
    sub_tl = {}
    sub_tl['x'] = itemized[subIdx]["bounding_box"].vertices[0].x
    sub_tl['y'] = itemized[subIdx]["bounding_box"].vertices[0].y
    #top right 
    sub_tr = {}
    sub_tr['x'] = itemized[subValIdx]["bounding_box"].vertices[1].x
    sub_tr['y'] = itemized[subValIdx]["bounding_box"].vertices[1].y
    #bottom left 
    sub_bl = {}
    sub_bl['x'] = itemized[subIdx]["bounding_box"].vertices[3].x
    sub_bl['y'] = itemized[subIdx]["bounding_box"].vertices[3].y
    #top left 
    sub_br = {}
    sub_br['x'] = itemized[subValIdx]["bounding_box"].vertices[2].x
    sub_br['y'] = itemized[subValIdx]["bounding_box"].vertices[2].y
    vertices = [sub_tl, sub_tr, sub_br, sub_bl]

    bounds.append({})
    bi = 0
    bounds[bi]['vertices'] = vertices
    bi += 1
    #find middle of box value box for subtotal 
    sub_tot_mid_x = (itemized[subValIdx]["bounding_box"].vertices[0].x + itemized[subValIdx]["bounding_box"].vertices[1].x)/2
    sub_tot_mid_y = (itemized[subValIdx]["bounding_box"].vertices[0].y + itemized[subValIdx]["bounding_box"].vertices[3].y)/2
    
    parsed['subtotal'] = itemized[subValIdx]['text']

    tax_tl = {}
    tax_tl['x'] = itemized[taxIdx]["bounding_box"].vertices[0].x
    tax_tl['y'] = itemized[taxIdx]["bounding_box"].vertices[0].y
    #top right 
    tax_tr = {}
    tax_tr['x'] = itemized[taxValIdx]["bounding_box"].vertices[1].x
    tax_tr['y'] = itemized[taxValIdx]["bounding_box"].vertices[1].y
    #bottom left 
    tax_bl = {}
    tax_bl['x'] = itemized[taxIdx]["bounding_box"].vertices[3].x
    tax_bl['y'] = itemized[taxIdx]["bounding_box"].vertices[3].y
    #top left 
    tax_br = {}
    tax_br['x'] = itemized[taxValIdx]["bounding_box"].vertices[2].x
    tax_br['y'] = itemized[taxValIdx]["bounding_box"].vertices[2].y
    vertices = [tax_tl, tax_tr, tax_br, tax_bl]

    bounds.append({})
    bounds[bi]['vertices'] = vertices
    bi += 1
    #find middle of box value box for subtotal 
    tax_tot_mid_x = (itemized[taxValIdx]["bounding_box"].vertices[0].x + itemized[taxValIdx]["bounding_box"].vertices[1].x)/2
    tax_tot_mid_y = (itemized[taxValIdx]["bounding_box"].vertices[0].y + itemized[taxValIdx]["bounding_box"].vertices[3].y)/2
    
    parsed['tax'] = itemized[taxValIdx]['text']




    parsed['items'] = []
    pi = -1

    #find item prices
    i=0
    for item in itemized:
        # avgItmY = (item["bounding_box"].vertices[0].y + item["bounding_box"].vertices[3].y)/2
        if(item["text"].lower() == "sub/ttl" or item["text"].lower() == "subtotal" or item["text"].lower() == "sub total"  or item["text"].lower() == "sub-total" or item["text"].lower() == "sub"):
            i+=1
            continue
        if(item["bounding_box"].vertices[0].y < itemized[subValIdx]["bounding_box"].vertices[0].y and sub_tot_mid_x >= item["bounding_box"].vertices[0].x and sub_tot_mid_x <= item["bounding_box"].vertices[1].x and item["text"].__contains__(".")):
            avgItmY = (item["bounding_box"].vertices[0].y + item["bounding_box"].vertices[3].y)/2
            parsed['items'].append({})
            pi += 1
            parsed['items'][pi]['name'] = ''
            num_words_in_item = 0
            for it in itemized:
                # avgItmY = (item["bounding_box"].vertices[0].y + item["bounding_box"].vertices[3].y)/2
                if(it["text"].lower() == "sub/ttl" or it["text"].lower() == "subtotal" or it["text"].lower() == "sub total"  or it["text"].lower() == "sub-total" or item["text"].lower() == "sub"):
                    i+=1
                    continue
                if(avgItmY >= it["bounding_box"].vertices[0].y and avgItmY <= it["bounding_box"].vertices[3].y and item["bounding_box"].vertices[0].x > it["bounding_box"].vertices[2].x):
                    # print("ITEM: " + it["text"])
                    parsed['items'][pi]['name'] += it["text"] + " "
                    if(num_words_in_item == 0):
                        sub_tl = {}
                        sub_tl['x'] = it["bounding_box"].vertices[0].x
                        sub_tl['y'] = it["bounding_box"].vertices[0].y
                        #top right 
                        sub_tr = {}
                        sub_tr['x'] = item["bounding_box"].vertices[1].x
                        sub_tr['y'] = item["bounding_box"].vertices[1].y
                        #bottom left 
                        sub_bl = {}
                        sub_bl['x'] = it["bounding_box"].vertices[3].x
                        sub_bl['y'] = it["bounding_box"].vertices[3].y
                        #top left 
                        sub_br = {}
                        sub_br['x'] = item["bounding_box"].vertices[2].x
                        sub_br['y'] = item["bounding_box"].vertices[2].y
                        vertices = [sub_tl, sub_tr, sub_br, sub_bl]

                        bounds.append({})
                        bounds[bi]['vertices'] = vertices
                        bi += 1
                    num_words_in_item += 1
                i+=1
            # print("ITEM PRICE   " + item["text"])
            parsed['items'][pi]['price'] = item["text"]
        i+=1    
    
    # # #create a bounding box for the two
    # print(itemized[subIdx])
    # print(itemized[subValIdx])
    # # #create a bounding box for the two
    # print(itemized[subIdx])
    # print(itemized[subValIdx])
    # my_font1=('times', 12, 'normal')
    # l1 = tk.Label(my_w,text=parsed,width=30,font=my_font1)  
    # l1.grid(row=5,column=1)
    print(parsed)
    return parsed
def draw_boxes(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        # print(bound)
        draw.polygon(
            [
                bound.vertices[0].x,
                bound.vertices[0].y,
                bound.vertices[1].x,
                bound.vertices[1].y,
                bound.vertices[2].x,
                bound.vertices[2].y,
                bound.vertices[3].x,
                bound.vertices[3].y,
            ],
            None,
            color,
        )
    return image
def draw_boxes1(image, bounds, color):
    """Draw a border around the image using the hints in the vector list."""
    draw = ImageDraw.Draw(image)

    for bound in bounds:
        # print(bound)
        draw.polygon(
            [
                bound['vertices'][0]['x'],
                bound['vertices'][0]['y'],
                bound['vertices'][1]['x'],
                bound['vertices'][1]['y'],
                bound['vertices'][2]['x'],
                bound['vertices'][2]['y'],
                bound['vertices'][3]['x'],
                bound['vertices'][3]['y'],
            ],
            fill=None,
            outline=color,
            width = 5
        )
    return image

def get_document_bounds(image_file, feature):
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient()

    bounds = []
    with io.open(image_file, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    response_json = AnnotateImageResponse.to_json(response)
    resp = json.loads(response_json)
    document = response.full_text_annotation

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    w = ""
                    obj = {}
                    for symbol in word.symbols:
                        if feature == FeatureType.SYMBOL:
                            bounds.append(symbol.bounding_box)
                            w = w + str(symbol.text)
                    if feature == FeatureType.WORD:
                        bounds.append(word.bounding_box)
                    obj["text"] = w
                    obj["bounding_box"] = word.bounding_box
                    if(obj["text"] != ''):
                        itemized.append(obj)

                if feature == FeatureType.PARA:
                    bounds.append(paragraph.bounding_box)

            if feature == FeatureType.BLOCK:
                bounds.append(block.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    # print(itemized)
    return bounds


def get_document_bounds1(image_file, feature):
    """Returns document bounds given an image."""
    client = vision.ImageAnnotatorClient()

    bounds = []
    with io.open(image_file, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    response_json = AnnotateImageResponse.to_json(response)
    resp = json.loads(response_json)
    document = response.full_text_annotation

    # Collect specified feature bounds by enumerating all document features
    for page in document.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    w = ""
                    obj = {}
                    for symbol in word.symbols:
                        if feature == FeatureType.SYMBOL:
                            bounds.append(symbol.bounding_box)
                            w = w + str(symbol.text)
                    if feature == FeatureType.WORD:
                        bounds.append(word.bounding_box)
                    obj["text"] = w
                    obj["bounding_box"] = word.bounding_box
                    if(obj["text"] != ''):
                        itemized_tip.append(obj)

                if feature == FeatureType.PARA:
                    bounds.append(paragraph.bounding_box)

            if feature == FeatureType.BLOCK:
                bounds.append(block.bounding_box)

    # The list `bounds` contains the coordinates of the bounding boxes.
    # print(itemized)
    return bounds


def select_person(cur_sel):
	print(cur_sel)
	global sel_num
	sel_num = cur_sel
	current_selection.set("Person "+ str(cur_sel + 1))
	person_label.update()

def assign_item(item):
	# if item['name'] + " $" + item['price'] not in item_to_person:
	item_to_person[item['name'] + " $" + item['price']] = [sel_num , item['price']]
	summarize(item_to_person)
def summarize(dic):

	person_to_item = {}
	#create a key for each person
	for num in range(slider.get()):
		person_to_item[num] = []
	print("STEP1: " + str(person_to_item))
	print("SELNUM: " + str(slider.get()))
	#Add the iteams that correspond to each person
	for key in dic:
		item = {}
		item['price'] = dic[key][1]
		item['name'] = key
		if(dic[key][0] not in person_to_item):
			continue
		# print(person_to_item[dic[key][0]])
		person_to_item[dic[key][0]].append(item)
	print("STEP2: " + str(person_to_item))
	sf = tk.Frame(my_w)

	sum_txt = ""
	sum_txt_var= StringVar()

	#Populate sum_txt
	sum_txt = "SUMMARY\n"

	for person in person_to_item:
		person_sum = 0
		tip_per_person = float(tip_amount_final) / float(slider.get())
		for item in person_to_item[person]:
			person_sum += float(item['price']) 
		tax_per_person = float(parsed['tax']) * float((float(person_sum) / float(parsed['subtotal'])))
		sum_txt += "Person " + str(person + 1) + "\t owes $" + str(round(person_sum + tip_per_person + tax_per_person, 2)) + "\n"
		for item in person_to_item[person]:
			sum_txt += "\t\t" + item['name'] + "\n"
		sum_txt += "\t\tTip " +  str(round(tip_per_person, 2)) + "\n"
		sum_txt += "\t\tTax " +  str(round(tax_per_person, 2)) + "\n"

	sum_txt_var.set(sum_txt)
	sf.grid(row=3, column=4, rowspan=5, sticky = "E") 
	my_font1=('times', 15, 'normal')
	summary_label = tk.Label(sf,textvariable=sum_txt_var,width=60,font=my_font1, justify="left")  
	summary_label.grid(row=2,columnspan=2)
	# print(person_to_item)
	print("DIC" + str(item_to_person))
	print("BUP" + str(person_to_item))


def render_doc_text(filein, fileout):
    image = Image.open(filein)
    # bounds = get_document_bounds(filein, FeatureType.BLOCK)
    # draw_boxes(image, bounds, "blue")
    # bounds = get_document_bounds(filein, FeatureType.PARA)
    # draw_boxes(image, bounds, "red")
    # bounds = get_document_bounds(filein, FeatureType.WORD)


    grouped_bounds = []
    if(fileout == "output.jpg"):
    	bounds = get_document_bounds(filein, FeatureType.SYMBOL)
    	draw_boxes(image, bounds, "yellow")
    	parsed = find_features(grouped_bounds)
    	draw_boxes1(image, grouped_bounds, "red")
    else:
    	bounds = get_document_bounds1(filein, FeatureType.SYMBOL)
    	draw_boxes(image, bounds, "yellow")
    	print(str(itemized_tip))

    	global img_tip
    	global tip_amount
    	tip_box = find_tip_box(grouped_bounds)
    	im = cv2.imread(filein)
    	print("VERTICIES" + str(tip_box[0]["vertices"]))
    	print("x1" + str(tip_box[0]["vertices"][0]["x"]) + "x2" + str(tip_box[0]["vertices"][1]["x"]))
    	draw_boxes1(image, tip_box, "red")
    	crop_image = im[ tip_box[0]["vertices"][1]["y"]:tip_box[0]["vertices"][3]["y"], tip_box[0]["vertices"][0]["x"]:tip_box[0]["vertices"][1]["x"]]
    	cv2.imwrite("tip_nums.jpg", crop_image)
    	image = image.convert('RGB')
    	image.save(fileout)
    	time.sleep(1)
    	img_tip=Image.open(fileout)
    	img_resized=img_tip.resize((200,400)) # new width & height
    	img_tip=ImageTk.PhotoImage(img_resized)
    	b2 =tk.Button(my_w,image=img_tip) # using Button 
    	b2.grid(row=5,column=2)
    	return 
    # elif(fileout == "output_tip.jpg"):
    # 	parsed = find_features(grouped_bounds)

    if fileout != 0:
    	global img
    	global person_label
    	global current_selection
    	global summary_text
    	image = image.convert('RGB')
    	image.save(fileout)
    	time.sleep(1)
    	img=Image.open(fileout)
    	img_resized=img.resize((200,400)) # new width & height
    	img=ImageTk.PhotoImage(img_resized)
    	b2 =tk.Button(my_w,image=img) # using Button 
    	b2.grid(row=5,column=1)


    	current_selection = StringVar()
    	current_selection.set("NO PERSON\n\tSELECTED ")

    	buttonframe = tk.Frame(my_w)
    	buttonframe.grid(row=4, column=3, rowspan=2, sticky = "N")  
    	# print("SLIDER VL" + str(slider_val))
    	for j in range(slider.get()):
    		tk.Button(buttonframe, text = "Person"+str((j+1)), command=lambda j=j:select_person(j) ).grid(row=1, column=j)
    	my_font1=('times', 16, 'normal')
    	person_label = tk.Label(buttonframe,textvariable=current_selection,width=20,font=my_font1)  
    	person_label.grid(row=2,columnspan=3)

    	i = 3
    	for item in parsed['items']:
    		tk.Button(buttonframe, text = item['name'] + "     $" + item['price'], command=lambda item=item:assign_item(item)).grid(row=i, columnspan=3)
    		i+=1


    else:
        image.show()

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

# def set_slider(val):
# 	global slider_val
# 	slider_val = val

my_w = tk.Tk()
my_w.geometry("1200x600")  # Size of the window 
my_w.title('Computer Vision Project')
my_font1=('times', 18, 'bold')
l1 = tk.Label(my_w,text='Receipt Parser',width=30,font=my_font1)  
l1.grid(row=1,column=1)


my_font1=('times', 16, 'normal')
l1 = tk.Label(my_w,text='How many people are splitting the bill?',width=40,font=my_font1)  
l1.grid(row=2,column=1)


slider = tk.Scale( my_w, from_=1, to=5, orient='horizontal')
slider.grid(row=3,column=1) 

b1 = tk.Button(my_w, text='Upload Itemized Receipt', 
   width=20,command = lambda:upload_file())
b1.grid(row=4,column=1) 

b1 = tk.Button(my_w, text='Upload Tip Receipt', 
   width=20,command = lambda:upload_file_tip())
b1.grid(row=4,column=2) 

b1 = tk.Button(my_w, text='Parse', 
   width=20,command = lambda:parse_receipt())
b1.grid(row=6,column=1) 




def upload_file():
	global filename
	global img
	f_types = [('Jpg Files', '*.jpg')]
	filename = filedialog.askopenfilename(filetypes=f_types)
	img=Image.open(filename)
	img_resized=img.resize((200,400)) # new width & height
	img=ImageTk.PhotoImage(img_resized)
	b2 =tk.Button(my_w,image=img) # using Button 
	b2.grid(row=5,column=1)

def upload_file_tip():
	global filename_tip
	global img_tip
	f_types = [('Jpg Files', '*.jpg')]
	filename_tip = filedialog.askopenfilename(filetypes=f_types)
	img_tip=Image.open(filename_tip)
	img_resized=img_tip.resize((200,400)) # new width & height
	img_tip=ImageTk.PhotoImage(img_resized)
	b2 =tk.Button(my_w,image=img_tip) # using Button 
	b2.grid(row=5,column=2)

def parse_receipt():
	print("FILENAME " + filename)
	render_doc_text(filename, "output.jpg")
	render_doc_text(filename_tip, "output_tip.jpg")
	global tip_amount_final
	strint = ""
	for digit in find_tip('tip_nums.jpg'):
		if digit == -1:
			strint += "."
			continue
		strint += str(digit)
	tip_amount_final = float(strint)
	print("TIP FOUND: " + str(tip_amount_final))

my_w.mainloop()  # Keep the window open