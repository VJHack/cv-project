import argparse
from enum import Enum
import io, json

from google.cloud import vision
from PIL import Image, ImageDraw
from google.cloud.vision_v1 import AnnotateImageResponse

itemized = []
parsed = {}





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
    print(parsed)
    return 0
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
            None,
            color,
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


def render_doc_text(filein, fileout):
    image = Image.open(filein)
    # bounds = get_document_bounds(filein, FeatureType.BLOCK)
    # draw_boxes(image, bounds, "blue")
    # bounds = get_document_bounds(filein, FeatureType.PARA)
    # draw_boxes(image, bounds, "red")
    # bounds = get_document_bounds(filein, FeatureType.WORD)
    bounds = get_document_bounds(filein, FeatureType.SYMBOL)
    draw_boxes(image, bounds, "yellow")

    grouped_bounds = []
    find_features(grouped_bounds)
    draw_boxes1(image, grouped_bounds, "red")

    if fileout != 0:
        image.save(fileout)
    else:
        image.show()

class FeatureType(Enum):
    PAGE = 1
    BLOCK = 2
    PARA = 3
    WORD = 4
    SYMBOL = 5

render_doc_text("receipt1.jpg", 0)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("detect_file", help="The image for text detection.")
#     parser.add_argument("-out_file", help="Optional output file", default=0)
#     args = parser.parse_args()

#     render_doc_text(args.detect_file, args.out_file)