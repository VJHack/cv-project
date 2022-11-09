import cv2
import requests
import sys
url = "https://api.mindee.net/v1/products/mindee/expense_receipts/v4/predict"
extracted_info = {}
coords = [[]]

def draw(coordinates, output_img):
    # Open the image
	cv_image = cv2.imread(output_img)
    # Create image mask
	overlay = cv_image.copy()
	h, w = cv_image.shape[:2]
    # Iterate through coordinates and draw boxes
	print(coords)
	for coord in coords:
		if(len(coord) == 0):
			continue
		pt1 = (int(w*coord[0][0]), int(h*coord[0][1]))
		pt2 = (int(w*coord[2][0]), int(h*coord[2][1]))
		cv2.rectangle(overlay, pt1, pt2, (70, 230, 244), cv2.FILLED)
    # Overlay mask and original image with alpha
	final_image = cv2.addWeighted(overlay, 0.5, cv_image, 0.5, 0)
    # Display the final image
	cv2.imshow("highlghted_image", cv2.resize(final_image, (400, int(400*h/w))))
	cv2.waitKey(10000)


def parse(js):
    coords.append(js["document"]["inference"]["pages"][0]["prediction"]["total_amount"]["polygon"])
    coords.append(js["document"]["inference"]["pages"][0]["prediction"]["total_net"]["polygon"])
    coords.append(js["document"]["inference"]["pages"][0]["prediction"]["tip"]["polygon"])
    coords.append(js["document"]["inference"]["pages"][0]["prediction"]["tip"]["polygon"])
    coords.append(js["document"]["inference"]["pages"][0]["prediction"]["tip"]["polygon"])
    coords.append(js["document"]["inference"]["pages"][0]["prediction"]["taxes"][0]["polygon"])

def make_request():
	with open(sys.argv[1], "rb") as curfile:
	    file = {"document": curfile}
	    headers = {"Authorization": "Token 44efd7bbf833e76986f75cce89890a7d"}
	    response = requests.post(url, files=file, headers=headers)
	    if response.status_code != 201:
	        print("Request error")
	        return None
	    else:
	        json_response = response.json()
	        return json_response
	        # features, coords = get_features(json_response)
	        # print("Date:", features["date"])
	        # print("Time:", features["time"])
	        # print("Total amount:", features["total_amount"])
	        # print("Category:", features["category"])
	        # highlight_features("receipt1.jpg", coords)

json_res = make_request()
if(json_res != None):
	parse(json_res)
	draw(coords, sys.argv[1])
