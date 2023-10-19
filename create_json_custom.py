import json
import glob
import os

def get_image_list_from_folder(img_folder):
    """
    Obtain list of image path from the custom folder
    """
    img_pattern = os.path.join(img_folder, '*.jpg')
    img_list = [img_path for img_path in glob.glob(img_pattern)]
    return img_list

def write_to_json(img_list, output_json):
    """
    Save the image list to a JSON file
    """
    with open(output_json, 'w') as json_file:
        json.dump(img_list, json_file)

if __name__ == '__main__':
    # Path to the folder that contains images
    IMG_FOLDER = '/Users/kshitiz/Documents/GitHub/CSRNet-pytorch/test_data/images'

    # Path to the final JSON file
    OUTPUT_JSON = '/Users/kshitiz/Documents/GitHub/CSRNet-pytorch/test_data/test.json'

    image_list = get_image_list_from_folder(IMG_FOLDER)
    for img in image_list:
        print(img)

    write_to_json(image_list, OUTPUT_JSON)
    