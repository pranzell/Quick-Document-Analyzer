#################################################################
#   @version: 1.0
#   @date: 26/02/2022
#################################################################

# Imports
import sys
import os
import uuid
import re
import time
import requests
import random
import re
import ntpath
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from io import BytesIO
import http.client, urllib.request, urllib.parse, urllib.error, base64
import logging
logger = logging.getLogger('ROOT')

# AZURE
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

# OCR formatter
from ocrlayout.bboxhelper import BBOXOCRResponse,BBoxHelper

# OpenAI GPT-3 API(s) free license - trial version
import openai


#### <--------- DEAFULT INSIGHT OPTIONS ---------> ####
# TO BE SENT DURING API REQUEST
INSIGHT_OPTIONS = {

    "default-summary": "Summarize TEXT in 300 words.",
    "default-faq": "Generate 5 FAQs from the TEXT.",
    "default-keypoints": "Tl;dr",
    "default-keywords": "Find important words in the TEXT.",
    "default-phrases": "Find important phrases in the TEXT.",
    "default-sentiment": "Find sentiment words in the TEXT.",
    "default-abbrv": "Find abbreviations in the TEXT."
}
#### <--------------------------------------------> ####


class run_ocr_process:
    
    def __init__(self, FILE_PATH):
        if not FILE_PATH.startswith(('https', 'http', 'www', 'ftp', 'localhost')) and not os.path.exists(FILE_PATH):
            raise Exception("ERROR: File not found!")

        self.supported_formats = ['png', 'jpg', 'jpeg', 'heic', 'gif', 'bmp', 'tiff', 'pdf', 'txt', 'md']
        
        self.file_path, \
        self.file_basename,  \
        self.file_ext = str(FILE_PATH), \
                        ntpath.basename(FILE_PATH).split(".")[0].lower().strip(), \
                        ntpath.basename(FILE_PATH).split(".")[1].lower().strip()
        
        if self.file_ext not in self.supported_formats:
            raise Exception("ERROR: Unsupported file format!")
            
        key = os.environ.get("OCR_KEY")
        self.endpoint = os.environ.get("OCR_ENDPOINT")

        # Set credentials & create client
        self.credentials = CognitiveServicesCredentials(key)
        self.client = ComputerVisionClient(self.endpoint, self.credentials)

        return

    def run_ocr(self):
        """
        Runs OCR READ API V3.2 using client libraries.
        :return: result - obj: Azure ReadAPI v3.2 response
        """
        start = time.time()
        response = None
        result = None

        if self.file_path.startswith(('https', 'http', 'www', 'ftp', 'localhost')):
            # > hosted doc url (blob)
            self.file_type = 'online'
            response = self.client.read(self.file_path, raw=True)

        elif self.file_ext in ['png', 'jpg', 'jpeg', 'heic', 'gif', 'bmp', 'tiff', 'pdf']:
            # > image/pdf
            fp = open(self.file_path,'rb')
            response = self.client.read_in_stream(fp, raw=True)
            fp.close()

        elif self.file_ext in ['txt', 'md']:
            # > local plain text files
            with open(self.file_path, "r") as f:
                response = f.readlines()

        elif self.file_ext in ['doc', 'docs', 'docx']:
            # > Word file - coming soon
            sys.exit()

        else:
            raise Exception("ERROR: Unsupported file format!")

        # Get the operation location and ID from the response:
        operation_location = response.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        print("Success - Submitted a request | Operation ID: %s |" % operation_id)

        # read 'results' using operation-id:
        while True:
            result = self.client.get_read_result(operation_id)
            if result.status.lower () not in ['notstarted', 'running']:
                break
            print ('Waiting for result...')
            time.sleep(1)

        print("%%time taken(s): ", (time.time() - start) )
        return result

    def format_boundingBox(self, ocr_result):
        """
        Cleans and formats (sorting) on BB coordinates/font-size.
        :prarm:  ocr_result - obj: origninal Azure ReadAPI v3.2 response obj
        :return: bboxresponse.text - str: cleaned corpus (with delimitters & context info)
        """
        # ref - https://puthurr.github.io/
        COMPUTERVISION_SUBSCRIPTION_KEY = os.environ["OCR_KEY"]
        COMPUTERVISION_LOCATION = "eastus"
        
        ocr_str = json.dumps(ocr_result.serialize())
        
        # Create BoundingBox OCR Response (intersect with coordinates)
        bboxresponse = BBoxHelper().processAzureOCRResponse(ocr_str)
        return bboxresponse.text

    def process_ocr_results(self, ocr_result):
        """
        Extracts meaningful info from Azure's ReadAPI v3.2 response object.
        :prarm:  ocr_result - obj: origninal Azure ReadAPI v3.2 response obj
        :return: output_df - pd.DataFrame: line-by-line text with BB coords; corpus_json - dict; ocr_output - dict;
        """
        # Displays text captured and its bounding box
        result = ocr_result
        output_df = pd.DataFrame()
        
        # Print the detected text, line by line
        if result.status == OperationStatusCodes.succeeded:
            
            for readResult in result.analyze_result.read_results:
                page_change_marker = "\n\n"
                for line in readResult.lines:
                    # print(line.text)
                    # print(">>", line.bounding_box)
                    """
                    bounding box:
                    X top left, Y top left, 
                    X top right, Y top right, 
                    X bottom right, Y bottom right, 
                    X bottom left, Y bottom left
                    """
                    data = {
                        "lines": page_change_marker + line.text,
                        "top_left_XY": (line.bounding_box[0], line.bounding_box[1]),
                        "top_right_XY": (line.bounding_box[2], line.bounding_box[3]),
                        "bottom_right_XY": (line.bounding_box[4], line.bounding_box[5]),
                        "bottom_left_XY": (line.bounding_box[6], line.bounding_box[7])
                    }                      
                    output_df = output_df.append(data, ignore_index=True)
                    page_change_marker=""
        
        
        # :: output ::
        #  'output_df'    : dataframe - stores line-by-line info with BB info XY (top-l, top-r, bottom-r, bottom-l)
        #  'corpus_json'  : dict - raw original response
        #  'ocr_output'   : dict - usable summarized-response 
        #
        corpus_json = result.as_dict()
        if self.file_ext=='pdf': 
            print("\nPDF file found! >> No. of pages= ", len(corpus_json['analyze_result']['read_results']) )
        
        ocr_output = {
            'title': self.file_basename + "." + self.file_ext,
            'format': self.file_ext,
            'isPdf': True if self.file_ext=='pdf' else False,
            'corpus': "\n".join(output_df['lines'].tolist()),
            'lines': output_df.to_dict(orient='records')
        }

        # CLEAN TEXT [ALTERNTATE]
        #   'corpus': self.format_boundingBox(ocr_result),
        
        return output_df, corpus_json, ocr_output

    def display_results(self, ocr_result):
        """
        Display BB overlapped on text. (**pdf needs poppler installed!**)
        :prarm:  ocr_result - obj: origninal Azure ReadAPI v3.2 response obj
        :return: None
        """
        
        result = ocr_result
        
        # Extract the word bounding boxes and text.
        word_infos=[]
        for readResult in result.analyze_result.read_results:
            for line in readResult.lines:
                for word_info in line.words:
                    word_infos.append(word_info)

        # Display the image and overlay it with the extracted text.
        plt.figure(figsize=(10, 8))
        image = Image.open(self.file_path)
        ax = plt.imshow(image, alpha=0.5)
        for word in word_infos:
            word = word.as_dict()
            bbox = [int(num) for num in word["bounding_box"]]
            text = word["text"]
            origin = (bbox[0], bbox[1])
            patch = Rectangle(origin, bbox[2], bbox[3],
                              fill=False, linewidth=2, color='y')
            ax.axes.add_patch(patch)
            plt.text(origin[0], origin[1], text, fontsize=20, weight="bold", va="top")
        plt.show()
        plt.axis("off")
        return

    def main(self):
        
        # run Azure's OCR
        ocr_result = self.run_ocr()
        
        # format bounding-box coords ** takes approx. 2mins for 4 page pdf **
        # ocr_clean_text = format_boundingBox(ocr_result)
        
        # extract info
        output_df, corpus_json, ocr_output = self.process_ocr_results(ocr_result)
        
        # if self.file_ext not in ['pdf'] and self.file_type!='online':
        #    self.display_results(ocr_result)
        
        return ocr_output #, output_df


class extract_metadata:
    
    def __init__(self, OPENAI_INSTANCE, OCR_RESULT_JSON):

        if not OCR_RESULT_JSON:
            raise Exception("ERROR: Please pass a 'OCR Ouptut'")

        # License key
        # openai.organization = NOT_NEEDED_FOR_MS
        self.openai = OPENAI_INSTANCE
        self.openai.api_key = os.environ.get("OPENAI_KEY")
        
        # :: config ::
        self.engine = "text-davinci-001"   # engine choose gpt3 from list, older - gpt2.1
        self.temp = 0.01          # higher values means the model will take more risks, 0.9 means creative, 0 is well-defined.
        self.max_tokens = 250     # max number of tokens to generate in completion API.
        self.doc_token_size = 1450  # free license supports upto 2049 tokens
        
        # ocr'ed document
        self.doc_title = OCR_RESULT_JSON['title']
        corpus = OCR_RESULT_JSON['corpus'].strip("\n\n").strip("\n")
        corpus = " ".join(corpus.split(" ")[:self.doc_token_size])
        self.doc = "TEXT:\n\n###\n\n" + corpus + "\n\n###\n\n"
        
        return
    
    def run_on_doc(self, prompt, insight=None):
        """
        Runs OpenAI on given input text.
        :param: prompt - str: input text;  insight - str: default key;
        :return: output - str: response from OpenAI's GPT-3 davinci-001 model-completion service.
        """

        # max words
        # max_word_limit = self.max_tokens
        # if insight=="summary":
        #    max_word_limit = 500

        # run OpenAI api
        response = self.openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=self.temp,
            max_tokens=self.max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0)
        
        # >>> REPSONE JSON <<<
        output_json = response
        # print(output_json)
        
        available_choices = len(response.choices)
        if available_choices > 1:
            selected_choice = random.choice(response.choices)
        else:
            selected_choice = response.choices[0]
            
        # Final output
        output = selected_choice.text.strip().strip("\n").strip()
        # print("Available choices --->", available_choices)
        # print("Response\n###\n", output, "\n###")
        
        return output
    
    def execute(self, prompt, insight=None):
        """
        Processes input and performs extraction.
        :param: prompt - str: input text;  insight_type - str: default key;
        :return: output - str: response from OpenAI's GPT-3 davinci-001 model-completion service.
        """
        
        # TO BE SENT DURING API REQUEST
        prompt_mapping = INSIGHT_OPTIONS
        
        if insight is None and (prompt is None or str(prompt).strip()=="" or str(prompt)=="nan"):
            return Exception("ERROR: Incorrect or no prompt passed!")
        
        if insight and insight.lower().strip() in prompt_mapping.keys(): 
            insight = insight.lower().strip()
            input_text = self.doc + prompt_mapping[insight]
            result = self.run_on_doc(input_text).strip("\n").strip()

        else:
            # USER INPUT (FAQ)
            user_input = str(prompt).rstrip(" ").rstrip(".").rstrip(":").rstrip("-").rstrip(",").rstrip("?").rstrip("!")
            input_text = self.doc + user_input + " in the TEXT."
            result = self.run_on_doc(input_text).strip("\n").strip()
        
        print(input_text.split("###")[2], ":\n\n", result, "\n", "--"*50)
        return result


def checkVersion():
    status = {
        "openai_version": "gpt-3",
        "openai_model": "da-vinci-001",
        "read_api_version": "v3.2",
    }
    return status


def main(file_path, query, default_insight=None):
    
    # 1> load json config
    load_config("azure_config.json")
    
    # 2> run OCR
    execute = run_ocr_process(file_path)
    ocr_output = execute.main()

    # 3> extract meta-data
    run_docAI = extract_metadata(openai, ocr_output)
    if default_insight:
        output = run_docAI.execute("", default_insight)
    else:
        output = run_docAI.execute(query)
    
    # print("\n\nOUT:\n\n", output)
    return output

# EOF