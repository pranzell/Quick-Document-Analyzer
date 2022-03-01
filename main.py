#################################################################
#   @version: 1.0
#   @date: 26/02/2022
#
#   QUICK DOCUMENT ANALYZER
#################################################################

# Imports
import sys
import os
import re
import json
import uuid
import re
import time
import requests
import random
import re
import ntpath
import json
from flask import Flask, jsonify, request, abort, Response
import datetime
import logging

# defined Classes
from searchHelper import *
from runDocAI import *


# Setting up a Logger
LOG_path = "logs/"
if not os.path.exists(LOG_path): os.mkdir(LOG_path)
logger_filename = LOG_path + "DocAIBot__{}.log".format(datetime.datetime.now().strftime("_%d-%m-%Y_%H.%M"))
logger = logging.getLogger('ROOT')
logger.setLevel(logging.INFO)
handler = logging.FileHandler(logger_filename)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s: %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Load configuration
def load_config(config_fp):
    with open(config_fp, "r") as f:
        config_data = json.load(fp=f)
        os.environ["STORAGE_NAME"] = config_data["STORAGE_NAME"]
        os.environ["STORAGE_CONN"] = config_data["STORAGE_CONN"]
        os.environ["OCR_KEY"] = config_data["OCR_KEY"]
        os.environ["OCR_ENDPOINT"] = config_data["OCR_ENDPOINT"]
        os.environ["OPENAI_KEY"] = config_data["OPENAI_KEY"]
        os.environ["SEARCH_ENDPOINT"] = config_data["SEARCH_ENDPOINT"]
        os.environ["INDEX_NAME"] = config_data["INDEX_NAME"]
        os.environ["SEARCH_KEY"] = config_data["SEARCH_KEY"]
    return


# server
app = Flask(__name__)

###################################################
# SEARCH ENDPOINT
###################################################
@app.route('/docai/<string:query>', methods=["GET", "POST"])
def search(query):
    """
    Searches the blob storage for documents matching user query
    Output: Returns ranked results with content type, name of document, storage path, and search score.
    """
    final_result = list()
    try:
        search_client = initiate_search()
    except Exception as e:
        error_code = "SEARCH CONNECTION ERROR | Trace: \n{}".format(str(e))
        search_logger.error(error_code)
        return jsonify({"Search Failed:": error_code})

    if(search_client is not None):
        search_logger.info("Connection with search service established!")
    
    try:
        # select = ("metadata_content_type", "metadata_storage_name", "metadata_storage_path")
        # final_results = list(search_client.search(search_text=query, query_type="semantic", query_language ="en-us", select=",".join(select)))
        results = search_client.search(search_text=query)
        for result in results:
           doc = dict()
           doc["metadata_content_type"] = result["metadata_content_type"]
           doc["metadata_storage_name"] = result["metadata_storage_name"]
           doc["metadata_storage_path"] = result["metadata_storage_path"]
           doc["@search.score"] = result["@search.score"] 
           final_result.append(doc)
        search_logger.info("OUTPUT :: {}".format(json.dumps(final_result, indent=4)))
    except Exception as e:
        error_code = "SEARCH EXECUTION ERROR | Trace: \n{}".format(str(e))
        search_logger.error(error_code)
        return jsonify({"Search Error" : error_code})
    
    return jsonify(final_result)


###################################################
# ANALYZER ENDPOINTS
###################################################
@app.route('/docai/version', methods=['GET'])
def version_check():
    """
    To check status of the bot.
    Returns: version information.
    """
    output = checkVersion()
    return jsonify({"output": output})


@app.route('/docai/run_ocr', methods=['POST'])
def run_ocr_engine():
    """
    Executes Azure's OCR Engine (ReadAPI v3.2) on a given azure-blob URL.
        Headers : { "Content-Type": "application/json" }
        Payload : { "file_path": <<azure-storage-blob-filepath>> }
        Auth    : NA
    Returns: OCR'ed document in JSON format, which should be stored in a azure-blob.
    """
    status=False
    try:
        # INPUT
        if request.method=="POST":
            input_json = request.json
            file_path = input_json.get("file_path")

            # RUN OCR SERVICE
            if file_path:
                execute = run_ocr_process(file_path)
                ocr_output = execute.main()
                status=True
            else:
                raise Exception("ERROR: Variable 'file_path' not sent in the payload!")
        else:
            raise Exception("ERROR: Incorrect API method used - please use a 'post' request")
    
    except Exception as e:
        error_code = "OCR EXECUTION ERROR | Trace: \n{}".format(str(e))
        logger.error(error_code)

    if status:
        logger.info("OUTPUT :: {}".format(json.dumps(ocr_output, indent=4)))
        return jsonify({"output": ocr_output})
    else:
        return jsonify({"error": error_code})


@app.route('/docai/extract_insights', methods=['POST'])
def extract_info():
    """
    Extracts meta-data from given blob document path.
        Headers : { "Content-Type": "application/json" }
        
        Payload : { "document": << OCR_OUTPUT_JSON >>,
                    "query": << USER-TYPED_STRING_PROMPT --OR-- "" IF default_insight IS PASSED >>,
                    "default_insight": << NULL IF PROMPT IS PASSED --OR-- A SELECTED VALUE >>
                  }
        Auth    : NA
    Returns: Insights from OCR'ed document.
    """
    status=False
    try:
        # INPUT
        if request.method=="POST":
            # EXTRACT USING OPEN AI
            input_json = request.json
            if input_json:
                ocr_output = input_json.get("document")
                prompt = input_json.get("query")
                default_insight = input_json.get("default_insight")
                
                if ocr_output:
                    if (prompt is None or str(prompt).strip()=="" or str(prompt)=="nan" or str(prompt)=="null") and default_insight is None:
                        raise Exception("ERROR: User input or default-insight either is missing - please pass atleast one!")
                    else:
                        # RUN META-DATA SERVICE
                        run_docAI = extract_metadata(openai, ocr_output)
                        if default_insight:
                            
                            if default_insight in INSIGHT_OPTIONS.keys():
                                output = run_docAI.execute("", default_insight)
                            else:
                                raise Exception("ERROR: Chosen 'default-insight' option is unavailable!")
                        
                        else:
                            output = run_docAI.execute(prompt)
                        status=True
                        # print("\n\nOUT:\n\n", output)
                else:
                    raise Exception("ERROR: OCR document json is missing in payload - please pass using 'document' in body!")
            else:
                raise Exception("ERROR: Payload is missing!")
        else:
             raise Exception("ERROR: Incorrect API method used - please use a 'post' request")
    
    except Exception as e:
        error_code = "OPEN AI EXECUTION ERROR | Trace: \n{}".format(str(e))
        logger.error(error_code)

    if status:
        logger.info("EXTRACT OUTPUT :: {}".format(json.dumps(output, indent=4)))
        return jsonify({"output": output})
    else:
        return jsonify({"error": error_code})


@app.route('/docai/run', methods=['POST'])
def run_docai():
    """
    Executes Azure's OCR Engine and then OpenAI on a given azure-blob URL.
        Headers : { "Content-Type"    : "application/json" }
        Payload : { "file_path"       : <<azure-storage-blob-filepath>>,
                    "query"           : << USER-TYPED_STRING_PROMPT --OR-- "" IF default_insight IS PASSED >>,
                    "default_insight" : << NULL IF PROMPT IS PASSED --OR-- A SELECTED VALUE >>
                    }
        Auth    : NA
    Returns: Insights from OCR'ed document in JSON format.
    """
    
    status=False
    try:
        # INPUT
        if request.method=="POST":
            
            input_json = request.json
            file_path = input_json.get("file_path")
            prompt = input_json.get("query")
            default_insight = input_json.get("default_insight")

            if file_path:
            
                # RUN OCR SERVICE
                execute = run_ocr_process(file_path)
                ocr_output = execute.main()
                
                # RUN META-DATA SERVICE
                run_docAI = extract_metadata(openai, ocr_output)
                if default_insight:

                    if default_insight in INSIGHT_OPTIONS.keys():
                        output = run_docAI.execute("", default_insight)
                    else:
                        raise Exception("ERROR: Chosen 'default-insight' option is unavailable!")

                else:
                    output = run_docAI.execute(prompt)
                
                # print("\n\nOUT:\n\n", output)
                status=True

            else:
                raise Exception("ERROR: Variable 'file_path' not sent in the payload!")
        else:
            raise Exception("ERROR: Incorrect API method used - please use a 'post' request")
    
    except Exception as e:
        error_code = "DOC AI EXECUTION ERROR | Trace: \n{}".format(str(e))
        logger.error(error_code)

    if status:
        logger.info("FINAL OUTPUT :: {}".format(json.dumps(output, indent=4)))
        return jsonify({"output" : output})
    else:
        return jsonify({"error" : error_code})


# START
if __name__ == '__main__':
    print("Starting app now...\n\n")
    load_config("azure_config.json")
    app.run(debug=False)

# EOF