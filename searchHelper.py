#################################################################
#   @version: 1.0
#   @date: 26/02/2022
#################################################################

# imports
import json
from flask import Flask, jsonify
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
import logging
logger = logging.getLogger('ROOT')

# AZURE
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient


######################################################
# Setting up the search service connection
######################################################
def initiate_search():
    try:
        # get creds
        search_endpoint = os.environ.get("SEARCH_ENDPOINT")
        search_index_name = os.environ.get("INDEX_NAME")
        search_key = os.environ.get("SEARCH_KEY")

        if (search_key == "" or search_endpoint == "" or search_index_name == ""):
            raise Exception("ERROR: Unable to read search configuration file!")

        # get search setup
        search_client = SearchClient(search_endpoint, search_index_name, AzureKeyCredential(search_key))

        if (search_client is None):
            raise Exception("ERROR: Unable to setup the search service connection!")

    except Exception as e:
        logger.error(str(e))

    return search_client