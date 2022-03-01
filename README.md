
# Project Description

Quick guide to get started on analyzing documents using open source technologies

This project deals with open-source implementations of Microsoft Azure Cognitive services for extracting meaningful insights and keyword search service in a pool of documents.

- Documents are stored in a Azure container as blob items.
- Each document carries a unique Azure-Blob URL.
- Documents could be of following data types: pdf, png, jpg, jpeg, tiff, heic, txt, md.
- Components for this project are: 
        - Azure Search services
        - Azure Cognitive Computer Vision services
        - OpenAI API endpoints

Give a input query, search serivce locates the top ranked documents, which are then sent to OCR services for extraction and lastly OpenAI uses Completion services to extract insights.

----
## :: API DOCUMENTATION ::

#### Endpoint 1 :
> Make a get call to get search results.

    url = /search/<<QUERY>>
    r = requests.get(url)
    OUTPUT:
        [ 
            {'metadata':....}
        ]


#### Endpoint 2 :
> Make a get call to get the version info and check if the server is running...

    url = /docai/version
    r = requests.get(url)
    OUTPUT:
            {
              openai_model: da-vinci-001, 
              openai_version: gpt-3, 
              read_api_version: v3.2
            }

#### Endpoint 3 :
> Make a post call to run OCR on a storage-azure-blob document, passing it's Blob-URL.

    url = /docai/run_ocr
    payload = {
        file_path httpsdiscoveraiwork5625301287.blob.core.windows.netdocumentsThe%20Prospect%20of%20a%20Continued%20Correction.pdf
    }
    r = requests.post(url, json=payload)
    ocr_response_json = r.json()            # upload this in azure-blob as a JSON object. Need to pass this to next api!
    OUTPUT:
            {
              title: DOC_NAME, 
              lines: INFO,
               ...
              corpus: CORPUS
            }

#### Endpoint 4 : 
> Make a post call to run doc-ai on a document, need to send document's OCR-RESPONSE-JSON Object (saved in #2).
    
    # Option 1 ---- When user types a input query -----
            url = /docai/extract_insights
            payload = {
                document ocr_response_json,
                query Generate summary for this in 100 words!
            }
            r = requests.post(url, json=payload)
            OUTPUT:
                { 'output' : 'The summary is.....'}
    
    # Option 2 ----- When user clicks on a BUTTON  -----
            'default_insight': <<could be one of from the list>>
                                    default-summary,
                                    default-faq,
                                    default-keypoints,
                                    default-keywords,
                                    default-phrases,
                                    default-sentiment,
                                    default-abbrv
                                    ...
            url = /docai/extract_insights
            payload = {
                document ocr_response_json,
                query ,
                default_insight default-summary
            }
            r = requests.post(url, json=payload)
            OUTPUT:
                { 'output' : 'The summary is.....'}
