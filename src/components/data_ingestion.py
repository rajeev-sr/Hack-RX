import os
from unstructured.partition.pdf import partition_pdf
def processDocument():
    
    
    directory_path = "/Users/rahulraj/Hack-RX/datafile"

    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path):
    
            # Initialize the loader with config and desired transcript format
            loader = partition_pdf(
                file_path=full_path,
                
                
            )
            # Load the transcription document(s)
            docs4 = loader.load()

            audio_text.append(docs4[0].page_content)
            # print(docs4[0].page_content)
    return audio_text