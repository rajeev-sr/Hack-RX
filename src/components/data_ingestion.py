def audio_to_text():
    

    directory_path = "/home/rajeev-kumar/Desktop/Hack-RX/datafile"

    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        if os.path.isfile(full_path):
    
            # Initialize the loader with config and desired transcript format
            loader = AssemblyAIAudioTranscriptLoader(
                file_path=full_path,
                transcript_format=TranscriptFormat.TEXT,
                config=config,
                api_key=ass_api_key
            )
            # Load the transcription document(s)
            docs4 = loader.load()

            audio_text.append(docs4[0].page_content)
            # print(docs4[0].page_content)
    return audio_text