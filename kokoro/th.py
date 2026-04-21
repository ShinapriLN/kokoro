from transformers import pipeline



class THG2P:
    def __init__(self):
        self._generator = pipeline("text2text-generation", model = "pythainlp/thaig2p-v2.0")

    def __call__(self, grapheme: str):
        phoneme = self._generator(grapheme)
        phoneme = phoneme[0]['generated_text'].replace(" ", "").replace(".", " ")
        return phoneme, None
    
