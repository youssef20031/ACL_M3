import spacy
nlp = spacy.load("en_core_web_sm")

class EntityExtractor:
    def __init__(self, document):
        self.document = document
        self.doc = nlp(document)
    
    def extract_entities(self):
        entities = []
        for ent in self.doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_),
                'start': ent.start_char,
                'end': ent.end_char
            })
        return entities