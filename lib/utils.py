import spacy
from standoffconverter import Standoff, View
from lxml import etree
from pathlib import Path
from tqdm import tqdm
from spacy.tokens import Doc, Span

nlp_model_fr = spacy.load("fr_core_news_lg")
nlp_model_fr.remove_pipe('ner')

class OffsetResolver(object):
    
    def __init__(self, view: View, standoff_obj: Standoff) -> None:
        self.tei_view = view
        self.tei_so = standoff_obj
        self.standoffs = self.tei_so.standoffs

    def get_tag_from_char_index(self, start_index: int, end_index) -> str:
        token_length = end_index - start_index
        # get the offset-based of the token in the source TEI document
        so_position = self.tei_view.table.iloc[self.tei_view.get_table_index(start_index)]['position']
        end_position = so_position + token_length
        # comment out line below to enable debugging
        # print(f"Start: {so_position}, End: {end_position}")
        matches = [st for st in self.standoffs if st['begin'] >= so_position and st['end'] <= end_position]
        if matches:
            return matches[0]['el'].tag.split("}")[-1]
        else:
            return None

def tei_element_to_ner_label(tag: str) -> str:
    """
    Convert a TEI element tag to a Named Entity Recognition (NER) label.

    Args:
        tag (str): The TEI element tag to be converted.

    Returns:
        str: The corresponding NER label. Returns "PER" for "persName", 
             "LOC" for "placeName", and None for any other tag.
    """
    if tag == "persName":
        return "PER"
    elif tag == "placeName":
        return "LOC"
    else:
        return None

def tei2spacy(tei_file_path: Path) -> Doc:
    """
    Convert a TEI (Text Encoding Initiative) XML file to a SpaCy Doc object with named entities (pre-annotated in the TEI).
    Args:
        tei_file_path (Path): The file path to the TEI XML file.
    Returns:
        Doc: A SpaCy Doc object containing the text and named entities from the TEI file.
    The function performs the following steps:
    1. Reads the TEI document using standoffconverter to create an XML tree and a standoff object.
    2. Creates a view of the TEI document, excluding elements outside the specified tag and inserting tag text.
    3. Converts the view to a SpaCy Doc object.
    4. Initializes an OffsetResolver object to map character offsets between the TEI and SpaCy documents.
    5. Iterates over the tokens in the SpaCy document, projecting entities from the TEI document onto the tokens.
    6. Converts the identified entities to SpaCy format and injects them into the Doc object.
    Note:
        The function assumes that `nlp_model_fr`, `Standoff`, `View`, `OffsetResolver`, `tei_element_to_ner_label`, 
        and `Span` are defined elsewhere in the codebase.
    """

    # read the TEI document with standoffconverter (view and standoff)
    print(f"Processing file: {tei_file_path}")
    xml_tree = etree.parse(tei_file_path)
    tei_so = Standoff(xml_tree, namespaces={"tei":"http://www.tei-c.org/ns/1.0"})
    tei_view = (View(tei_so)
        .exclude_outside("{http://www.tei-c.org/ns/1.0}reg")
        .insert_tag_text("{http://www.tei-c.org/ns/1.0}reg", " ")
        .shrink_whitespace()
    )
    
    # create a spacy doc object
    doc = nlp_model_fr(tei_view.get_plain())
    doc.user_data['path'] = str(tei_file_path) 
    doc.user_data['filename'] = str(tei_file_path.name)
    print(f"There are {len(doc)} tokens in document {doc.user_data['filename']}")

    # initialise the OffsetResolver object
    resolver = OffsetResolver(tei_view, tei_so)

    entities = []
    entity = {
        'label': None,
        'chunks':[]
    }
    inside_entity = False

    # Iterate over the tokens in the document and project the entities from the TEI document
    # onto character offsets of tokens in the SpaCy document
    for token in tqdm(doc, desc="Projecting NER labels from TEI onto SpaCy tokens"):
        tei_tag = resolver.get_tag_from_char_index(token.idx, token.idx + len(token.text))
        ner_label = tei_element_to_ner_label(tei_tag)
        
        if inside_entity:
            if ner_label is None:
                entities.append(entity)
                entity = {}
                inside_entity = False
            else:
                if entity['label'] == ner_label:
                    entity['chunks'].append(token)
                else:
                    entities.append(entity)
                    entity = {
                        'label': ner_label,
                        'chunks': [token]
                    } 
        else:
            if ner_label is not None:
                entity['label'] = ner_label
                entity['chunks'] = [token]
                inside_entity = True
    
    ## Convert the entities to Spacy format
    entities_to_add = []
    for entity in entities:
        spacy_ent = {}
        spacy_ent['start'] = entity['chunks'][0].i
        spacy_ent['end'] = entity['chunks'][-1].i + 1
        spacy_ent['label'] = entity['label']
        entities_to_add.append(spacy_ent)

    # Create Span objects for each entity and inject them into the Doc object
    doc.ents = [Span(doc, ent["start"], ent["end"], label=ent["label"]) for ent in entities_to_add]
    print(f"Found {len(list(doc.ents))} entities in document {doc.user_data['filename']}")
    print("##############################################")
    return doc