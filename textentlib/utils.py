import dataclasses
import spacy
import random
import operator
import pandas as pd
from dataclasses import dataclass
from standoffconverter import Standoff, View
import xml.etree.ElementTree as ET
from lxml import etree
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup as bs
from spacy.tokens import Doc, Span, DocBin
from spacy.language import Language
from typing import List, Tuple

nlp_model_fr = spacy.load("fr_core_news_lg")
nlp_model_fr.remove_pipe('ner')
nlp_model_fr.add_pipe(
    "entityfishing", config={
        "api_ef_base": "http://nerd.huma-num.fr/nerd/service"
    }
)

def sample_files(folder: Path, n: int, files_to_exclude: List[Path] = None) -> List[Path]:
    """
    Randomly sample a specified number of files from a given folder.

    Args:
        folder (Path): The path to the folder from which to sample files.
        n (int): The number of files to sample.
        List[Path]: A list of files to exclude from the sampling process.

    Returns:
        List[Path]: A list of Paths to the sampled files.
    """
    all_files = set(folder.iterdir())
    print(f'Found {len(all_files)} files in {folder}')
    if n:
        if files_to_exclude:
            exclude = set(files_to_exclude)
            filtered = all_files - exclude
            print(f'Excluded {len(exclude)} files: kept {len(filtered)} files')
            sample = random.sample(list(filtered), n)
        else:
            sample = random.sample(list(all_files), n)
        return sample
    else:
        return list(all_files)

def print_corpus_summary(corpus: DocBin, spacy_model: Language):
    """
    Print a summary of spaCy corpus (`DocBin`) by printing the number of documents, entities, and tokens.

    Args:
        corpus (DocBin): The corpus to be summarized, represented as a spaCy DocBin object.
        spacy_model (spacy.language.Language): The spaCy language model used to process the corpus.

    Returns:
        None
    """
    docs = list(corpus.get_docs(spacy_model.vocab))
    print(f"Number of documents in the corpus: {len(docs)}")
    print(f"Number of entities in the corpus: {sum([len(list(doc.ents)) for doc in docs])}")
    print(f"Number of tokens in the corpus: {sum([len(doc) for doc in docs])}")

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

def get_tag_from_char_index(char_start: int, char_end: int, entities: dict) -> str:
    for (start, end), tag in entities.items():
        if start <= char_start and end >= char_end:
            return tag
    return None

def extract_metadata_from_tei(root: ET) -> dict:
    """
    Extracts metadata from a TEI (Text Encoding Initiative) XML element.

    Args:
        root (ET.Element): The root element of the TEI XML document.

    Returns:
        dict: A dictionary containing the extracted metadata with keys 'author', 'title', and 'date'.
              The values are strings or None if the corresponding metadata is not found.
    """
    ns = {"tei":"http://www.tei-c.org/ns/1.0"}
    bibl = root.find('.//tei:bibl', namespaces=ns)
    author = None
    title = None
    date = None

    if bibl is not None:
        author_elem = bibl.find('.//tei:author/tei:persName', namespaces=ns)
        title_elem = bibl.find('.//tei:title', namespaces=ns)
        date_elem = bibl.find('.//tei:date', namespaces=ns)
        ptr_elem = bibl.find('.//tei:ptr', namespaces=ns)

    if author_elem is not None:
        author = author_elem.text
    if title_elem is not None:
        title = title_elem.text
    if date_elem is not None:
        date = date_elem.attrib['when'] if date_elem is not None and 'when' in date_elem.attrib else None
    if ptr_elem is not None:
        link = ptr_elem.attrib['target'] if ptr_elem is not None and 'target' in ptr_elem.attrib else None

    return {
        'author': author,
        'title': title,
        'date': date,
        'link': link
    }

def tei2spacy(tei_file_path: Path, project_entities: bool, disable_progress_bar: bool) -> Doc:
    """
    Convert a TEI (Text Encoding Initiative) XML file to a SpaCy Doc object with named entities (pre-annotated in the TEI).
    Args:
        tei_file_path (Path): The file path to the TEI XML file.
        disable_progress_bar (bool): A flag to disable the progress bar when projecting entities.
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

    # extract metadata from TEI
    metadata = extract_metadata_from_tei(xml_tree)
    
    # create a spacy doc object
    doc = nlp_model_fr(tei_view.get_plain())
    doc.user_data['author'] = metadata['author']
    doc.user_data['title'] = metadata['title']
    doc.user_data['publication_date'] = metadata['date']
    doc.user_data['path'] = str(tei_file_path) 
    doc.user_data['filename'] = str(tei_file_path.name)
    doc.user_data['entity_linking'] = None
    print(f"There are {len(doc)} tokens in document {doc.user_data['filename']}")

    # initialise the OffsetResolver object
    resolver = OffsetResolver(tei_view, tei_so)

    entities = []
    entity = {
        'label': None,
        'chunks':[]
    }
    inside_entity = False

    if project_entities:
        # Iterate over the tokens in the document and project the entities from the TEI document
        # onto character offsets of tokens in the SpaCy document
        for token in tqdm(doc, desc="Projecting NER labels from TEI onto SpaCy tokens", disable=disable_progress_bar):
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
    else:
        print("Skipping entity projection")
        return doc

def tei2spacy_simple(tei_file_path: Path) -> Doc:
    soup = bs(tei_file_path.read_text(), 'xml')
    output_text = ""
    chunks = {}
    entities = []
    entity = {}
    inside_entity = False

    for elem in soup.findAll('reg'):
        for node in elem.contents:
            if isinstance(node, str):
                output_text += node.text
            else:
                if node.name == 'persName' or node.name == 'placeName':
                    offset_start = len(output_text)
                    output_text += node.text
                    offset_end = len(output_text)
                    ner_tag = tei_element_to_ner_label(node.name)
                    chunks[(offset_start, offset_end)] = ner_tag
                else:
                    output_text += node.text
        output_text += " "

    # extract metadata from TEI
    metadata = extract_metadata_from_tei(etree.parse(tei_file_path))
    
    # create a spacy doc object
    doc = nlp_model_fr(output_text)
    doc.user_data['author'] = metadata['author']
    doc.user_data['title'] = metadata['title']
    doc.user_data['publication_date'] = metadata['date']
    doc.user_data['link'] = metadata['link']
    doc.user_data['path'] = str(tei_file_path) 
    doc.user_data['filename'] = str(tei_file_path.name)
    doc.user_data['document_id'] = doc.user_data['filename'].split('.')[0]
    doc.user_data['entity_linking'] = None


    # Iterate over the tokens in the document and project the entities from the TEI document
    # onto character offsets of tokens in the SpaCy document
    for token in doc:
        ner_label = get_tag_from_char_index(token.idx, token.idx + len(token.text), chunks)
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

    # Convert the entities to Spacy format
    # NB: start and end are token indices, not character offsets
    entities_to_add = []
    for entity in entities:
        spacy_ent = {}
        spacy_ent['start'] = entity['chunks'][0].i
        spacy_ent['end'] = entity['chunks'][-1].i + 1
        spacy_ent['label'] = entity['label']
        entities_to_add.append(spacy_ent)

    # Create Span objects for each entity and inject them into the Doc object
    doc.ents = [Span(doc, ent["start"], ent["end"], label=ent["label"]) for ent in entities_to_add]
    return doc

def load_or_create_corpus(spacy_corpus_path: str) -> DocBin:
    """
    Load an existing spaCy corpus from disk or create a new one if it doesn't exist.

    Args:
        spacy_corpus_path (str): The file path to the spaCy corpus.

    Returns:
        DocBin: The loaded or newly created spaCy DocBin object.

    If the specified file path exists, the function loads the serialized spaCy corpus from the disk
    and prints a summary of the corpus. If the file path does not exist, it creates a new empty
    DocBin object.
    """
    if Path(spacy_corpus_path).exists():
        spacy_corpus = DocBin(store_user_data=True).from_disk(spacy_corpus_path)
        print(f"Loaded serialize spacy corpus from {spacy_corpus_path}")
        print_corpus_summary(spacy_corpus, nlp_model_fr)
    else:
        spacy_corpus = DocBin(store_user_data=True)
    return spacy_corpus

@dataclass
class Entity:
    qid: str
    ner_labels: List[str] # ner tags for entity mentions
    mention_frequency: int # document-level mention frequency
    unique_surface_forms: List[str]
    short_desc: str
    
    def to_json(self, include_null=False) -> dict:
        """Converts this to json. Assumes variables are snake cased, converts to camel case.

        Args:
            include_null (bool, optional): Whether null values are included. Defaults to False.

        Returns:
            dict: Json dictionary
        """
        return dataclasses.asdict(
            self,
            dict_factory=lambda fields: {
                key: value
                for (key, value) in fields
                if value is not None or include_null
            },
        )