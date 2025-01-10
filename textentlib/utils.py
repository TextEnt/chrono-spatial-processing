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
from spacy.tokens import Doc, Span, DocBin
from typing import List, Tuple

nlp_model_fr = spacy.load("fr_core_news_lg")
nlp_model_fr.remove_pipe('ner')

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

def print_corpus_summary(corpus: DocBin, spacy_model: spacy.language.Language):
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

    if author_elem is not None:
        author = author_elem.text
    if title_elem is not None:
        title = title_elem.text
    if date_elem is not None:
        date = date_elem.attrib['when'] if date_elem is not None and 'when' in date_elem.attrib else None

    return {
        'author': author,
        'title': title,
        'date': date
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

@dataclass
class Entity:
    qid: str
    ner_labels: List[str] # ner tags for entity mentions
    mention_frequency: int # document-level mention frequency
    unique_surface_forms: List[str]
    short_desc: str

class SalientSentenceSelector(object):
    
    def __init__(self, spacy_doc: Doc):
        self.doc = spacy_doc
        self.person_entities = self._mentions2entities(ner_label='PER')
        self.place_entities = self._mentions2entities(ner_label='LOC')
        self.sentences = {sent_i + 1: sent for sent_i, sent in enumerate(self.doc.sents)}
        self.sent2ent_idx = self._build_sentence2entity_index()

    def _mentions2entities(self, ner_label : str = 'PER') -> List[str]:
        # transform the entity mentions from spacy into a dataframe for easier manipulation
        self._mentions_df = pd.DataFrame(
            [
                {
                    'mention': ent.text,
                    'ner_label': ent.label_,
                    'qid': ent._.kb_qid,
                    'url_wikidata': ent._.url_wikidata,
                    'nerd_score': ent._.nerd_score
                }
                for ent in self.doc.ents
                if ent.label_ == ner_label
            ]
        )
        linked_entities_df = self._mentions_df[self._mentions_df.qid.notna()]
        n_nonlinked_entities = len(self._mentions_df[self._mentions_df.qid.isna()])
        n_linked_entities = len(linked_entities_df)
        print(
            f'Document {self.doc.user_data["filename"]} contains {self._mentions_df.shape[0]} {ner_label} entities;',
            f'{n_linked_entities} linked and {n_nonlinked_entities} non-linked'
        )

        # unique entities
        unique_qids = linked_entities_df.qid.unique()
        print(f'Document {self.doc.user_data["filename"]} contains {len(unique_qids)} {ner_label} unique entities')

        entities = []
        for qid in unique_qids:
            mentions  = linked_entities_df[linked_entities_df.qid == qid].mention
            mention_frequency = len(mentions.tolist())
            ner_labels = linked_entities_df[linked_entities_df.qid == qid].ner_label.unique().tolist()
            unique_surface_forms = mentions.unique().tolist()
            entities.append(
                Entity(
                    qid=qid,
                    ner_labels=ner_labels,
                    mention_frequency=mention_frequency,
                    unique_surface_forms=unique_surface_forms,
                    short_desc=''
                )
            )
        return {entity.qid: entity for entity in entities}

    def _build_sentence2entity_index(self) -> dict:
        sentence2entity_index = {}
        for sent_i, sent in self.sentences.items():
            for ent in sent.ents:
                if ent._.kb_qid:
                    if sent_i not in sentence2entity_index:
                        sentence2entity_index[sent_i] = set()
                    sentence2entity_index[sent_i].add(ent._.kb_qid)
        return sentence2entity_index

    # TODO: select sentences for people and places separately
    def _find_sentences_for_entity(self, entity: Entity) -> List[str]:
        sentences = []
        for sentence_id, entity_qids in self.sent2ent_idx.items():
            if entity.qid in entity_qids:
                sentences.append(self.sentences[sentence_id])
        sentences.sort(key=lambda x: len(x), reverse=True)
        return sentences

    # take the most frequent person|place and return the first `k`` sentences where the entity appears,
    # ranked by sentence length (rationale: the longer, the more informative)
    def select(self, entity_type : str  = 'person', top_k_sentences: int = 5) -> Tuple[Entity, List[str]]:
        if entity_type == 'person':
            sorted_person_entities = sorted(self.person_entities.values(), key=operator.attrgetter('mention_frequency'), reverse=True)
            top_person = sorted_person_entities[0]
            return (top_person, self._find_sentences_for_entity(top_person)[:top_k_sentences])
        elif entity_type == 'place':
            sorted_place_entities = sorted(self.place_entities.values(), key=operator.attrgetter('mention_frequency'), reverse=True)
            top_place = sorted_place_entities[0]
            return (top_place, self._find_sentences_for_entity(top_place)[:top_k_sentences])
        else:
            raise ValueError(f"Entity type {entity_type} not supported")