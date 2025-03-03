import os
import operator, json
from pathlib import Path
from typing import List, Tuple
from spacy.tokens import Doc
import pandas as pd
from textentlib.utils import Entity

# the summary should contain:
# document metadata: author, title, publication date
# top 5 person mentions
# top 5 place mentions
# most salient person entity + top 5 sentences
# most salient place entity + top 5 sentences

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

def build_JSON_document_summary(spacy_doc):
    sass = SalientSentenceSelector(spacy_doc)

    # extract the top 5 sentences for the most frequent geographical place in the document
    # this entity context made of the k sentences can then be fed to an LLM
    top_place, place_rel_sentences = sass.select(top_k_sentences=5, entity_type='place')

    # extract the top 5 sentences for the most frequent person in the document
    # this entity context made of the k sentences can then be fed to an LLM
    top_person, person_rel_sentences = sass.select(top_k_sentences=5, entity_type='person')

    top_5_persons = sorted(
        list(sass.person_entities.values()),
        key=operator.attrgetter('mention_frequency'),
        reverse=True
    )[:5]

    top_5_places = sorted(
        list(sass.place_entities.values()),
        key=operator.attrgetter('mention_frequency'),
        reverse=True
    )[:5]

    summary = {
        "metadata": {
            "author": spacy_doc.user_data['author'],
            "title": spacy_doc.user_data['title'],
            "publication_date": spacy_doc.user_data['publication_date'],
            "document_id": spacy_doc.user_data['filename'].split('.')[0]
        },
        "context": {
            "people": {
                "top_1_person": {
                    "entity": {
                        "label": top_person.unique_surface_forms[0],
                        "frequency": top_place.mention_frequency
                    },
                    "related_sentences": [str(sent) for sent in person_rel_sentences]
                },
                "top_5_persons": [e.unique_surface_forms[0] for e in top_5_persons]
            },
            "places":{
                "top_1_place": {
                    "entity": {
                        "label": top_place.unique_surface_forms[0],
                        "frequency": top_place.mention_frequency
                    },
                    "related_sentences": [str(sent) for sent in place_rel_sentences]
                },
                "top_5_places": [e.unique_surface_forms[0] for e in top_5_places]
            }
        }
    }
    return summary
    
def generate_document_summaries(spacy_docs: List[Doc], output_path: Path) -> None:
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Iterate over documents in spacy_docs
    for i, spacy_doc in enumerate(spacy_docs):
        try:
            # Build JSON summary
            doc_summary = build_JSON_document_summary(spacy_doc)
            
            # Define the file path
            file_name = f"{doc_summary['metadata']['document_id']}_summary.json"
            file_path = output_path / file_name
            
            # Write the JSON summary to a file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(doc_summary, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error processing document {i}: {e}")
