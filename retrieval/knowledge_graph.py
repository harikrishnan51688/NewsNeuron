from neo4j import GraphDatabase
from core.config import settings
from typing import List, Tuple
import spacy

nlp = spacy.load("en_core_web_lg")

class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            settings.NEO4J_URI,
            auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
        )

    def close(self):
        self.driver.close()

    def extract_entities(self, text: str) -> List[str]:
        doc = nlp(text)
        return list(set([ent.text for ent in doc.ents]))

    def extract_svo_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Extract Subject–Verb–Object triples using spaCy dependency parser.
        Returns list of (subject, verb, object).
        """
        doc = nlp(text)
        triples = []
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":  # main verb
                subjects = [w.text for w in token.lefts if w.dep_ in ("nsubj", "nsubjpass")]
                objects = [w.text for w in token.rights if w.dep_ in ("dobj", "pobj", "attr")]
                for subj in subjects:
                    for obj in objects:
                        triples.append((subj, token.lemma_, obj))  # store lemma of verb
        return triples

    def create_entity_node(self, tx, entity: str, label: str = "Entity"):
        """
        Create an entity node with an optional label (Person, Location, etc.)
        """
        query = f"MERGE (e:{label} {{name: $name}})"
        tx.run(query, name=entity)

    def create_relationship(self, tx, source: str, target: str, relation: str):
        """
        Create relationship with verb as type.
        Example: (ElonMusk)-[:VISIT]->(Paris)
        """
        rel_type = relation.upper().replace(" ", "_")  # Neo4j rel types must be uppercase, no spaces
        query = f"""
        MATCH (a {{name: $source}}), (b {{name: $target}})
        MERGE (a)-[r:{rel_type}]->(b)
        """
        tx.run(query, source=source, target=target)

    def add_article_to_graph(self, text: str):
        doc = nlp(text)

        # Step 1: Add entity nodes with proper labels
        with self.driver.session() as session:
            for ent in doc.ents:
                session.execute_write(self.create_entity_node, ent.text, ent.label_)

        # Step 2: Extract and add SVO triples as relationships
        triples = self.extract_svo_triples(text)
        with self.driver.session() as session:
            for subj, verb, obj in triples:
                session.execute_write(self.create_entity_node, subj)   # ensure nodes exist
                session.execute_write(self.create_entity_node, obj)
                session.execute_write(self.create_relationship, subj, obj, verb)
