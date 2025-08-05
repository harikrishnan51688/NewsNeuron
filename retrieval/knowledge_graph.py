from neo4j import GraphDatabase
from core.config import settings
from typing import List
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
        entities = list(set([ent.text for ent in doc.ents]))
        return entities

    def create_entity_node(self, tx, entity: str):
        tx.run("MERGE (e:Entity {name: $name})", name=entity)

    def create_relationship(self, tx, source: str, target: str):
        tx.run("""
            MATCH (a:Entity {name: $source}), (b:Entity {name: $target})
            MERGE (a)-[:RELATED_TO]->(b)
        """, source=source, target=target)

    def add_article_to_graph(self, text: str):
        entities = self.extract_entities(text)
        with self.driver.session() as session:
            for entity in entities:
                session.execute_write(self.create_entity_node, entity)

            # naive way: connect all entities with RELATED_TO
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    session.execute_write(self.create_relationship, entities[i], entities[j])