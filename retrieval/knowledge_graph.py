from neo4j import GraphDatabase
from core.config import settings
from typing import List, Tuple, Dict, Any
import spacy
import re

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

    def run_query(self, cypher: str, params: dict = None):
            with self.driver.session() as session:
                return session.run(cypher, params or {}).data()


    def query_relationships_by_entities(self, entities: List[str]) -> Dict[str, Any]:
        """
        Query relationships involving the specified entities.
        """
        try:
            if not entities:
                return {"error": "No entities provided for querying"}

            # Create a flexible query that finds relationships involving any of the entities
            entity_conditions = " OR ".join([f"n.name CONTAINS '{entity}' OR m.name CONTAINS '{entity}'" for entity in entities])
            
            cypher_query = f"""
            MATCH (n)-[r]->(m)
            WHERE {entity_conditions}
            RETURN n.name as source, type(r) as relationship, m.name as target, 
                   labels(n) as source_labels, labels(m) as target_labels
            LIMIT 50
            """
            
            results = self.run_query(cypher_query)
            
            if not results:
                # Try a more flexible search with partial matching
                cypher_query = """
                MATCH (n)-[r]->(m)
                WHERE ANY(entity IN $entities WHERE 
                    toLower(n.name) CONTAINS toLower(entity) OR 
                    toLower(m.name) CONTAINS toLower(entity))
                RETURN n.name as source, type(r) as relationship, m.name as target,
                       labels(n) as source_labels, labels(m) as target_labels
                LIMIT 50
                """
                results = self.run_query(cypher_query, {"entities": entities})
            
            return {
                "success": True,
                "entities_searched": entities,
                "relationships_found": len(results),
                "relationships": results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error querying knowledge graph: {str(e)}"
            }

    def analyze_question_and_query(self, question: str) -> Dict[str, Any]:
        """
        Analyze a natural language question and query the knowledge graph accordingly.
        """
        try:
            # Extract entities from the question
            entities = self.extract_entities(question.lower())
            
            # Determine query type based on question patterns
            question_lower = question.lower()
            
            if any(word in question_lower for word in ["who", "what", "which"]):
                # Entity-focused questions
                if entities:
                    primary_entity = entities[0]
                    result = self.query_entity_information(primary_entity)
                    result["query_type"] = "entity_information"
                    result["original_question"] = question
                    return result
                else:
                    return {"error": "Could not identify entities in the question"}
                    
            elif any(word in question_lower for word in ["relationship", "related", "connection", "connect"]):
                # Relationship-focused questions
                result = self.query_relationships_by_entities(entities)
                result["query_type"] = "relationship_search"
                result["original_question"] = question
                return result
                
            elif any(word in question_lower for word in ["how many", "count", "number"]):
                # Count-based questions
                if entities:
                    cypher_query = """
                    MATCH (n)-[r]-(m)
                    WHERE ANY(entity IN $entities WHERE 
                        toLower(n.name) CONTAINS toLower(entity) OR 
                        toLower(m.name) CONTAINS toLower(entity))
                    RETURN count(r) as relationship_count
                    """
                    results = self.run_query(cypher_query, {"entities": entities})
                    return {
                        "success": True,
                        "query_type": "count",
                        "original_question": question,
                        "entities_searched": entities,
                        "count": results[0]["relationship_count"] if results else 0
                    }
                else:
                    return {"error": "Could not identify entities for counting"}
            
            else:
                # Default: search for relationships involving extracted entities
                result = self.query_relationships_by_entities(entities)
                result["query_type"] = "general_search"
                result["original_question"] = question
                return result
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Error analyzing question: {str(e)}",
                "original_question": question
            }