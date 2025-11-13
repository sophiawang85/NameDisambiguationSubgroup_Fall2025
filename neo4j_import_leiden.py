"""
Neo4j import script (step 2 of 2)

Purpose
- Read the cached JSON produced by `neo4j_data.py`
- Create publication nodes and similarity edges in Neo4j for disambiguation experiments

Current capabilities
- PUBLICATION nodes with properties: id, title, year, authors (JSON string), venue
- COAUTHOR edges: connect publications that share at least one author
- COVENUE edges: connect publications that share the same venue

Usage
1) Ensure a Neo4j instance is running and accessible
2) Set connection variables in the main block (URI, USER, PASSWORD, DB, PATH)
3) Run from repo root, e.g. `PYTHONPATH=. python neo4j_import.py`

Outputs (example counts from "David Nathan")
- 413 nodes
- 22,953 relationships total
- 5,113 COVENUE relationships
- 17,840 COAUTHOR relationships

Notes
- Relationships are currently modeled as directional in code, but clustering can treat the graph as undirected
- Consider converting to undirected by creating a single relationship with `MERGE` or by normalizing during analysis
"""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, Neo4jError
import json
from typing import List, Dict, Tuple

# For CoTitle similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as _e:  # pragma: no cover - optional at import time
    TfidfVectorizer = None
    cosine_similarity = None

class Neo4jImportData:

    def __init__(self, uri, user, password, db, data_path):
        """
        Initialize a Neo4j driver and load the cached data

        Args
            uri (str): Neo4j URI (e.g. bolt://localhost:7687)
            user (str): instance username
            password (str): instance password
            db (str): database name
            data_path (str): Path to JSON created by neo4j_data.py (cache/<Author>_data.json)
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user,password))
            self.driver.verify_connectivity()
            print("Connection to Neo4j database successful!")
        except ServiceUnavailable as e:
            print(f"Connection failed: {e}")

        self.db = db

        with open(data_path,'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def close(self):
        self.driver.close()

    
    def publication_as_nodes(self):
        """
        Create PUBLICATION nodes with properties id, title, year, authors (JSON string), venue
        """
        for work in self.data['works_data']:
            pub_id, title, year, authors, venue = (
                self.data['works_data'][work][k] for k in ['id', 'title', 'year', 'authors', 'venue']
            )

            # Must convert authors data to json string
            author_string = json.dumps(authors)

            try:
                summary = self.driver.execute_query("""
                    CREATE (n:PUBLICATION {id: $pub_id, title: $pub_title, year: $pub_year, authors: $pub_authors, venue: $pub_venue})
                    """,
                    pub_id = pub_id, 
                    pub_title = title, 
                    pub_year = year, 
                    pub_authors = author_string, 
                    pub_venue = venue,
                    database = self.db,
                ).summary
                print("Added {nodes_created} nodes in {time} ms.".format(
                    nodes_created = summary.counters.nodes_created,
                    time = summary.result_available_after
                ))

            except KeyError as ke:
                print(f"Missing key in work data: {ke}")
            except Neo4jError as ne:
                print(f"Neo4j error while inserting '{title}': {ne}")
            except Exception as e:  # Keep broad catch to continue bulk ingestion
                print(f"Unexpected error creating PUBLICATION node: {e}")

        print("All nodes were successfully added.")


    def node_count(self):
        """
        Print total node count
        """
        result = self.driver.execute_query("""
            MATCH (n) RETURN count(n) AS node_count
        """,
        database = self.db)
        count = result.records[0]["node_count"]
        print(f"Number of nodes: {count}")

    def edge_count(self):
        """
        Print total relationship count
        """
        result = self.driver.execute_query(
            """
            MATCH ()-[r]->() RETURN COUNT(r) AS totalRelationships
            """,
            database=self.db
        )
        count = result.records[0]["totalRelationships"]
        print(f"Total relationships: {count}")


    def delete_all_nodes(self):
        """
        Delete all nodes and relationships from the selected database
        """
        self.driver.execute_query("""
            MATCH (n) DETACH DELETE n
        """,
        database = self.db)
        print("All nodes were successfully deleted.")
        self.node_count()


    def add_covenue_edge(self):
        """
        Create COVENUE edges between publications that share the same venue
        Currently directional; community detection can treat as undirected
        """
        id_venue = {}
        for work_data in self.data['works_data'].values():
            pub_id = work_data['id']
            venue = work_data['venue']
            id_venue[pub_id] = venue

        created_edges = 0

        pub_keys = list(id_venue.keys())
        for i, pub1 in enumerate(pub_keys):
            for pub2 in pub_keys[i+1:]:
                if id_venue[pub1] == id_venue[pub2]:
                    #print(f"Trying edge between {pub1} and {pub2}, venue: {id_venue[pub1]}")
                    self.driver.execute_query("""
                        MATCH (p1:PUBLICATION {id: $pub_name1}), (p2:PUBLICATION {id: $pub_name2})
                        CREATE (p1) - [:COVENUE {venue: $pub_venue}] -> (p2)
                    """,
                    pub_name1 = pub1, 
                    pub_name2 = pub2, 
                    pub_venue = id_venue[pub1],
                    database = self.db
                    )
                    created_edges += 1
        
        print(f" Created {created_edges} CoVenue relationships.")
    
    
    def apply_leiden_clustering(self, gamma: float = 1.0, theta: float = 0.01, 
                                merge_small: bool = False, min_size: int = 3):
        """
        Apply Leiden community detection using Neo4j GDS
        Returns community assignments for each publication
        
        Args:
            gamma: Resolution parameter (higher values lead to more communities). Default: 1.0
            theta: Tolerance parameter for quality improvement (lower is more precise). Default: 0.01
            merge_small: Whether to merge small communities into larger ones. Default: False
            min_size: Minimum community size when merge_small=True. Smaller communities get merged. Default: 3
        """
        try:
            # First, drop existing graph projection if it exists (prevents conflicts)
            try:
                self.driver.execute_query("""
                    CALL gds.graph.drop('publicationGraph')
                """, database=self.db)
                print("Dropped existing graph projection.")
            except Neo4jError:
                pass  # Graph doesn't exist, that's fine
            
            # Create an in-memory graph projection
            self.driver.execute_query("""
                CALL gds.graph.project(
                    'publicationGraph',
                    'PUBLICATION',
                    {
                        COAUTHOR: {orientation: 'UNDIRECTED'},
                        COVENUE: {orientation: 'UNDIRECTED'},
                        COTITLE: {orientation: 'UNDIRECTED'}
                    },
                    {
                        relationshipProperties: ['weight', 'similarity']
                    }
                )
            """, database=self.db)
            print("Graph projection created.")
            
            # Run Leiden algorithm
            result = self.driver.execute_query("""
                CALL gds.leiden.stream('publicationGraph', {
                    gamma: $gamma,
                    theta: $theta,
                    includeIntermediateCommunities: false
                })
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId).id AS publicationId, 
                    gds.util.asNode(nodeId).title AS title,
                    communityId
                ORDER BY communityId
            """, gamma=gamma, theta=theta, database=self.db)
            
            # Process results
            communities = {}
            for record in result.records:
                pub_id = record["publicationId"]
                community = record["communityId"]
                title = record["title"]
                
                if community not in communities:
                    communities[community] = []
                communities[community].append({"id": pub_id, "title": title})
            
            print(f"Found {len(communities)} communities (before merging)")
            
            # Merge small communities if enabled
            if merge_small:
                communities = self._merge_small_communities(communities, min_size)
            
            # Drop the graph projection when done
            self.driver.execute_query("""
                CALL gds.graph.drop('publicationGraph')
            """, database=self.db)
            
            return communities
            
        except Neo4jError as e:
            print(f"Error running Leiden: {e}")
            # Try to cleanup graph projection on error
            try:
                self.driver.execute_query("""
                    CALL gds.graph.drop('publicationGraph')
                """, database=self.db)
            except:
                pass
            return None
    
    def _merge_small_communities(self, communities: Dict[int, List[Dict]], min_size: int = 3) -> Dict[int, List[Dict]]:
        """
        Merge small communities into larger ones based on shared connections
        
        Args:
            communities: Dictionary of community_id -> list of publications
            min_size: Communities smaller than this get merged into larger ones
        
        Returns:
            Updated communities dictionary with small communities merged
        """
        # Separate small and large communities
        small_communities = {cid: members for cid, members in communities.items() if len(members) < min_size}
        large_communities = {cid: members for cid, members in communities.items() if len(members) >= min_size}
        
        if not small_communities:
            print(f"No small communities to merge (all >= {min_size})")
            return communities
        
        if not large_communities:
            print(f"No large communities to merge into (all < {min_size})")
            return communities
        
        print(f"Merging {len(small_communities)} small communities into {len(large_communities)} larger ones...")
        
        merged_count = 0
        orphans = []
        
        # For each small community, find the best large community to merge with
        for small_cid, small_members in small_communities.items():
            best_target_cid = None
            best_score = 0
            
            # Extract publication IDs from small community
            small_pub_ids = {pub['id'] for pub in small_members}
            
            # Check connectivity to each large community
            for large_cid, large_members in large_communities.items():
                large_pub_ids = {pub['id'] for pub in large_members}
                
                # Count shared edges between small and large community
                shared_edges = self._count_shared_edges(small_pub_ids, large_pub_ids)
                
                # Calculate score (shared edges / size of small community)
                # This favors communities with more connections per publication
                score = shared_edges / len(small_members) if len(small_members) > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_target_cid = large_cid
            
            # Merge if we found a good match (at least 1 connection per publication on average)
            if best_target_cid is not None and best_score > 0.5:
                large_communities[best_target_cid].extend(small_members)
                merged_count += 1
            else:
                # No good match found, keep as orphans
                orphans.extend(small_members)
        
        # Add orphans as a separate "unclustered" community if they exist
        if orphans:
            # Find the next available community ID
            max_cid = max(large_communities.keys()) if large_communities else 0
            large_communities[max_cid + 1] = orphans
            print(f"  Merged {merged_count} small communities")
            print(f"  Created orphan group with {len(orphans)} publications (no strong connections found)")
        else:
            print(f"  Successfully merged all {merged_count} small communities")
        
        print(f"Final community count: {len(large_communities)}")
        return large_communities
    
    def _count_shared_edges(self, pub_ids1: set, pub_ids2: set) -> int:
        """
        Count the number of edges (relationships) between two sets of publications
        
        Args:
            pub_ids1: Set of publication IDs from first community
            pub_ids2: Set of publication IDs from second community
        
        Returns:
            Number of edges connecting the two sets
        """
        try:
            # Count relationships between the two sets of publications
            result = self.driver.execute_query("""
                MATCH (p1:PUBLICATION)-[r]-(p2:PUBLICATION)
                WHERE p1.id IN $ids1 AND p2.id IN $ids2
                RETURN count(r) as edge_count
            """, ids1=list(pub_ids1), ids2=list(pub_ids2), database=self.db)
            
            return result.records[0]["edge_count"] if result.records else 0
        except Neo4jError as e:
            print(f"Error counting edges: {e}")
            return 0

    def add_coauthor_edge(self):
        """
        Create COAUTHOR edges between publications that share at least one author
        Adds a `weight` equal to the number of shared authors
        Currently directional; community detection can treat as undirected
        """

        works_data = self.data['works_data']

        #print(works_data)
        created_edges = 0

        pub_keys = list(works_data.keys())
        # Creates a set of pub1 authors
        for i, pub1 in enumerate(pub_keys):
            authors1 = works_data[pub1]['authors']
            set1 = {(a['id'], a['name']) for a in authors1}

            # Set of pub2 authors
            for pub2 in pub_keys[i+1:]:
                authors2 = works_data[pub2]['authors']
                set2 = {(a['id'], a['name']) for a in authors2}

                # Shared authors between two publications, including ambiguous name
                shared_authors = set1 & set2

                #### FUTURE: may need to create metric for number of shared authors for weighted edge
                if shared_authors:
                    shared_authors_json = [{"id": aid, "name": name} for (aid, name) in shared_authors]
                    json_string = json.dumps(shared_authors_json)

                    # Add weight for COAUTHOR (# of shared authors)
                    weight = len(shared_authors)

                    self.driver.execute_query(
                        """
                        MATCH (p1:PUBLICATION {id: $pub_name1}), (p2:PUBLICATION {id: $pub_name2})
                        CREATE (p1)-[:COAUTHOR {coauthor: $pub_coauthor, weight: $weight}]->(p2)
                        """,
                        pub_name1=pub1,
                        pub_name2=pub2,
                        pub_coauthor=json_string,
                        weight = weight,
                        database=self.db
                    )
                    created_edges += 1
                    #print(f" Created {created_edges} relationships.")
            
        print(f" Created {created_edges} CoAuthor relationships.")


    def add_cotitle_edge(self, min_similarity: float = 0.35, top_k: int = 5):
        """
        Create COTITLE edges using TF-IDF cosine similarity between publication titles.

        Args
            min_similarity: minimum cosine similarity to create an edge
            top_k: for each publication, create edges only to its top_k most similar publications

        Requirements
            scikit-learn must be installed (see requirements.txt)
        """
        if TfidfVectorizer is None or cosine_similarity is None:
            print("Cannot add CoTitle edges: scikit-learn not installed.")
            return

        works_data: Dict[str, Dict] = self.data['works_data']
        pub_ids: List[str] = list(works_data.keys())
        titles: List[str] = [works_data[pid].get('title') or '' for pid in pub_ids]

        # Vectorize titles
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=10000
        )
        tfidf = vectorizer.fit_transform(titles)

        # Compute cosine similarities
        sim_matrix = cosine_similarity(tfidf)

        created_edges = 0

        for i, src_id in enumerate(pub_ids):
            # Get top_k similar indices excluding self
            sims = sim_matrix[i]
            # Create list of (j, sim) excluding i
            candidates: List[Tuple[int, float]] = [(j, float(sims[j])) for j in range(len(pub_ids)) if j != i]
            # Sort by similarity descending
            candidates.sort(key=lambda x: x[1], reverse=True)
            # Take top_k above threshold
            selected = [(j, s) for (j, s) in candidates[:top_k] if s >= min_similarity]

            for j, sim in selected:
                dst_id = pub_ids[j]
                try:
                    self.driver.execute_query(
                        """
                        MATCH (p1:PUBLICATION {id: $pub1}), (p2:PUBLICATION {id: $pub2})
                        CREATE (p1)-[:COTITLE {similarity: $sim}]->(p2)
                        """,
                        pub1=src_id,
                        pub2=dst_id,
                        sim=float(sim),
                        database=self.db
                    )
                    created_edges += 1
                except Neo4jError as ne:
                    print(f"Neo4j error while adding COTITLE between {src_id} and {dst_id}: {ne}")

        print(f" Created {created_edges} CoTitle relationships (min_similarity={min_similarity}, top_k={top_k}).")


if __name__ == "__main__":

    # Configure your connection and cached data path here
    URI = "neo4j://127.0.0.1:7687"  # Example local instance
    USER = "neo4j"
    PASSWORD = "12345678"
    DB = "neo4j"
    # Example cache path: created by running neo4j_data.py
    PATH = "cache/David Nathan_data.json"

    imp = Neo4jImportData(URI, USER, PASSWORD, DB, PATH)
    imp.delete_all_nodes()
    imp.publication_as_nodes()
    #imp.node_count()
    #imp.delete_all_nodes()
    #imp.close()
    imp.add_covenue_edge()
    imp.add_coauthor_edge()
    # Add CoTitle edges using TF-IDF title similarity
    imp.add_cotitle_edge(min_similarity=0.35, top_k=5)
    imp.node_count()
    imp.edge_count()

    # =====================================================
    # TEST 1: Gamma=1.0, No merging (baseline)
    # =====================================================
    print("\n" + "="*60)
    print("TEST 1: Baseline - gamma=1.0, merge_small=False")
    print("="*60)
    
    communities_baseline = imp.apply_leiden_clustering(
        gamma=1.0,
        theta=0.01,
        merge_small=False
    )
    
    # Display results
    if communities_baseline:
        print(f"\n{'='*60}")
        print(f"RESULTS: Baseline (gamma=1.0, no merging)")
        print(f"{'='*60}\n")
        
        # Statistics
        sizes_baseline = [len(members) for members in communities_baseline.values()]
        print(f"Total communities: {len(communities_baseline)}")
        print(f"Average size: {sum(sizes_baseline)/len(sizes_baseline):.1f}")
        print(f"Median size: {sorted(sizes_baseline)[len(sizes_baseline)//2]}")
        print(f"Size range: {min(sizes_baseline)} - {max(sizes_baseline)}")
        
        # Distribution
        singletons = sum(1 for s in sizes_baseline if s == 1)
        small = sum(1 for s in sizes_baseline if 2 <= s <= 5)
        medium = sum(1 for s in sizes_baseline if 6 <= s <= 20)
        large = sum(1 for s in sizes_baseline if s > 20)
        print(f"\nSize distribution:")
        print(f"  Singletons (1): {singletons}")
        print(f"  Small (2-5): {small}")
        print(f"  Medium (6-20): {medium}")
        print(f"  Large (>20): {large}")
        
        # Sort and display communities
        sorted_communities = sorted(communities_baseline.items(), 
                                    key=lambda x: len(x[1]), 
                                    reverse=True)
        
        print(f"\nTop 5 communities:")
        for comm_id, members in sorted_communities[:5]:
            print(f"\nCommunity {comm_id}: {len(members)} publications")
            for i, pub in enumerate(members[:3]):
                print(f"  {i+1}. {pub['title']}")
            if len(members) > 3:
                print(f"  ... and {len(members) - 3} more")
        
        # Save results
        output_path_baseline = PATH.replace('_data.json', '_communities_baseline_gamma1.0.json')
        with open(output_path_baseline, 'w', encoding='utf-8') as f:
            json.dump(communities_baseline, f, indent=2, ensure_ascii=False)
        print(f"\nBaseline results saved to: {output_path_baseline}")

    # =====================================================
    # TEST 2: Gamma=0.5, Merge small (min_size=5)
    # =====================================================
    print("\n" + "="*60)
    print("TEST 2: With merging - gamma=0.5, merge_small=True, min_size=5")
    print("="*60)
    
    communities_merged = imp.apply_leiden_clustering(
        gamma=0.5,
        theta=0.01,
        merge_small=True,
        min_size=5
    )
    
    # Display results
    if communities_merged:
        print(f"\n{'='*60}")
        print(f"RESULTS: With merging (gamma=0.5, min_size=5)")
        print(f"{'='*60}\n")
        
        # Statistics
        sizes_merged = [len(members) for members in communities_merged.values()]
        print(f"Total communities: {len(communities_merged)}")
        print(f"Average size: {sum(sizes_merged)/len(sizes_merged):.1f}")
        print(f"Median size: {sorted(sizes_merged)[len(sizes_merged)//2]}")
        print(f"Size range: {min(sizes_merged)} - {max(sizes_merged)}")
        
        # Distribution
        singletons = sum(1 for s in sizes_merged if s == 1)
        small = sum(1 for s in sizes_merged if 2 <= s <= 5)
        medium = sum(1 for s in sizes_merged if 6 <= s <= 20)
        large = sum(1 for s in sizes_merged if s > 20)
        print(f"\nSize distribution:")
        print(f"  Singletons (1): {singletons}")
        print(f"  Small (2-5): {small}")
        print(f"  Medium (6-20): {medium}")
        print(f"  Large (>20): {large}")
        
        # Sort and display communities
        sorted_communities = sorted(communities_merged.items(), 
                                    key=lambda x: len(x[1]), 
                                    reverse=True)
        
        print(f"\nTop 5 communities:")
        for comm_id, members in sorted_communities[:5]:
            print(f"\nCommunity {comm_id}: {len(members)} publications")
            for i, pub in enumerate(members[:3]):
                print(f"  {i+1}. {pub['title']}")
            if len(members) > 3:
                print(f"  ... and {len(members) - 3} more")
        
        # Save results
        output_path_merged = PATH.replace('_data.json', '_communities_merged_gamma0.5_min5.json')
        with open(output_path_merged, 'w', encoding='utf-8') as f:
            json.dump(communities_merged, f, indent=2, ensure_ascii=False)
        print(f"\nMerged results saved to: {output_path_merged}")

    imp.close()
