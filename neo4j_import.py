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
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

class Neo4jImportData:
    def close(self):
        self.driver.close()

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
        self.db = db
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user,password))
            self.driver.verify_connectivity()
            print("Connection to Neo4j database successful!")
        except ServiceUnavailable as e:
            raise RuntimeError(f"❌ Connection failed: {e}")

        # ✅ Load the JSON cache first
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # ✅ Optional: quick sanity check
        if not isinstance(self.data, dict):
            raise ValueError(f"Unexpected JSON format in {data_path}. Expected a dictionary at top level.")

    def publication_as_nodes(self):
        """
        Create PUBLICATION nodes with properties id, title, year, authors (JSON string), venue
        """
        print("Creating PUBLICATION nodes...")
        created = 0
        for au_name, info in self.data['authors'].items():
            if 'works_data' not in info:
                continue
            for work in info['works_data']:
                pub_id, title, year, authors, venue, keywordinfo= (
                info['works_data'][work][k] for k in ['id', 'title', 'year', 'authors', 'venue', 'keywords']
                )
                keywords = []
                for k in keywordinfo:
                    keywords.append(k['name'])

                # Must convert authors data to json string
                author_string = json.dumps(authors)
                
                try:
                    summary = self.driver.execute_query(
                        """
                        CREATE (n:PUBLICATION {
                            id: $pub_id,
                            title: $pub_title,
                            year: $pub_year,
                            authors: $pub_authors,
                            venue: $pub_venue,
                            keywords: $pub_keywords
                        })
                        """,
                        pub_id=pub_id,
                        pub_title=title,
                        pub_year=year,
                        pub_authors=author_string,
                        pub_venue=venue,
                        pub_keywords = keywords,
                        database=self.db,
                    ).summary
                    created += summary.counters.nodes_created
                except Neo4jError as e:
                    print(f"Neo4j error creating node for '{title}': {e}")
        print(f"✅ Created {created} PUBLICATION nodes total.")
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

    #fall 25 update:
    def add_keyword_edge(self, min_shared: int = 3):
        """
        Create COKEYWORD edges between publications that share at least two words
        want: adds a weight equal to the number of keywords shared
        Currently directional; community detection can treat as undirected
        """
        created_edges = 0
        works_data = {}

        for au_name, info in self.data['authors'].items():
            if 'works_data' not in info:
                continue
            for wid, work in info['works_data'].items():
                works_data[wid] = work

        pub_keys = list(works_data.keys())

        for i, pub1 in enumerate(pub_keys):
            k1 = works_data[pub1]['keywords']

            set1 = {(a['id'], a['name']) for a in k1}
            if not k1:
                continue
            for pub2 in pub_keys[i+1:]:
                 
                k2 = works_data[pub2]['keywords']
                set2 = {(a['id'], a['name']) for a in k2}
                if not k2:
                    continue
                shared = set1&set2
            
                if len(shared) >= min_shared:
                    shared_authors_json = [{"id": aid, "name": name} for (aid, name) in shared]
                    json_string = json.dumps(shared_authors_json)

                    try:
                        self.driver.execute_query(
                            """
                            MATCH (p1:PUBLICATION {id: $pub1}), (p2:PUBLICATION {id: $pub2})
                            CREATE (p1)-[:COKEYWORD {shared_keywords: $shared, weight: $count}]->(p2)
                            """,
                            pub1=pub1,
                            pub2=pub2,
                            shared=list(json_string),
                            count=len(shared),
                            database=self.db
                        )
                        created_edges += 1
                    except Exception as e:
                        print(f"Error creating COKEYWORD edge between {pub1} and {pub2}: {e}")


        print(f"Created {created_edges} COKEYWORD relationships (min_shared={min_shared}).")

    def add_covenue_edge(self):
        """Create COVENUE edges between publications with same venue"""
        id_venue = {} #maps work id to venue

        for au_name, info in self.data['authors'].items():
            if 'works_data' not in info:
                continue
            for work_data in info['works_data'].values():
                pub_id = work_data['id']
                venue = str(work_data['venue'])
                id_venue[pub_id] = venue

        created = 0
        pub_keys = list(id_venue.keys())

        for i, p1 in enumerate(pub_keys):
            for p2 in pub_keys[i + 1 :]:
                if id_venue[p1] and id_venue[p1] == id_venue[p2]:
                    self.driver.execute_query(
                        """
                        MATCH (a:PUBLICATION {id:$p1}), (b:PUBLICATION {id:$p2})
                        CREATE (a)-[:COVENUE {venue:$v}]->(b)
                        """,
                        p1=p1,
                        p2=p2,
                        v=id_venue[p1],
                        database=self.db,
                    )
                    created += 1
        print(f"✅ Created {created} COVENUE edges.")
    def add_coauthor_edge(self):
        """Create COAUTHOR edges between publications sharing ≥1 author"""
        created = 0
        works_data = {}

        for au_name, info in self.data['authors'].items():
            if 'works_data' not in info:
                continue
            for wid, work in info['works_data'].items():
                works_data[wid] = work

        wkeys = list(works_data.keys())

        for i, pub1 in enumerate(wkeys):
            authors1 = works_data[pub1]['authors']

            set1 = {(a['id'], a['name']) for a in authors1}

            for pub2 in wkeys[i + 1 :]:
                authors2 = works_data[pub2]['authors']

                set2 = {(a['id'], a['name']) for a in authors2}

                shared = set1 & set2
                
                if shared:
                    shared_authors_json = [{"id": aid, "name": name} for (aid, name) in shared]
                    json_string = json.dumps(shared_authors_json)

                    self.driver.execute_query(
                        """
                        MATCH (p1:PUBLICATION {id:$p1}), (p2:PUBLICATION {id:$p2})
                        CREATE (p1)-[:COAUTHOR {shared:$shared, weight:$w}]->(p2)
                        """,
                        p1=pub1,
                        p2=pub2,
                        shared=list(json_string),
                        w=len(json_string),
                        database=self.db
                    )
                    created += 1
        print(f"✅ Created {created} COAUTHOR edges.")

    def add_cotitle_edge(self, min_similarity: float = 0.35, top_k: int = 5):
        """Add COTITLE edges using TF-IDF cosine similarity"""
        if not (TfidfVectorizer and cosine_similarity):
            print("⚠️  scikit-learn not installed; skipping COTITLE edges.")
            return
        
        works_data = {}
        
        for au_name, info in self.data['authors'].items():
            if 'works_data' not in info:
                continue
            for wid, work in info['works_data'].items():
                works_data[wid] = work

        pub_ids = list(works_data.keys())

        titles: List[str] = [works_data[pid]['title'] or '' for pid in pub_ids]

        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        tfidf = vectorizer.fit_transform(titles)
        sim_matrix = cosine_similarity(tfidf)
        created = 0
        ##start here next time

        if TfidfVectorizer is None or cosine_similarity is None:
            print("Cannot add COTITLE edges: scikit-learn not installed.")
            return
        for i, src_id in enumerate(pub_ids):

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
                    created += 1
                except Neo4jError as ne:
                    print(f"Neo4j error while adding COTITLE between {src_id} and {dst_id}: {ne}")

        print(f" Created {created} CoTitle relationships (min_similarity={min_similarity}, top_k={top_k}).")

    def apply_louvain_clustering(self):
        """Apply Louvain community detection"""
        try:
            self.driver.execute_query(
                """
                CALL gds.graph.project(
                    'publicationGraph',
                    ['PUBLICATION'],
                    {
                        COAUTHOR: {orientation:'UNDIRECTED'},
                        COVENUE: {orientation:'UNDIRECTED'},
                        COTITLE: {
                            orientation: 'UNDIRECTED',
                            properties: ['weight', 'similarity']
                        },
                        COKEYWORD: {orientation:'UNDIRECTED'}
                    }
                )
                """,
                database=self.db,
            )
            print("🧠 Graph projection created.")
            # ---------------------------------------------------------
            # 3. Run Louvain (stream)
            # ---------------------------------------------------------
            result = self.driver.execute_query(
                """
                CALL gds.louvain.stream('publicationGraph')
                YIELD nodeId, communityId
                RETURN gds.util.asNode(nodeId).id AS pubId, communityId
                ORDER BY communityId
                """,
                database=self.db,
            )
            # ---------------------------------------------------------
            # 4. Community sizes
            # ---------------------------------------------------------
            result_sizes = self.driver.execute_query(
                """
                CALL gds.louvain.stream('publicationGraph')
                YIELD communityId
                RETURN communityId, count(*) AS size
                ORDER BY size DESC
                """,
                database=self.db,
            )
            # ---------------------------------------------------------
            # 5. Modularity stats
            # ---------------------------------------------------------
            result_mod = self.driver.execute_query(
                """
                CALL gds.louvain.stats('publicationGraph')
                YIELD modularity
                RETURN modularity
                """,
                database=self.db,
            )
            # ---------------------------------------------------------
            # 6. Format results
            # ---------------------------------------------------------
            comms = {}
            for r in result.records:
                comms.setdefault(r["communityId"], []).append(r["pubId"])

            print(f"✅ Found {len(comms)} communities.")

            print("\n📊 Community sizes:")
            for row in result_sizes.records:
                print(f"  Community {row['communityId']}: {row['size']} nodes")

            print("\n📈 Modularity:")
            print(result_mod.records[0]["modularity"])

            # ---------------------------------------------------------
            # 7. Drop graph after analysis
            # ---------------------------------------------------------
            self.driver.execute_query(
                "CALL gds.graph.drop('publicationGraph')",
                database=self.db
            )
            print("🧹 GDS graph dropped.")
            return comms
        except Neo4jError as e:
            print(f"⚠️ Louvain error: {e}")
            return None
    
    def run_leiden(G: nx.Graph, resolution: float = 1.0, seed: int = 42):
        """
        Run Leiden (CPM objective) on a NetworkX graph.
        Returns:
        (partition_dict, quality)
        """
        if ig is None or la is None:
            raise RuntimeError("Leiden not installed. `pip install python-igraph leidenalg`")

        # Map NetworkX nodes -> indices
        nodes = list(G.nodes())
        index_of = {n: i for i, n in enumerate(nodes)}

        # Build igraph from NetworkX
        edges = [(index_of[u], index_of[v]) for u, v in G.edges()]
        weights = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges()]
        g = ig.Graph(n=len(nodes), edges=edges, directed=False)
        g.es["weight"] = weights

        # Leiden with Constant Potts Model (CPM) resolution parameter
        part = la.find_partition(
            g,
            la.CPMVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=resolution,
            seed=seed
        )
        membership = part.membership
        partition = {nodes[i]: int(membership[i]) for i in range(len(nodes))}
        quality = part.quality()
        return partition, quality
    
    def load_pub_graph_from_neo4j(uri, user, password, db,
                              coauthor_scale: float = 1.0,
                              covenue_scale: float = 1.0,
                              cotitle_scale: float = 1.2,
                              cokeyword_scale: float = 3,
                              use_log_coauthor: bool = True) -> nx.Graph:
        """
        Build a weighted, undirected NetworkX graph from Neo4j PUBLICATION relationships.
        Weights:
        - COAUTHOR: coauthor_scale * (log(1 + weight) if use_log_coauthor else weight; defaults to 1.0 if missing)
        - COVENUE : covenue_scale * 1.0 (if r.weight is null or 0) else r.weight
        - COTITLE : cotitle_scale * coalesce(r.similarity, 0.0)
        - COKEYWORD: 
        """

        q = """
        MATCH (p1:PUBLICATION)-[r:COAUTHOR|COVENUE|COTITLE|COKEYWORD]-(p2:PUBLICATION)
        WITH p1.id AS a, p2.id AS b, type(r) AS t, r
        WITH a, b, t, r,
            CASE
            WHEN t = 'COAUTHOR' THEN $coauthorScale *
                    CASE
                    WHEN r.weight IS NULL THEN 1.0
                    ELSE (CASE WHEN $useLog THEN log(1 + toFloat(r.weight))
                                ELSE toFloat(r.weight) END)
                    END
            WHEN t = 'COVENUE'  THEN $covenueScale *
                    CASE
                    WHEN r.weight IS NULL OR toFloat(r.weight) = 0 THEN 1.0
                    ELSE toFloat(r.weight)
                    END
            WHEN t = 'COTITLE'  THEN $cotitleScale * coalesce(toFloat(r.similarity), 0.0)
            ELSE 0.0
            END AS w
        RETURN a, b, w
        """

        G = nx.Graph()
        with self.driver.session(database=db) as session:
            for rec in session.run(q,
                                coauthorScale=coauthor_scale,
                                covenueScale=covenue_scale,
                                cotitleScale=cotitle_scale,
                                cokeywordScale = cokeyword_scale,
                                useLog=use_log_coauthor):
                a, b, w = rec["a"], rec["b"], float(rec["w"])
                if not a or not b or a == b or w <= 0.0:
                    continue
                if G.has_edge(a, b):
                    G[a][b]["weight"] += w
                else:
                    G.add_edge(a, b, weight=w)
        return G
    def run_louvain(G: nx.Graph, resolution: float = 1.0, seed: int = 42):
        """
        Run Louvain on a NetworkX graph.
        Returns:
        (partition_dict, modularity)
        """

        part = community_louvain.best_partition(G, weight="weight",
                                                resolution=resolution,
                                                random_state=seed)
        Q = community_louvain.modularity(part, G, weight="weight")
        return part, Q


if __name__ == "__main__":
    # Configure your connection and cached data path here
    URI = "neo4j://127.0.0.1:7687"  # Example local instance
    USER = "neo4j"
    PASSWORD = "neo4jneo4j"
    DB = "neo4j"
    # Example cache path: created by running neo4j_data.py
    PATH = "cache/all_authors_data.json"
    imp = Neo4jImportData(URI, USER, PASSWORD, DB, PATH)

    # Load graph from Neo4j
    G = load_pub_graph_from_neo4j(uri, user, password, db)

    if method == "louvain":
        partition, modularity = run_louvain(G)
        for i, j in partition.items():
            print(i)
            print(": ")
            print(j)
        print(f"Louvain partition size: {len(set(partition.values()))}")
        print(f"Louvain modularity: {modularity:.4f}")
    elif method == "leiden":
        partition, quality = run_leiden(G)
        print(f"Leiden partition size: {len(set(partition.values()))}")
        print(f"Leiden quality: {quality:.4f}")


    '''
    imp.delete_all_nodes()
    imp.publication_as_nodes()

    imp.add_covenue_edge()
    imp.add_coauthor_edge()

    # Add CoTitle edges using TF-IDF title similarity
    imp.add_cotitle_edge(min_similarity=0.35, top_k=5)
    imp.add_keyword_edge()
    imp.node_count()
    imp.edge_count()
    #Apply Louvain clustering
    print("\n" + "="*50)
    print("Running Louvain clustering...")
    print("="*50)

    communities = imp.apply_louvain_clustering()
    
    if communities:
        output = PATH.replace("_data.json", "_communities.json")
        with open(output, "w", encoding="utf-8") as f:
            json.dump(communities, f, indent=2, ensure_ascii=False)
        print(f"🧾 Communities saved to {output}")
    '''
    
    imp.close()