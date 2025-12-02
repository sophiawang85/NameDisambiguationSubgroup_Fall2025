"""
Neo4j import script with external Leiden clustering

Purpose
- Read the cached JSON produced by `neo4j_data.py`
- Create publication nodes and similarity edges in Neo4j for disambiguation experiments
- Use external leidenalg library with Constant Potts Model (best performer from prior research)

New features
- External Leiden clustering using leidenalg Python library
- Constant Potts Model Vertex Partition (best metric from Mikaela's research)
- Adaptive resolution parameter selection
- Custom merge_small functionality
"""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, Neo4jError
import json
from typing import List, Dict, Tuple, Optional
import csv

# For CoTitle similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as _e:
    TfidfVectorizer = None
    cosine_similarity = None

# For external Leiden clustering
try:
    import igraph as ig
    import leidenalg as la
except ImportError as _e:
    ig = None
    la = None
    print("Warning: igraph and leidenalg not installed. External Leiden clustering unavailable.")
    print("Install with: pip install python-igraph leidenalg")


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
            except Exception as e:
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
    

    def add_coauthor_edge(self):
        """
        Create COAUTHOR edges between publications that share at least one author
        Adds a `weight` equal to the number of shared authors
        Currently directional; community detection can treat as undirected
        """

        works_data = self.data['works_data']

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


    def _export_graph_from_neo4j(self) -> Optional[Tuple[ig.Graph, Dict[int, str], Dict[str, int]]]:
        """
        Export the Neo4j graph to an igraph Graph object for external Leiden clustering
        
        Returns:
            Tuple of (igraph.Graph, id_to_pub, pub_to_id) or None if libraries not available
            - Graph: igraph Graph object with weighted edges
            - id_to_pub: mapping from vertex index to publication ID
            - pub_to_id: mapping from publication ID to vertex index
        """
        if ig is None or la is None:
            print("Cannot export graph: igraph/leidenalg not installed")
            return None
        
        # Get all publications
        result = self.driver.execute_query("""
            MATCH (p:PUBLICATION)
            RETURN p.id AS pub_id, p.title AS title
            ORDER BY pub_id
        """, database=self.db)
        
        publications = [(record["pub_id"], record["title"]) for record in result.records]
        
        # Create mappings
        pub_to_id = {pub_id: idx for idx, (pub_id, _) in enumerate(publications)}
        id_to_pub = {idx: pub_id for idx, (pub_id, _) in enumerate(publications)}
        
        # Create igraph
        g = ig.Graph(n=len(publications))
        g.vs["name"] = [pub_id for pub_id, _ in publications]
        g.vs["title"] = [title for _, title in publications]
        
        # Get all relationships with weights
        result = self.driver.execute_query("""
            MATCH (p1:PUBLICATION)-[r]->(p2:PUBLICATION)
            RETURN p1.id AS src, p2.id AS dst, 
                   type(r) AS rel_type,
                   COALESCE(r.weight, 1.0) AS weight,
                   COALESCE(r.similarity, 1.0) AS similarity
        """, database=self.db)
        
        edges = []
        weights = []
        
        for record in result.records:
            src_id = pub_to_id[record["src"]]
            dst_id = pub_to_id[record["dst"]]
            rel_type = record["rel_type"]
            
            # Use weight for COAUTHOR, similarity for COTITLE, 1.0 for COVENUE
            if rel_type == "COAUTHOR":
                weight = float(record["weight"])
            elif rel_type == "COTITLE":
                weight = float(record["similarity"])
            else:  # COVENUE
                weight = 1.0
            
            edges.append((src_id, dst_id))
            weights.append(weight)
        
        # Add edges to graph
        g.add_edges(edges)
        g.es["weight"] = weights
        
        # Convert to undirected (sum weights for duplicate edges)
        g = g.as_undirected(mode="collapse", combine_edges="sum")
        
        print(f"Exported graph: {g.vcount()} vertices, {g.ecount()} edges")
        
        return g, id_to_pub, pub_to_id


    def apply_external_leiden_clustering(
        self,
        resolution_parameter: Optional[float] = None,
        auto_resolution: bool = True,
        merge_small: bool = False,
        min_size: int = 3,
        n_iterations: int = 2,
        seed: Optional[int] = None
    ) -> Optional[Dict[int, List[Dict]]]:
        """
        Apply Leiden clustering using external leidenalg library with Constant Potts Model
        This was found to be the best partition type in prior research (Mikaela's report)
        
        Args:
            resolution_parameter: Resolution parameter for clustering. If None and auto_resolution=True,
                                 will use adaptive selection (recommended). Default: None
            auto_resolution: If True, automatically select resolution parameter using adaptive algorithm.
                           This finds the most detailed partition before cluster count spikes. Default: True
            merge_small: Whether to merge small communities into larger ones. Default: False
            min_size: Minimum community size when merge_small=True. Default: 3
            n_iterations: Number of iterations for the Leiden algorithm. Default: 2
            seed: Random seed for reproducibility. Default: None
        
        Returns:
            Dictionary mapping community_id to list of publications, or None if libraries unavailable
        """
        if ig is None or la is None:
            print("Cannot run external Leiden: igraph/leidenalg not installed")
            print("Install with: pip install python-igraph leidenalg")
            return None
        
        # Export graph from Neo4j
        graph_data = self._export_graph_from_neo4j()
        if graph_data is None:
            return None
        
        g, id_to_pub, pub_to_id = graph_data
        
        # Determine resolution parameter
        if auto_resolution and resolution_parameter is None:
            print("Using adaptive resolution parameter selection...")
            resolution_parameter = self._find_optimal_resolution(g)
            print(f"Selected resolution parameter: {resolution_parameter:.3f}")
        elif resolution_parameter is None:
            resolution_parameter = 1.0
            print(f"Using default resolution parameter: {resolution_parameter}")
        
        # Run Leiden with Constant Potts Model (best from research)
        print(f"Running Leiden clustering with Constant Potts Model...")
        print(f"  Resolution: {resolution_parameter}")
        print(f"  Iterations: {n_iterations}")
        print(f"  Random seed: {seed}")
        
        partition = la.find_partition(
            g,
            la.CPMVertexPartition,  # Constant Potts Model - best performer
            weights='weight',
            resolution_parameter=resolution_parameter,
            n_iterations=n_iterations,
            seed=seed
        )
        
        print(f"Clustering complete. Modularity: {partition.modularity:.4f}")
        print(f"Found {len(partition)} communities (before merging)")
        
        # Convert to output format
        communities = {}
        for comm_id, community in enumerate(partition):
            communities[comm_id] = [
                {
                    "id": id_to_pub[vertex_idx],
                    "title": g.vs[vertex_idx]["title"]
                }
                for vertex_idx in community
            ]
        
        # Merge small communities if requested
        if merge_small:
            communities = self._merge_small_communities_external(
                communities, g, pub_to_id, min_size
            )
        
        return communities


    def _find_optimal_resolution(self, g: ig.Graph, 
                                 coarse_step: float = 0.1,
                                 fine_step: float = 0.01) -> float:
        """
        Adaptive resolution parameter selection based on Mikaela's algorithm
        
        Finds the most detailed partition before the number of clusters spikes
        
        Args:
            g: igraph Graph object
            coarse_step: Initial step size for coarse search. Default: 0.1
            fine_step: Fine step size for detailed search. Default: 0.01
        
        Returns:
            Optimal resolution parameter
        """
        # Coarse search from 0.1 to 1.0
        coarse_resolutions = [i * coarse_step for i in range(1, 11)]  # 0.1 to 1.0
        coarse_cluster_counts = []
        
        print("  Coarse search (step=0.1)...")
        for res in coarse_resolutions:
            partition = la.find_partition(
                g,
                la.CPMVertexPartition,
                weights='weight',
                resolution_parameter=res,
                n_iterations=2
            )
            coarse_cluster_counts.append(len(partition))
        
        # Find biggest increase in coarse search
        max_increase = 0
        max_increase_idx = 0
        for i in range(len(coarse_cluster_counts) - 1):
            increase = coarse_cluster_counts[i + 1] - coarse_cluster_counts[i]
            if increase > max_increase:
                max_increase = increase
                max_increase_idx = i
        
        # Fine search in the range with biggest increase
        lower = coarse_resolutions[max_increase_idx]
        upper = coarse_resolutions[max_increase_idx + 1]
        
        print(f"  Fine search in range [{lower:.2f}, {upper:.2f}] (step=0.01)...")
        fine_resolutions = [lower + i * fine_step for i in range(int((upper - lower) / fine_step) + 1)]
        fine_cluster_counts = []
        
        for res in fine_resolutions:
            partition = la.find_partition(
                g,
                la.CPMVertexPartition,
                weights='weight',
                resolution_parameter=res,
                n_iterations=2
            )
            fine_cluster_counts.append(len(partition))
        
        # Find biggest increase in fine search
        max_increase = 0
        max_increase_idx = 0
        for i in range(len(fine_cluster_counts) - 1):
            increase = fine_cluster_counts[i + 1] - fine_cluster_counts[i]
            if increase > max_increase:
                max_increase = increase
                max_increase_idx = i
        
        # Return resolution right before the spike
        optimal_resolution = fine_resolutions[max_increase_idx]
        
        print(f"  Cluster counts at key resolutions:")
        print(f"    {lower:.2f}: {fine_cluster_counts[0]} clusters")
        print(f"    {optimal_resolution:.2f}: {fine_cluster_counts[max_increase_idx]} clusters (selected)")
        print(f"    {fine_resolutions[max_increase_idx + 1]:.2f}: {fine_cluster_counts[max_increase_idx + 1]} clusters (spike)")
        
        return optimal_resolution


    def _merge_small_communities_external(
        self,
        communities: Dict[int, List[Dict]],
        g: ig.Graph,
        pub_to_id: Dict[str, int],
        min_size: int = 3
    ) -> Dict[int, List[Dict]]:
        """
        Merge small communities using the igraph structure for faster edge counting
        
        Args:
            communities: Dictionary of community_id -> list of publications
            g: igraph Graph object
            pub_to_id: Mapping from publication ID to vertex index
            min_size: Communities smaller than this get merged
        
        Returns:
            Updated communities dictionary
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
            
            # Get vertex indices for small community
            small_vertices = {pub_to_id[pub['id']] for pub in small_members}
            
            # Check connectivity to each large community
            for large_cid, large_members in large_communities.items():
                # Get vertex indices for large community
                large_vertices = {pub_to_id[pub['id']] for pub in large_members}
                
                # Count edges between communities using igraph
                edge_count = 0
                for v in small_vertices:
                    neighbors = set(g.neighbors(v))
                    edge_count += len(neighbors & large_vertices)
                
                # Calculate score (shared edges / size of small community)
                score = edge_count / len(small_members) if len(small_members) > 0 else 0
                
                if score > best_score:
                    best_score = score
                    best_target_cid = large_cid
            
            # Merge if we found a good match
            if best_target_cid is not None and best_score > 0.5:
                large_communities[best_target_cid].extend(small_members)
                merged_count += 1
            else:
                orphans.extend(small_members)
        
        # Add orphans as a separate community if they exist
        if orphans:
            max_cid = max(large_communities.keys()) if large_communities else 0
            large_communities[max_cid + 1] = orphans
            print(f"  Merged {merged_count} small communities")
            print(f"  Created orphan group with {len(orphans)} publications")
        else:
            print(f"  Successfully merged all {merged_count} small communities")
        
        print(f"Final community count: {len(large_communities)}")
        return large_communities


    @staticmethod
    def export_communities_to_csv(communities: Dict[int, List[Dict]], output_path: str):
        """
        Export clustering results to CSV for manual review
        
        Creates a single CSV with one row per publication
        Columns: cluster_id, cluster_size, publication_id, title
        
        Args:
            communities: Dictionary mapping community_id to list of publications
            output_path: Path where CSV should be saved
        """
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'cluster_id',
                'cluster_size',
                'publication_id',
                'title'
            ])
            
            # Sort clusters by size (largest first), then by cluster_id
            sorted_clusters = sorted(communities.items(), 
                                    key=lambda x: (-len(x[1]), x[0]))
            
            for cluster_id, members in sorted_clusters:
                cluster_size = len(members)
                
                # Sort publications within cluster by title for consistency
                sorted_members = sorted(members, key=lambda x: x['title'])
                
                for pub in sorted_members:
                    writer.writerow([
                        cluster_id,
                        cluster_size,
                        pub['id'],
                        pub['title']
                    ])
        
        print(f"CSV exported to: {output_path}")


    @staticmethod
    def print_community_statistics(communities: Dict[int, List[Dict]]):
        """
        Print summary statistics about the clustering results
        
        Args:
            communities: Dictionary mapping community_id to list of publications
        """
        sizes = [len(members) for members in communities.values()]
        
        print("\n" + "="*60)
        print("CLUSTERING STATISTICS")
        print("="*60)
        print(f"Total clusters: {len(communities)}")
        print(f"Total publications: {sum(sizes)}")
        print(f"Average cluster size: {sum(sizes)/len(sizes):.1f}")
        print(f"Median cluster size: {sorted(sizes)[len(sizes)//2]}")
        print(f"Largest cluster: {max(sizes)}")
        print(f"Smallest cluster: {min(sizes)}")
        
        print("\nSize distribution:")
        singletons = sum(1 for s in sizes if s == 1)
        small = sum(1 for s in sizes if 2 <= s <= 5)
        medium = sum(1 for s in sizes if 6 <= s <= 20)
        large = sum(1 for s in sizes if s > 20)
        
        print(f"  Singletons (1):        {singletons:4d} ({singletons/len(sizes)*100:.1f}%)")
        print(f"  Small (2-5):           {small:4d} ({small/len(sizes)*100:.1f}%)")
        print(f"  Medium (6-20):         {medium:4d} ({medium/len(sizes)*100:.1f}%)")
        print(f"  Large (>20):           {large:4d} ({large/len(sizes)*100:.1f}%)")
        print("="*60 + "\n")


if __name__ == "__main__":

    # Configure your connection and cached data path here
    URI = "neo4j://127.0.0.1:7687"
    USER = "neo4j"
    PASSWORD = "12345678"
    DB = "neo4j"
    PATH = "cache/David Nathan_data.json"

    imp = Neo4jImportData(URI, USER, PASSWORD, DB, PATH)
    imp.delete_all_nodes()
    imp.publication_as_nodes()
    imp.add_covenue_edge()
    imp.add_coauthor_edge()
    imp.add_cotitle_edge(min_similarity=0.35, top_k=5)
    imp.node_count()
    imp.edge_count()

    # =====================================================
    # TEST 1: Auto-resolution with Constant Potts Model
    # =====================================================
    print("\n" + "="*60)
    print("TEST 1: External Leiden - Auto-resolution, no merging")
    print("Using Constant Potts Model (best from prior research)")
    print("="*60)
    
    communities_auto = imp.apply_external_leiden_clustering(
        auto_resolution=True,
        merge_small=False,
        seed=42  # For reproducibility
    )
    
    if communities_auto:
        print(f"\n{'='*60}")
        print(f"RESULTS: Auto-resolution (Constant Potts Model)")
        print(f"{'='*60}\n")
        
        # Print statistics using the new method
        Neo4jImportData.print_community_statistics(communities_auto)
        
        # Display top communities
        sorted_communities = sorted(communities_auto.items(), 
                                    key=lambda x: len(x[1]), 
                                    reverse=True)
        
        print(f"Top 5 communities:")
        for comm_id, members in sorted_communities[:5]:
            print(f"\nCommunity {comm_id}: {len(members)} publications")
            for i, pub in enumerate(members[:3]):
                print(f"  {i+1}. {pub['title']}")
            if len(members) > 3:
                print(f"  ... and {len(members) - 3} more")
        
        # Save JSON results
        json_output_path = PATH.replace('_data.json', '_communities_external_auto.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(communities_auto, f, indent=2, ensure_ascii=False)
        print(f"\nJSON results saved to: {json_output_path}")
        
        # Export to CSV for manual review
        csv_output_path = PATH.replace('_data.json', '_communities_external_auto.csv')
        Neo4jImportData.export_communities_to_csv(communities_auto, csv_output_path)
        print(f"CSV exported to: {csv_output_path}")

   
    imp.close()
