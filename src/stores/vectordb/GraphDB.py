import neo4j


class GraphDB:
    def __init__(self, uri, user, password):
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))

    def run_query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return result.data()    

    def search_metrics(self, domain : str, modality : str, task : str):
        query = (
                    "match (m:MODALITY)-[:TYPE]->(t:TASK)-[:WITH]->(d:DOMAIN)-[:GOT]->(met:METRICS)"
                    "WHERE t.name = $task AND m.name = $modality AND d.name = $domain "
                    "return met.name as name, met.forumla as formula, met.threshold_pass as pass_threshold, met.threshold_fail as fail_threshold, met.threshold_warn as warn_threshold"
                )
        results = self.run_query(query, {"domain": domain.lower(), "task": task.lower(), "modality": modality.lower()})
        print(f"\n[DEBUG] query:\n {results}")
        print(f"\n[DEBUG] query values:\n {{\"domain\": {domain.lower()}, \"task\": {task.lower()}, \"modality\": {modality.lower()}}}")
        print(f"\n[DEBUG] search_metrics results:\n {results}")
        return results

    def get_modalities(self):
        query = (
            "MATCH (m:MODALITY) "
            "RETURN DISTINCT m.name as name"
        )
        results = self.run_query(query)
        print(f"\n[DEBUG] get_modalities results:\n {[record["name"] for record in results]}")
        return [record["name"] for record in results]


    def get_domains(self):
        query = "MATCH (d:DOMAIN) RETURN DISTINCT d.name as name"
        results = self.run_query(query)
        print(f"\n[DEBUG] get_domains results:\n {[record["name"] for record in results]}")
        return [record["name"] for record in results]
    
    def get_tasks(self):
        query = (
            "MATCH (d:TASK) "
            "RETURN DISTINCT d.name as name"
        )
        results = self.run_query(query)
        print(f"\n[DEBUG] get_tasks results:\n {[record["name"] for record in results]}")
        return [record["name"] for record in results]
