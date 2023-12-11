import collections

class ProteinClusterLookup(object):

  def __init__(self, clusters_by_entity_file_handler):
    self.entity_to_cluster_map = dict()
    for r in clusters_by_entity_file_handler:
      entities = r.split()
      num_entities = len(entities)
      for e in entities:
        self.entity_to_cluster_map[e] = num_entities

  def NumEntitiesInProteinsCluster(self, pdb_id, entity_id):
    return self.entity_to_cluster_map.get('{}_{}'.format(pdb_id, entity_id), 0)
