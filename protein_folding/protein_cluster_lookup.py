import collections

class ProteinClusterLookup(object):

  def __init__(self, clusters_by_entity_file_handler):
    self.entity_to_cluster_map = dict()
    cluster = 0
    for r in clusters_by_entity_file_handler:
      entities = r.split()
      for e in entities:
        self.entity_to_cluster_map[e] = cluster
      cluster += 1

  def ProteinCluster(self, pdb_id, entity_id):
    return self.entity_to_cluster_map.get('{}_{}'.format(pdb_id, entity_id), -1)
