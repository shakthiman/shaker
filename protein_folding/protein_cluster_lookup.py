import collections

class ProteinClusterLookup(object):

  def __init__(self, clusters_by_entity_file_handler):
    self._pid_to_clusters_map = dict()
    cluster = 0
    self._max_cluster = 0
    for r in clusters_by_entity_file_handler:
      self._max_cluster = cluster
      entities = r.split()
      for e in entities:
        pid = (e.strip().rsplit('_', 1))[0].upper()
        self._pid_to_clusters_map.setdefault(pid, set()).add(cluster)
      cluster += 1

  def ProteinCluster(self, pdb_id):
    return self._pid_to_clusters_map.get(pdb_id.upper(), None)

  def GetMaxCluster(self):
    return self._max_cluster
