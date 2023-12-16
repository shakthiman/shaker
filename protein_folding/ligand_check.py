import collections
import logging

class LigandCheck(object):

  def __init__(self, lig_pairs_file_handler):
    self.pdbs_with_ligands = set()
    for r in lig_pairs_file_handler:
      parts = r.strip().split(':')
      if len(parts) == 2:
        self.pdbs_with_ligands.add(parts[0].strip())
      else:
        raise Exception('A pdb with no ligands')

  def HasLigands(self, pdb_id):
    return pdb_id in self.pdbs_with_ligands
