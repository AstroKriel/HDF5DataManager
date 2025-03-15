## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import os
import h5py


## ###############################################################
## FUNCTIONS
## ###############################################################
def deleteEmptyGroupsHDF5(filepath_file):
  ## helper function
  def findEmptyGroups(group):
    for _, item in group.items():
      if isinstance(item, h5py.Group):
        if len(item) == 0:
          groups_to_delete.append(item.name)
        else: findEmptyGroups(item)
  ## do stuff
  with h5py.File(filepath_file, "a") as hdf:
    groups_to_delete = []
    findEmptyGroups(hdf)
    for group_name in groups_to_delete:
      del hdf[group_name]
  repackHDF5(filepath_file)

def repackHDF5(filepath_file):
  ## helper function
  def _recursiveCopy(source, destination):
    for name, item in source.items():
      if isinstance(item, h5py.Group):
        new_group = destination.create_group(name)
        _recursiveCopy(item, new_group)
      elif isinstance(item, h5py.Dataset):
        destination.create_dataset(name, data=item[()])
  ## do stuff
  _filepath_file = filepath_file + "_temp"
  with h5py.File(filepath_file, "r") as old_file:
    with h5py.File(_filepath_file, "w") as new_file:
      _recursiveCopy(old_file, new_file)
  os.remove(filepath_file)
  os.rename(_filepath_file, filepath_file)
  print("Repacked:", filepath_file)


## END OF MODULE