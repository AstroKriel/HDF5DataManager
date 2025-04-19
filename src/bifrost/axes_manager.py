## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from enum import Enum


## ###############################################################
## RELEVANT UNITS FOR YOUR WORK
## ###############################################################
class AxisUnits(Enum):
  NOT_SPECIFIED = "not_specified"
  DIMENSIONLESS = "dimensionless"
  T_TURB = "t_turb"
  K_TURB = "k_turb"


## ###############################################################
## AXIS MANAGER
## ###############################################################
class AxisObject:
  def __init__(
      self,
      group, name, values,
      units      = AxisUnits.NOT_SPECIFIED.value,
      notes      = "",
      global_ref = None,
    ):
    self.group  = group
    self.name   = name
    self.values = numpy.array(values)
    units = self.cast_units_to_string(units)
    if not isinstance(notes, str):
      raise TypeError("notes need to be a string")
    ## local properties should be private
    self._units = units
    self._notes = notes
    self._global_ref = global_ref

  @property
  def units(self):
    if self._global_ref is not None:
      return self._global_ref.units
    return self._units

  @units.setter
  def units(self, units):
    units = self.cast_units_to_string(units)
    if self._global_ref is not None:
      self._global_ref.units = units
    else: self._units = units

  @property
  def notes(self):
    if self._global_ref is not None:
      return self._global_ref.notes
    return self._notes

  @notes.setter
  def notes(self, notes):
    if not isinstance(notes, str):
      raise TypeError("notes need to be a string.")
    if self._global_ref is not None:
      self._global_ref.notes = notes
    else: self._notes = notes

  @staticmethod
  def _validate_inputs(group, name, values, units, notes):
    if not isinstance(group, str) or (group.strip() == ""): raise ValueError("Axis group must be a non-empty string.")
    if not isinstance(name, str) or (name.strip() == ""): raise ValueError("Axis name must be a non-empty string.")
    if not isinstance(values, (list, numpy.ndarray)): raise TypeError("Axis values must be a list or numpy array.")
    values = numpy.array(values)
    if values.size == 0: raise ValueError("Axis values cannot be empty.")
    if values.ndim != 1: raise ValueError("Axis values must be a 1D array.")
    if not numpy.issubdtype(values.dtype, numpy.number): raise TypeError("Axis values must be numeric (either integers or floats).")
    if numpy.any(numpy.isnan(values)): raise ValueError(f"Axis `{name}` contains NaN values.")
    if not numpy.all(numpy.diff(values) >= 0): raise ValueError("Axis values must be monotonically increasing.")
    if len(values) != len(numpy.unique(values)): raise ValueError("Axis values must be unique.")
    if not isinstance(units, AxisUnits): raise TypeError(f"Invalid axis unit: {units}. Must be an element from AxisUnits.")
    if not isinstance(notes, str): raise TypeError("Notes must be a string.")

  @staticmethod
  def cast_units_to_string(units):
    if not isinstance(units, (str, AxisUnits)): raise TypeError("Error: Provided `units` was neither a string or an element of AxisUnits.")
    if isinstance(units, AxisUnits): units = units.value
    return units

  @staticmethod
  def create_dict_inputs(
      group, name, values,
      units = AxisUnits.NOT_SPECIFIED,
      notes = "",
    ):
    AxisObject._validate_inputs(group, name, values, units, notes)
    units = AxisObject.cast_units_to_string(units)
    return {
      "group"  : group,
      "name"   : name,
      "values" : values,
      "units"  : units,
      "notes"  : notes,
    }

  def add(self, values):
    ## merge the new + unique values into the existing set
    ## note: the following is equivelant to `numpy.sort(numpy.array(list(set(self.values) | set(values))))`
    self.values = numpy.unique(numpy.concatenate((self.values, numpy.array(values))))

  def get_dict(self):
    return {
      "group"  : self.group,
      "name"   : self.name,
      "values" : self.values,
      "units"  : self.units,
      "notes"  : self.notes,
    }

  def __eq__(self, obj_axis):
    if isinstance(obj_axis, AxisObject):
      bool_properties_are_the_same = all([
        self.group == obj_axis.group,
        self.name  == obj_axis.name,
        numpy.array_equal(self.values, obj_axis.values)
      ])
      return bool_properties_are_the_same
    return False


## END OF MODULE