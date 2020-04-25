import os
import re
import pandas as pd
import numpy as np
from collections import Iterable
from tempfile import mkstemp


class MemoryMatrix(object):
    """ A wrapper for NumPy 'memmap' functionality which allows the storage and recall of arrays from disk """

    def __init__(self, dimensions, dtype=np.float32, path=None, existing=False, name='null', index_dim=0,
                 verbose=False):
        self.dtype = dtype
        self.path = path
        self.existing = existing
        self.name = name

        # Initialize dimensions of array
        self.dimensions = tuple(np.array(d) if isinstance(d, Iterable) else d for d in dimensions)
        self.labels = tuple(d if isinstance(d, Iterable) else None for d in self.dimensions)
        self.shape = tuple(map(int, (d.size if isinstance(d, Iterable) else d for d in self.dimensions)))
        self.index = self.labels[index_dim]

        # Add an alias if it's called for
        if self.index is not None:
            self.lookup = {label: i for i, label in enumerate(self.index)}
            self.aliased = True
        else:
            self.lookup = None
            self.aliased = False

        self.initialize_array(verbose)

    def alias_to_index(self, aliases, verbose=False):
        found = None
        if type(aliases) in (pd.Series, list, set, np.array, np.ndarray):
            indices = np.array([self.lookup.get(alias, np.nan) for alias in aliases])
            found = np.where(~np.isnan(indices))[0]
            indices = indices[found]
            if verbose and found.size != len(aliases):
                missing = np.array(aliases)[~found]
                report("Missing {} of {} needed indices from array {}".format(missing.size, len(aliases), self.name))
        else:
            indices = self.lookup.get(aliases)
            if indices is None and verbose:
                report("Alias {} not found in array {}".format(aliases, self.name), warn=2)
                return None, None
        return np.int32(indices), found

    def fetch(self, index, copy=False, verbose=False, return_found=False, iloc=False, pop=False):

        # Initialize reader (with write capability in case of 'pop')
        array = self.reader

        # Convert aliased item(s) to indices if applicable
        if self.aliased and not iloc:
            index, found = self.alias_to_index(index, verbose)
            if index is None:
                return

        # Extract data from array
        output = array[index]

        # Return a copy of the selection instead of a view if necessary
        if pop or copy:
            output = np.array(output)
            if pop:  # Set the selected rows to zero after extracting array
                array[index] = 0.

        del array
        if return_found:
            return output, found
        else:
            return output

    def initialize_array(self, verbose=False):

        # Load from saved file if one is specified, else generate
        if self.path is None:
            self.existing = False
            self.path = mkstemp(suffix=".dat", dir=os.path.join("..", "bin", "temp"))[1]
        else:
            # Add suffix to path if one wasn't provided
            if not self.path.endswith("dat"):
                self.path += ".dat"
            if os.path.exists(self.path):
                self.existing = True

        if not self.existing:
            if verbose:
                report("Creating memory map {}...".format(self.path))
            try:
                os.makedirs(os.path.dirname(self.path))
            except FileExistsError:
                pass
            np.memmap(self.path, dtype=self.dtype, mode='w+', shape=self.shape)  # Allocate memory

    def update(self, index, values, return_found=False, verbose=False, iloc=False):
        array = self.writer
        if self.aliased and not iloc:
            index = self.alias_to_index(index, verbose)
        array[index] = values
        del array

    @property
    def reader(self):
        return np.memmap(self.path, dtype=self.dtype, mode='r+', shape=self.shape)

    @property
    def writer(self):
        mode = 'r+' if os.path.isfile(self.path) else 'w+'
        return np.memmap(self.path, dtype=self.dtype, mode=mode, shape=self.shape)


class DateManager(object):
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def date_offset(self, start, end, coerce=True, return_msg=False):
        messages = []
        start_offset, end_offset = 0, 0
        if self.start_date > start:  # scenarios start later than selected start date
            messages.append('start date is earlier')
        else:
            start_offset = (start - self.start_date).astype(int)
        if self.end_date < end:
            messages.append('end date is later')
        else:
            end_offset = (end - self.end_date).astype(int)

        if coerce:
            self.start_date = self.start_date + np.timedelta64(int(start_offset), 'D')
            self.end_date = self.end_date + np.timedelta64(int(end_offset), 'D')

        if return_msg:
            return start_offset, end_offset, messages
        else:
            return start_offset, end_offset

    @property
    def dates(self):
        return pd.date_range(self.start_date, self.end_date)

    @property
    def dates_julian(self):
        return (self.dates - self.dates[0]).days.astype(int)

    @property
    def mid_month(self):
        return self.months[:-1] + (self.months[1:] - self.months[:-1]) / 2

    @property
    def mid_month_julian(self):
        return np.int32((self.mid_month - self.dates[0]).days)

    @property
    def months(self):
        return pd.date_range(self.dates.min(), self.dates.max() + np.timedelta64(1, 'M'), freq='MS')

    @property
    def months_julian(self):
        return pd.date_range(self.dates.min(), self.dates.max() + np.timedelta64(1, 'M'), freq='MS').month

    @property
    def new_year(self):
        return np.int32([(np.datetime64("{}-01-01".format(year)) - self.start_date).astype(int)
                         for year in np.unique(self.dates.year)])

    @property
    def n_dates(self):
        return self.dates.size

    @property
    def month_index(self):
        return self.dates.month

    @property
    def year_index(self):
        return np.int16(self.dates.year - self.dates.year[0])

    @property
    def year_length(self):
        return np.unique(self.dates, return_counts=True)[1]


class FieldManager(object):
    """
    The Field Manager loads the table fields_and_qc.csv
    and uses that table to manage fields. Field management functions include: field name
    conversions from raw input data sources to internal field names, extending fields
    that are expansible (such as those linked to soil horizon or month), and performing
    QAQC by comparing the values in a table with the specified QC ranges in fields_and_qc.py
    This class is inherited by other classes which wrap pandas DataFrames.
    """

    def __init__(self, path, name_col='internal_name'):
        """ Initialize a FieldManager object. """
        self.path = path
        self.name_col = name_col
        self.matrix = None
        self.extended = []
        self.refresh()
        self._convert = None

        self.qc_fields = ['range_min', 'range_max', 'range_flag',
                          'general_min', 'general_max', 'general_flag',
                          'blank_flag', 'fill_value']

    def data_type(self, fetch=None, cols=None):
        """
        Give dtypes for fields in table
        :param fetch: Fetch a subset of keys, e.g. 'monthly' (str)
        :param how: 'internal' or 'external'
        :param cols: Only return certain columns (iter of str)
        :return: Dictionary with keys as field names and dtypes as values
        """
        if fetch:
            matrix = self.fetch(fetch, self.name_col, False)
        else:
            matrix = self.matrix
        data_types = matrix.set_index(self.name_col).data_type.to_dict()

        if cols is not None:
            data_types = {key: val for key, val in data_types.items() if key in cols}
        return {key: eval(val) for key, val in data_types.items()}

    def expand(self, select_field, numbers):
        """
        Certain fields are repeated during processing - for example, the streamflow (q) field becomes monthly
        flow (q_1, q_2...q_12), and soil parameters linked to soil horizon will have multiple values for a single
        scenario (e.g., sand_1, sand_2, sand_3). This function adds these extended fields to the FieldManager.
        :param mode: 'depth', 'horizon', or 'monthly'
        :param n_horizons: Optional parameter when specifying a number to expand to (int)
        """
        if type(numbers) == int:
            numbers = np.arange(numbers) + 1

        # Check to make sure it's only been extended once
        if not select_field in self.extended:
            condition = select_field + '_extended'
            # Find each row that applies, duplicate, and append to the matrix
            self.matrix[condition] = 0
            burn = self.matrix[condition].copy()
            new_rows = []
            for idx, row in self.matrix[self.matrix[select_field] == 1].iterrows():
                burn.iloc[idx] = 1
                for i in numbers:
                    new_row = row.copy()
                    new_row[self.name_col] = row[self.name_col] + "_" + str(i)
                    new_row[condition] = 1
                    new_rows.append(new_row)
            new_rows = pd.concat(new_rows, axis=1).T

            # Filter out the old rows and add new ones
            self.matrix = pd.concat([self.matrix[~(burn == 1)], new_rows], axis=0)

            # Record that the duplication has occurred
            self.extended.append(select_field)

    def fetch(self, source, names_only=True, dtypes=False, col=None):
        """
        Subset the FieldManager matrix (fields_and_qc.csv) based on the values in a given column, or a field group in
        the 'data_source' or 'source_table' columns. For example, fetch_field('sam_scenario', 'internal_name')
        would return all the fields with a value > 0 in the 'sam_scenario' column of fields_and_qc.csv, and
        fetch_field('SSURGO') would return all the fields with 'SSURGO' in the 'data_source' column'.
        If the numbers are ordered, the returned list of fields will be in the same order. The names_only parameter
        can be turned off to return all other fields (e.g., QAQC fields) from fields_and_qc.csv for the same subset.
        :param source: The column in fields_and_qc.csv used to make the selection (str)
        :param field: The field values to return, usually 'internal_name' or 'external_name'
        :param names_only: Return simply the selected field, or all fields (bool)
        :return: Subset of the field matrix (df)
        """

        def extract_num(field_name):
            match = re.search("(\d{1,2})$", field_name)
            if match:
                return float(match.group(1)) / 100.
            else:
                return 0.

        # Check to see if provided 'col' value is a group in the
        # 'data_source' or 'source_table' columns in fields_and_qc.csv
        col = self.name_col if col is None else col

        out_fields = None
        source_cols = sorted((c for c in self.matrix.columns if c.startswith("source_")))
        for column in source_cols:
            if source in self.matrix[column].values:
                out_fields = self.matrix[self.matrix[column] == source]
                break

        # If 'col' not found, check to see if provided 'col' value corresponds to a column in fields_and_qc.csv
        if out_fields is None:
            if source in self.matrix.columns:
                out_fields = self.matrix[self.matrix[source] > 0]
                if out_fields[source].max() > 1:  # field order is given
                    out_fields.loc[:, 'order'] = out_fields[source] + np.array(
                        [extract_num(f) for f in out_fields[col]])
                    out_fields = out_fields.sort_values('order')
        if out_fields is None:
            report("Unrecognized sub-table '{}'".format(source))
            return None
        if names_only:
            out_fields = out_fields[col].tolist()
        if dtypes:
            return out_fields, self.data_type(cols=out_fields)
        else:
            return out_fields

    @property
    def convert(self, from_col='external_name', to_col='internal_name'):
        """ Dictionary that can be used to convert 'external' variable names to 'internal' names """
        if self._convert is None:
            self._convert = {row[from_col]: row[to_col] for _, row in self.matrix.iterrows()}
        return self._convert

    def qc_table(self):
        """ Initializes an empty QAQC table with the QAQC fields from fields_and_qc_csv. """

        return self.matrix.set_index(self.name_col)[self.qc_fields] \
            .apply(pd.to_numeric, downcast='integer') \
            .dropna(subset=self.qc_fields, how='all')

    def perform_qc(self, other):
        """
        Check the value of all parameters in table against the prescribed QAQC ranges in fields_and_qc.csv.
        There are 3 checks performed: (1) missing data, (2) out-of-range data, and (3) 'general' ranges.
        The result of the check is a copy of the data table with the data replaced with flags. The flag values are
        set in fields_and_qc.csv - generally, a 1 is a warning and a 2 is considered invalid. The outfile parameter
        gives the option of writing the resulting table to a csv file if a path is provided.
        :param other: The table upon which to perform the QAQC check (df)
        :param outfile: Path to output QAQC file (str)
        :return: QAQC table (df)
        """
        # Confine QC table to fields in other table
        active_fields = {field for field in self.qc_table().index.values if field in other.columns.tolist()}
        qc_table = self.qc_table().loc[active_fields]

        # Flag missing data
        # Note - if this fails, check for fields with no flag or fill attributes
        # This can also raise an error if there are duplicate field names in fields_and_qc with qc parametersz
        flags = pd.isnull(other).astype(np.int8)
        duplicates = qc_table.index[qc_table.index.duplicated()]
        if not duplicates.empty:
            raise ValueError(f"Multiple QC ranges specified for {', '.join(duplicates.values)} in fields table")
        flags = flags.mask(flags > 0, qc_table.blank_flag, axis=1)

        # Flag out-of-range data
        for test in ('general', 'range'):
            ranges = qc_table[[test + "_min", test + "_max", test + "_flag"]].dropna()
            for param, (param_min, param_max, flag) in ranges.iterrows():
                if flag > 0:
                    out_of_range = ~other[param].between(param_min, param_max) * flag
                    flags[param] = np.maximum(flags[param], out_of_range).astype(np.int8)
        qc_table = pd.DataFrame(np.zeros(other.shape, dtype=np.int8), columns=other.columns)
        qc_table[flags.columns] = flags

        return qc_table

    def fill_value(self):
        """ Return the fill values for flagged data set in fields_and_qc.csv """
        return self.matrix.set_index(self.name_col).fill_value.dropna()

    def refresh(self):
        """ Reload fields_and_qc.csv, undoes 'extend' and other modifications """
        # Read the fields/QC matrix
        if self.path is not None:
            self.matrix = pd.read_csv(self.path)
        self.extended = []


def report(message, tabs=0):
    """ Display a message with a specified indentation """
    tabs = "\t" * tabs
    print(tabs + str(message))
