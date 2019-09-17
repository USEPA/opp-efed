import os
import math
import numpy as np
import pandas as pd

from collections import Iterable
from tempfile import mkstemp

from old.paths import fields_and_qc_path, endpoint_format_path


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


class FieldMatrix(object):
    def __init__(self, path=None):
        self.path = path
        self.matrix = pd.read_csv(self.path)

    def data_type(self, fields=None, old_fields=False):
        if fields is None:
            data_types = self.matrix.data_type
        else:
            index_col = "internal_name" if not old_fields else "external_name"
            data_types = self.matrix.set_index(index_col).loc[fields].data_type.values
        return np.array(list(map(eval, data_types)))

    def fetch_field(self, item, field):
        for column in 'data_source', 'source_table':
            if item in self.matrix[column].values:
                return self.matrix[self.matrix[column] == item][field].tolist()
        if item in self.matrix.columns:
            return self.matrix[self.matrix[item] == 1][field].tolist()
        else:
            report("Unrecognized sub-table '{}'".format(item), warn=1)

    def fetch_old(self, item):
        return self.fetch_field(item, 'external_name')

    def fetch(self, item):
        return self.fetch_field(item, 'internal_name')

    @property
    def fill_value(self):
        return self.matrix.set_index('internal_name').fill_value.dropna()


class HydroTable(pd.DataFrame):
    def __init__(self, region, path, table_type):
        super().__init__()
        self.region = region
        self.path = path.format(self.region, table_type)

        assert table_type in ("lake", "flow"), "Provided table type \"{}\" not in ('lake', 'flow')".format(table_type)
        data, header = self.read_table()

        super(HydroTable, self).__init__(data=data, columns=header)

        index_col = 'wb_comid' if 'wb_comid' in self.columns else 'comid'
        self.set_index(index_col, inplace=True)
        self.index = np.int32(self.index)

    def read_table(self):
        assert os.path.isfile(self.path), "Table file {} not found".format(self.path)
        data = np.load(self.path)
        return data['table'], data['key']

    def fetch(self, feature_id):
        return self.loc[feature_id]


class ImpulseResponseMatrix(MemoryMatrix):
    """ A matrix designed to hold the results of an impulse response function for 50 day offsets """

    def __init__(self, n_dates, size=50):
        self.n_dates = n_dates
        self.size = size
        super(ImpulseResponseMatrix, self).__init__([size, n_dates], name='impulse response')
        for i in range(size):
            irf = self.generate(i, 1, self.n_dates)
            self.update(i, irf)

    def fetch(self, index):
        if index <= self.size:
            irf = super(ImpulseResponseMatrix, self).fetch(index, verbose=False)
        else:
            irf = self.generate(index, 1, self.n_dates)
        return irf

    @staticmethod
    def generate(alpha, beta, length):
        def gamma_distribution(t, a, b):
            a, b = map(float, (a, b))
            tau = a * b
            return ((t ** (a - 1)) / (((tau / a) ** a) * math.gamma(a))) * math.exp(-(a / tau) * t)

        return np.array([gamma_distribution(i, alpha, beta) for i in range(length)])


class Navigator(object):
    def __init__(self, region_id, upstream_path):
        self.file = upstream_path.format(region_id, 'nav')
        self.paths, self.times, self.map, self.alias_to_reach, self.reach_to_alias = self.load()
        self.reach_ids = set(self.reach_to_alias.keys())

    def load(self):
        assert os.path.isfile(self.file), "Upstream file {} not found".format(self.file)
        data = np.load(self.file, mmap_mode='r')
        conversion_array = data['alias_index']
        reverse_conversion = dict(zip(conversion_array, np.arange(conversion_array.size)))
        return data['paths'], data['time'], data['path_map'], conversion_array, reverse_conversion

    def upstream_watershed(self, reach_id, mode='reach', return_times=False, return_warning=False, verbose=False):

        def unpack(array):
            first_row = [array[start_row][start_col:]]
            remaining_rows = list(array[start_row + 1:end_row])
            return np.concatenate(first_row + remaining_rows)

        # Look up reach ID and fetch address from pstream object
        reach = reach_id if mode == 'alias' else self.reach_to_alias.get(reach_id)
        reaches, adjusted_times, warning = np.array([]), np.array([]), None
        try:
            start_row, end_row, col = map(int, self.map[reach])
            start_col = list(self.paths[start_row]).index(reach)
        except TypeError:
            warning = "Reach {} not found in region".format(reach)
        except ValueError:
            warning = "{} not in upstream lookup".format(reach)
        else:
            # Fetch upstream reaches and times
            aliases = unpack(self.paths)
            reaches = aliases if mode == 'alias' else np.int32(self.alias_to_reach[aliases])

        # Determine which output to deliver
        output = [reaches]
        if return_times:
            times = unpack(self.times)
            adjusted_times = np.int32(times - self.times[start_row][start_col])
            output.append(adjusted_times)
        if return_warning:
            output.append(warning)
        if verbose and warning is not None:
            report(warning, warn=1)
        return output[0] if len(output) == 1 else output


def report(message, tabs=0, warn=0):
    tabs = "\t" * tabs
    prefix = {0: "", 1: "Warning: ", 2: "Failure: ", 3: "Debug: "}[warn]
    print(tabs + prefix + message)


def read_gdb(dbf_file, table_name, input_fields=None):
    """Reads the contents of a dbf table """
    import ogr

    # Initialize file
    driver = ogr.GetDriverByName("OpenFileGDB")
    gdb = driver.Open(dbf_file)

    # parsing layers by index
    tables = {gdb.GetLayerByIndex(i).GetName(): i for i in range(gdb.GetLayerCount())}
    table = gdb.GetLayer(tables[table_name])
    table_def = table.GetLayerDefn()
    table_fields = [table_def.GetFieldDefn(i).GetName() for i in range(table_def.GetFieldCount())]
    if input_fields is None:
        input_fields = table_fields
    else:
        missing_fields = set(input_fields) - set(table_fields)
        if any(missing_fields):
            report("Fields {} not found in table {}".format(", ".join(missing_fields), table_name), warn=1)
            input_fields = [field for field in input_fields if field not in missing_fields]
    data = np.array([[row.GetField(f) for f in input_fields] for row in table])

    return pd.DataFrame(data=data, columns=input_fields)


def read_dbf(dbf_file, fields='all'):
    from dbfread import DBF, FieldParser

    class MyFieldParser(FieldParser):
        def parse(self, field, data):
            try:
                return FieldParser.parse(self, field, data)
            except ValueError as e:
                print(e)
                return None

    try:
        dbf = DBF(dbf_file)
        table = pd.DataFrame(iter(dbf))
    except ValueError:
        dbf = DBF(dbf_file, parserclass=MyFieldParser)
        table = pd.DataFrame(iter(dbf))

    table.rename(columns={column: column.lower() for column in table.columns}, inplace=True)

    return table


# Initialize field matrix
fields = FieldMatrix(fields_and_qc_path)

# Initialize endpoints
endpoint_format = pd.read_csv(endpoint_format_path)
