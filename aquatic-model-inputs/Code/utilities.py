import datetime as dt
import os
import re

import numpy as np
import pandas as pd

from paths import fields_and_qc_path

""" Functions and classes utilized by multiple scripts """


class FieldMatrix(object):
    def __init__(self):
        self.path = fields_and_qc_path
        self.refresh()
        self.extended_monthly = False
        self.depth_weighted = False
        self.horizons_expanded = False
        self._qc_table = None
        self._convert = None

    def data_type(self, fetch=None, old_fields=False):
        col_type = "internal" if not old_fields else "external"
        if fetch:
            matrix = self.fetch(fetch, col_type, False)
        else:
            matrix = self.matrix
        data_types = matrix.set_index(col_type + "_name").data_type.to_dict()
        return {key: eval(val) for key, val in data_types.items()}

    def expand(self, mode='depth', n_horizons=None):

        from parameters import depth_bins, erom_months, max_horizons
        if n_horizons is not None:
            max_horizons = n_horizons
        try:
            condition, select_field, numbers = \
                {'depth': ('depth_weighted', 'depth_weight', depth_bins),
                 'horizon': ('horizons_expanded', 'horizontal', range(1, max_horizons + 1)),
                 'monthly': ('extended_monthly', 'monthly', erom_months)}[mode]

        except KeyError as e:
            message = "Invalid expansion mode '{}' specified: must be in ('depth', 'horizon', 'monthly')".format(
                mode)
            raise Exception(message) from e

        # Test to make sure it hasn't already been done
        if not getattr(self, condition):

            # Find each row that applies, duplicate, and append to the matrix
            self.matrix[condition] = 0
            burn = self.matrix[condition].copy()
            new_rows = []
            for idx, row in self.matrix[self.matrix[select_field] == 1].iterrows():
                burn.iloc[idx] = 1
                for i in numbers:
                    new_row = row.copy()
                    new_row['internal_name'] = row.internal_name + "_" + str(i)
                    new_row[condition] = 1
                    new_rows.append(new_row)
            new_rows = pd.concat(new_rows, axis=1).T
            # Filter out the old rows and add new ones
            self.matrix = pd.concat([self.matrix[~(burn == 1)], new_rows], axis=0)

            # Record that the duplication has occurred
            setattr(self, condition, True)

    def fetch_field(self, item, field, names_only=True):
        def extract_num(field_name):
            match = re.search("(\d{1,2})$", field_name)
            if match:
                return float(match.group(1)) / 100.
            else:
                return 0.

        out_matrix = None
        for column in 'data_source', 'source_table':
            if item in self.matrix[column].values:
                out_matrix = self.matrix[self.matrix[column] == item]
                break
        if out_matrix is None:
            if item in self.matrix.columns:
                out_matrix = self.matrix[self.matrix[item] > 0]
                if out_matrix[item].max() > 1:  # field order is given
                    out_matrix.loc[:, 'order'] = out_matrix[item] + np.array(
                        [extract_num(f) for f in out_matrix[field]])
                    out_matrix = out_matrix.sort_values('order')
        if out_matrix is None:
            report("Unrecognized sub-table '{}'".format(item))
        if names_only:
            return out_matrix[field].tolist()
        else:
            return out_matrix

    def fetch(self, item, how='internal', names_only=True):
        return self.fetch_field(item, '{}_name'.format(how), names_only)

    @property
    def convert(self):
        if self._convert is None:
            self._convert = {row.external_name: row.internal_name for _, row in self.matrix.iterrows()}
        return self._convert

    @property
    def qc_table(self):
        if self._qc_table is None:
            qc_fields = ['range_min', 'range_max', 'range_flag',
                         'general_min', 'general_max', 'general_flag',
                         'blank_flag', 'fill_value']
            self._qc_table = self.matrix.set_index('internal_name')[qc_fields] \
                .apply(pd.to_numeric, downcast='integer') \
                .dropna(subset=qc_fields, how='all')
        return self._qc_table

    def perform_qc(self, other, outfile=None, verbose=False):

        # Confine QC table to fields in other table
        missing_fields = set(other.columns.values) - set(self.qc_table.index.values)
        if verbose and any(missing_fields):
            report("Field(s) {} not found in QC table".format(", ".join(map(str, missing_fields))))
        active_fields = [field for field in self.qc_table.index.values if field in other.columns.tolist()]
        qc_table = self.qc_table.loc[active_fields]
        other = other[qc_table.index.values]

        # Flag missing data
        # Note - if this fails, check for fields with no flag or fill attributes
        flags = pd.isnull(other).astype(np.int32)
        flags = flags.mask(flags > 0, qc_table.blank_flag, axis=1)

        # Flag out-of-range data
        for test in ('general', 'range'):
            ranges = qc_table[[test + "_min", test + "_max", test + "_flag"]].dropna()
            for param, (param_min, param_max, flag) in ranges.iterrows():
                try:
                    out_of_range = (other[param] < param_min) | (other[param] > param_max)
                    if out_of_range.any():
                        flags.loc[out_of_range, param] = \
                            np.maximum(flags.loc[out_of_range, param].values, flag).astype(np.int8)
                except TypeError as e:
                    report("Can't evaluate validity of field '{}': {}".format(param, e))

        # Write QC file
        if outfile is not None:
            if not os.path.isdir(os.path.dirname(outfile)):
                os.makedirs(os.path.dirname(outfile))
            flags.to_csv(outfile)

        return flags

    @property
    def fill_value(self):
        return self.matrix.set_index('internal_name').fill_value.dropna()

    def refresh(self):
        # Read the fields/QC matrix
        if self.path is not None:
            self.matrix = pd.read_csv(self.path)
        elif self.matrix is not None:
            self.matrix = self.matrix

        self.erom_expanded = self.depth_weighted = self.horizons_expanded = False


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
            report(warning)
        return output[0] if len(output) == 1 else output


class WeatherCube(object):
    def __init__(self, weather_path, region):
        self.path = weather_path.format(region)
        self.storage_path = os.path.join(self.path, 'weather_cube.dat')
        self.points_path = os.path.join(self.path, 'weather_grid.csv')
        self.key_path = os.path.join(self.path, "key.csv")

        self._points = None
        self._dates = None

        self.years, self.columns, self.points = self.load_key()

    def fetch(self, point_num):
        array = np.memmap(self.storage_path, mode='r', dtype=np.float32, shape=self.shape)
        index = self.get_index(point_num)
        if index is not None:
            out_array = array[self.get_index(point_num)]
            del array
            return pd.DataFrame(data=out_array.T, columns=self.columns, index=self.dates)
        else:
            del array

    def write(self, point, data):
        array = np.memmap(self.storage_path, mode='r+', dtype=np.float32, shape=self.shape)
        index = self.get_index(point)
        if index is not None:
            array[index] = data
        del array

    def get_index(self, point):
        try:
            return np.where(self.points.index == point)[0][0]
        except IndexError:
            report("Point {} not found in array".format(point))
            return None

    def load_key(self):
        with open(self.key_path) as f:
            years = list(map(int, next(f).split(",")))
            cols = next(f).split(",")
        points = pd.read_csv(self.points_path, index_col=[0])
        return years, cols, points

    @property
    def dates(self):
        if self._dates is None:
            self._dates = pd.date_range(self.start_date, self.end_date)
        return self._dates

    @property
    def start_date(self):
        return dt.date(self.years[0], 1, 1)

    @property
    def end_date(self):
        return dt.date(self.years[-1], 12, 31)

    @property
    def shape(self):
        return (self.points.shape[0], len(self.columns), (self.end_date - self.start_date).days + 1)


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
            report("Fields {} not found in table {}".format(", ".join(missing_fields), table_name))
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
                report(e)
                return None

    try:
        dbf = DBF(dbf_file)
        table = pd.DataFrame(iter(dbf))
    except ValueError:
        dbf = DBF(dbf_file, parserclass=MyFieldParser)
        table = pd.DataFrame(iter(dbf))

    table.rename(columns={column: column.lower() for column in table.columns}, inplace=True)

    return table


def report(message, tabs=0):
    tabs = "\t" * tabs
    print(tabs + message)


# Initialize field matrix
fields = FieldMatrix()
