import os

import numpy as np
import pandas as pd

from utilities import report, Navigator, read_dbf, fields


class NHDTable(object):
    def __init__(self, path):
        self.matrix = pd.DataFrame(**np.load(path))
        self.sever_divergences()
        self.attribution()

    def sever_divergences(self):
        """ Remove rows in the condensed NHD table which signify a connection between a reach and a divergence.
        Retains only a single record for a given comid with the downstream divergence info for main divergence"""

        # Add the divergence and streamcalc of downstream reaches to each row

        downstream = self.matrix[['comid', 'divergence', 'streamcalc', 'fcode']]
        downstream.columns = ['tocomid'] + [f + "_ds" for f in downstream.columns.values[1:]]
        downstream = self.matrix[['comid', 'tocomid']].drop_duplicates().merge(
            downstream.drop_duplicates(), how='left', on='tocomid')

        # Where there is a divergence, select downstream reach with the highest streamcalc or lowest divergence
        downstream = downstream.sort_values('streamcalc_ds', ascending=False).sort_values('divergence_ds')
        downstream = downstream[~downstream.duplicated('comid')]

        self.matrix = self.matrix.merge(downstream, on=['comid', 'tocomid'], how='inner')

    def attribution(self):
        """ Calculate travel time, channel surface area, identify coastal reaches and reaches draining outside a region
        as outlets and sever downstream connection for outlet reaches"""
        # Replace NaN in tocomid
        self.matrix['tocomid'] = self.matrix.tocomid.fillna(-1)

        # Convert units
        self.matrix['length'] = self.matrix.pop('lengthkm') * 1000.  # km -> m
        for month in list(map(lambda x: str(x).zfill(2), range(1, 13))) + ['ma']:
            self.matrix["q_{}".format(month)] *= 2446.58  # cfs -> cmd
            self.matrix["v_{}".format(month)] *= 26334.7  # f/s -> md

        # Calculate travel time
        self.matrix["travel_time"] = self.matrix.length / self.matrix.v_ma

        # Calculate surface area
        stream_channel_a = 4.28
        stream_channel_b = 0.55
        cross_section = self.matrix.q_ma / self.matrix.v_ma
        self.matrix['surface_area'] = stream_channel_a * np.power(cross_section, stream_channel_b)

        # Indicate whether reaches are coastal
        self.matrix['coastal'] = np.int16(self.matrix.pop('fcode') == 56600)

        # Identify basin outlets
        self.matrix['outlet'] = 0
        # Identify all reaches that are a 'terminal path'. HydroSeq is used for Terminal Path ID in the NHD
        self.matrix.loc[self.matrix.hydroseq.isin(self.matrix.terminal_path), 'outlet'] = 1

        # Identify all reaches that empty into a reach outside the region
        self.matrix.loc[~self.matrix.tocomid.isin(self.matrix.comid) & (self.matrix.streamcalc > 0), 'outlet'] = 1

        # Designate coastal reaches as outlets. These don't need to be accumulated
        self.matrix.loc[self.matrix.coastal == 1, 'outlet'] = 1

        # Sever connection between outlet and downstream reaches
        self.matrix.loc[self.matrix.outlet == 1, 'tocomid'] = 0


class NavigatorBuilder(object):
    def __init__(self, nhd_table, output_path):

        report("Unpacking NHD...", 1)
        nodes, times, outlets, conversion = self.unpack_nhd(nhd_table)

        report("Tracing upstream...", 1)
        # paths, times = self.upstream_trace(nodes, outlets, times)
        paths, times = self.rapid_trace(nodes, outlets, times, conversion)

        report("Mapping paths...", 1)
        path_map = self.map_paths(paths)

        report("Collapsing array...", 1)
        paths, times, start_cols = self.collapse_array(paths, times)

        report("Write outfile...", 1)
        self.write_outfile(output_path, paths, path_map, conversion, times)

    @staticmethod
    def unpack_nhd(nhd_table):

        # Extract nodes and travel times
        nodes = nhd_table[['tocomid', 'comid']]
        times = nhd_table['travel_time'].values

        convert = pd.Series(np.arange(nhd_table.comid.size), index=nhd_table.comid.values)
        nodes = nodes.apply(lambda row: row.map(convert)).fillna(-1).astype(np.int32)

        # Extract outlets from aliased nodes
        outlets = nodes.comid[nhd_table.outlet == 1].values

        # Create a lookup key to convert aliases back to comids
        conversion_array = convert.sort_values().index.values

        # Return nodes, travel times, outlets, and conversion
        return nodes.values, times, outlets, conversion_array

    @staticmethod
    def map_paths(paths):
        # Get starting row and column for each value
        column_numbers = np.tile(np.arange(paths.shape[1]) + 1, (paths.shape[0], 1)) * (paths > 0)
        path_begins = np.argmax(column_numbers > 0, axis=1)
        max_reach = np.max(paths)
        path_map = np.zeros((max_reach + 1, 3))
        n_paths = paths.shape[0]
        for i, path in enumerate(paths):
            for j, val in enumerate(path):
                if val:
                    if i == n_paths:
                        end_row = 0
                    else:
                        next_row = (path_begins[i + 1:] <= j)
                        if next_row.any():
                            end_row = np.argmax(next_row)
                        else:
                            end_row = n_paths - i - 1
                    values = np.array([i, i + end_row + 1, j])
                    path_map[val] = values

        return path_map

    @staticmethod
    def collapse_array(paths, times):
        out_paths = []
        out_times = []
        path_starts = []
        for i, row in enumerate(paths):
            active_path = (row > 0)
            path_starts.append(np.argmax(active_path))
            out_paths.append(row[active_path])
            out_times.append(times[i][active_path])
        return np.array(out_paths), np.array(out_times), np.array(path_starts)

    @staticmethod
    def write_outfile(outfile, paths, path_map, conversion_array, times):
        np.savez_compressed(outfile, paths=paths, path_map=path_map, alias_index=conversion_array, time=times)

    @staticmethod
    def rapid_trace(nodes, outlets, times, conversion, max_length=3000, max_paths=500000):
        # Output arrays
        all_paths = np.zeros((max_paths, max_length), dtype=np.int32)
        all_times = np.zeros((max_paths, max_length), dtype=np.float32)

        # Bounds
        path_cursor = 0
        longest_path = 0

        progress = 0  # Master counter, counts how many reaches have been processed
        already = set()  # This is diagnostic - the traversal shouldn't hit the same reach more than once

        # Iterate through each outlet
        for i in np.arange(outlets.size):
            start_node = outlets[i]

            # Reset everything except the master. Trace is done separately for each outlet
            queue = np.zeros((nodes.shape[0], 2), dtype=np.int32)
            active_reach = np.zeros(max_length, dtype=np.int32)
            active_times = np.zeros(max_length, dtype=np.float32)

            # Cursors
            start_cursor = 0
            queue_cursor = 0
            active_reach_cursor = 0
            active_node = start_node

            # Traverse upstream from the outlet.
            while True:
                # Report progress
                progress += 1
                if not progress % 1000:
                    print(progress)
                upstream = nodes[nodes[:, 0] == active_node]

                # Check to make sure active node hasn't already been passed
                l1 = len(already)
                already.add(conversion[active_node])
                if len(already) == l1:
                    report("Loop at reach {}".format(conversion[active_node]))
                    exit()

                # Add the active node and time to the active path arrays
                active_reach[active_reach_cursor] = active_node
                active_times[active_reach_cursor] = times[active_node]

                # Advance the cursor and determine if a longest path has been set
                active_reach_cursor += 1
                if active_reach_cursor > longest_path:
                    longest_path = active_reach_cursor

                # If there is another reach upstream, continue to advance upstream
                if upstream.size:
                    active_node = upstream[0][1]
                    for j in range(1, upstream.shape[0]):
                        queue[queue_cursor] = upstream[j]
                        queue_cursor += 1

                # If not, write the active path arrays into the output matrices
                else:
                    all_paths[path_cursor, start_cursor:] = active_reach[start_cursor:]
                    all_times[path_cursor] = np.cumsum(active_times) * (all_paths[path_cursor] > 0)
                    queue_cursor -= 1
                    path_cursor += 1
                    last_node, active_node = queue[queue_cursor]
                    if last_node == 0 and active_node == 0:
                        break
                    for j in range(active_reach.size):
                        if active_reach[j] == last_node:
                            active_reach_cursor = j + 1
                            break
                    start_cursor = active_reach_cursor
                    active_reach[active_reach_cursor:] = 0.
                    active_times[active_reach_cursor:] = 0.

        return all_paths[:path_cursor, :longest_path], all_times[:path_cursor, :longest_path]


class LakeFileBuilder(object):
    def __init__(self, nhd_table, volume_table_path, nav, outfile_path):
        # Get a table of all lentic reaches, with the COMID of the reach and waterbody
        self.table = nhd_table[["comid", "wb_comid", "hydroseq", "q_ma"]].rename(columns={'q_ma': 'flow'})

        # Get the outlets for each reservoir
        self.identify_outlets()

        # Get residence times
        self.get_residence_times(volume_table_path)

        # Save table
        self.save_table(outfile_path)

    def identify_outlets(self):
        """ Identify the outlet reach corresponding to each reservoir """
        # Filter the reach table down to only outlet reaches by getting the minimum hydroseq for each wb_comid
        self.table = self.table.sort_values("hydroseq").groupby("wb_comid", as_index=False).first()
        self.table = self.table.rename(columns={'comid': 'outlet_comid'})
        del self.table['hydroseq']

    def get_residence_times(self, volume_path):
        # Read and reformat volume table
        volume_table = read_dbf(volume_path)[["comid", "volumecorr"]]
        volume_table = volume_table.rename(columns={"comid": "wb_comid", "volumecorr": "volume"})

        # Join reservoir table with volumes
        self.table = self.table.merge(volume_table, on="wb_comid")
        self.table['residence_time'] = self.table['volume'] / self.table.flow

    def save_table(self, outfile_path):
        self.table[["outlet_comid", "wb_comid"]] = \
            np.int32(self.table[["outlet_comid", "wb_comid"]].values)
        np.savez_compressed(outfile_path, table=self.table.values, key=self.table.columns)


class FlowFileBuilder(object):
    def __init__(self, nhd_table, out_table):
        fields.expand('monthly')
        self.table = nhd_table[fields.fetch('flow_file')]
        self.out_table = out_table
        self.save()

    def save(self):
        if not os.path.exists(os.path.dirname(self.out_table)):
            os.makedirs(os.path.dirname(self.out_table))
        np.savez_compressed(self.out_table, table=self.table.values, key=self.table.columns.tolist())


def main():
    from parameters import nhd_regions

    from paths import volume_path, condensed_nhd_path, hydro_file_path

    # Set run parameters
    build_flow_file = True
    build_navigator = True
    build_lake_file = True

    # Loop through regions
    for region in nhd_regions:
        output_dir = hydro_file_path.format(region)
        nhd_table = NHDTable(condensed_nhd_path.format(region)).matrix
        if build_flow_file:
            report("Building flow file...", 1)
            FlowFileBuilder(nhd_table, output_dir + "_flow")

        if build_navigator:
            report("Building navigator...", 1)
            NavigatorBuilder(nhd_table, output_dir + "_nav")

        if build_lake_file:
            report("Building lake file...", 1)
            nav = Navigator(region, os.path.dirname(output_dir))
            LakeFileBuilder(nhd_table, volume_path.format(region), nav, output_dir + "_lake")


main()
