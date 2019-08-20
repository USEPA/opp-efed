import numpy as np
from collections import OrderedDict

# PWC scenario selection parameters
pwc_selection_field = 'cdl_alias'
pwc_selection_pct = 25
pwc_min_selection = 1000

# Chunk size for reading scenarios
chunk_size = 100000

# Raster cell size
cell_size = 30

# Hydrologic soil groups
hydro_soil_groups = ['A', 'A/D', 'B', 'B/D', 'C', 'C/D', 'D']
hsg_cultivated = ['A', 'A', 'B', 'B', 'C', 'C', 'D']
hsg_non_cultivated = ['A', 'D', 'B', 'D', 'C', 'D', 'D']

# Curve number to use when curve number is missing
null_curve_number = 99

# USLEP (practices) values for aggregation based on Table 4 in SAM Scenario Input Parameter documentation.
# Original source: Table 5.6 in PRZM 3 Manual (Carousel et al, 2015).
# USLEP values for cultivated crops by slope bin (0-2, 2-5, 5-10, 10-15, 15-25, >25)
uslep_values = [0.6, 0.5, 0.5, 0.6, 0.8, 0.9]

# Soil depth bins;
depth_bins = np.array([5, 20, 50, 100])

# Aggegation bins for soil map units (see Addendum E of SAM Scenario Input Parameter Documentation)
aggregation_bins = {'slope': [0, 2, 5, 10, 15, 25, 200],
                    'orgC_5': [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 12, 20, 100],
                    'sand_5': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                    'clay_5': [0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100]}

# Maximum thickness of 'O' horizons to be removed
o_horizon_max = 5.  # cm
slope_length_max = 300.
slope_min = 1


# m values for calculation of USLE LS
usle_m_bins = np.array([0, 1, 3, 3.5, 4.5, 5, 100])
usle_m_vals = np.array([0.2, 0.3, 0.35, 0.4, 0.45, 0.5])

erom_months = [str(x).zfill(2) for x in range(1, 13)] + ['ma']

# The number of soil horizons available in each soil table
# Set to the soil with the most horizons in 2016 SSURGO (in Kansas)
max_horizons = 8

# NHD regions and the states that overlap
states_nhd = OrderedDict((('01', {"ME", "NH", "VT", "MA", "CT", "RI", "NY"}),
                          ('02', {"VT", "NY", "PA", "NJ", "MD", "DE", "WV", "DC", "VA"}),
                          ('03N', {"VA", "NC", "SC", "GA"}),
                          ('03S', {"FL", "GA"}),
                          ('03W', {"FL", "GA", "TN", "AL", "MS"}),
                          ('04', {"WI", "MN", "MI", "IL", "IN", "OH", "PA", "NY"}),
                          ('05', {"IL", "IN", "OH", "PA", "WV", "VA", "KY", "TN"}),
                          ('06', {"VA", "KY", "TN", "NC", "GA", "AL", "MS"}),
                          ('07', {"MN", "WI", "SD", "IA", "IL", "MO", "IN"}),
                          ('08', {"MO", "KY", "TN", "AR", "MS", "LA"}),
                          ('09', {"ND", "MN", "SD"}),
                          ('10U', {"MT", "ND", "WY", "SD", "MN", "NE", "IA"}),
                          ('10L', {"CO", "WY", "MN", "NE", "IA", "KS", "MO"}),
                          ('11', {"CO", "KS", "MO", "NM", "TX", "OK", "AR", "LA"}),
                          ('12', {"NM", "TX", "LA"}),
                          ('13', {"CO", "NM", "TX"}),
                          ('14', {"WY", "UT", "CO", "AZ", "NM"}),
                          ('15', {"NV", "UT", "AZ", "NM", "CA"}),
                          ('16', {"CA", "OR", "ID", "WY", "NV", "UT"}),
                          ('17', {"WA", "ID", "MT", "OR", "WY", "UT", "NV"}),
                          ('18', {"OR", "NV", "CA"})))

vpus_nhd = {'01': 'NE', '02': 'MA', '03N': 'SA', '03S': 'SA', '03W': 'SA', '04': 'GL', '05': 'MS',
            '06': 'MS', '07': 'MS', '08': 'MS', '09': 'SR', '10L': 'MS', '10U': 'MS', '11': 'MS',
            '12': 'TX', '13': 'RG', '14': 'CO', '15': 'CO', '16': 'GB', '17': 'PN', '18': 'CA'}

nhd_regions = sorted(states_nhd.keys())
states = sorted(set().union(*states_nhd.values()))
