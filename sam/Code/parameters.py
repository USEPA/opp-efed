import pandas as pd
import numpy as np
from collections import OrderedDict

from old.paths import types_path


class ParameterSet(object):
    def __init__(self, entries):
        self.__dict__.update(entries)


stage_one_chunksize = 10000
batch_size = 500

""" These parameters can be adjusted during test runs """
output_params = {
    # Turn output datasets on and off
    "write_contributions": False,
    "write_exceedances": True,
    "write_time_series": True,
}

""" Parameters below are hardwired model parameters """

# Parameters related directly to pesticide degradation
# These inputs have defaults because pesticide-specific data are generally not available
# Future versions of SAM could accommodate alternate inputs, if available
plant_params = {
    "deg_foliar": 0.0,  # per day; assumes stability on foliage.
    "washoff_coeff": 0.1,  # Washoff coefficient; default value
}

# Parameters related to soils in the field
# Sources: PRZM5 Revision A Documentation (Young and Fry, 2016); PWC User Manual (Young, 2016)
# New version of PWC uses non-uniform runoff extraction extending below 2 cm - future update to SAM
# New version of PWC has alternate approach to PRBEN based on pesticide Kd - future update to SAM
# deplallw and leachfrac apply when irrigation triggered
soil_params = {
    "cm_2": 0.75,  # Soil distribution, top 2 cm. Revised for 1 compartment - uniform extraction
    "runoff_effic": 0.266,  # Runoff efficiency, assuming uniform 2-cm layer, from PRZM User's Manual (PUM)
    "prben": 0.5,  # PRBEN factor - default PRZM5, MMF
    "erosion_effic": 0.266,  # Erosion effic. - subject to change, MMF, frac. of eroded soil interacting w/ pesticide
    "soil_depth": 0.1,  # soil depth in cm - subject to change, MMF; lowest depth erosion interacts w/ soil (PUM)
    "increments_1": 1,  # number of increments in top 2-cm layer: 1 COMPARTMENT, UNIFORM EXTRACTION
    "increments_2": 20,  # number of increments in 2nd 100-cm layer (not used in extraction)
    "surface_dx": 0.2,
    "layer_dx": 0.5,
    "cn_min": 0.001  # curve number to use if unavailable or <0
}

# Time of Travel defaults
hydrology_params = {
    "flow_interpolation": 'quadratic',  # Must be None, 'linear', 'quadratic', or 'cubic'
    "gamma_convolve": False,
    "convolve_runoff": False,
    "minimum_residence_time": 1.5  # Minimum residence time in days for a reservoir to be treated as a reservoir
}

# Water Column Parameters - USEPA OPP defaults used in PWC, from VVWM documentation
# Sources: PRZM5 Revision A Documentation (Young and Fry, 2016); PWC User Manual (Young, 2016)
# corrections based on PWC/VVWM defaults
water_column_params = {
    "dfac": 1.19,  # default photolysis parameter from VVWM
    "sused": 30,  # water column suspended solid conc (mg/L); corrected to PWC/VVWM default
    "chloro": 0.005,  # water column chlorophyll conc (mg/L); corrected to PWC/VVWM default
    "froc": 0.04,  # water column organic carbon fraction on susp solids; corrected to PWC/VVWM default
    "doc": 5,  # water column dissolved organic carbon content (mg/L)
    "plmas": 0.4  # water column biomass conc (mg/L); corrected to PWC/VVWM default
}

# Benthic Parameters - USEPA OPP defaults from EXAMS/VVWM used in PWC
# Sources: PRZM5 Revision A Documentation (Young and Fry, 2016); PWC User Manual (Young, 2016)
# corrections based on PWC/VVWM defaults
benthic_params = {
    "depth": 0.05,  # benthic depth (m)
    "porosity": 0.50,  # benthic porosity (fraction); corrected to PWC/VVWM default
    "bulk_density": 1.35,  # bulk density, dry solid mass/total vol (g/cm3); corrected to PWC/VVWM default
    "froc": 0.04,  # benthic organic carbon fraction; corrected to PWC/VVWM default
    "doc": 5,  # benthic dissolved organic carbon content (mg/L)
    "bnmas": 0.006,  # benthic biomass intensity (g/m2); corrected to PWC/VVWM default
    "d_over_dx": 1e-8  # mass transfer coeff. for exch. betw. benthic, water column (m/s); corrected to PWC/VVWM default
}

# Create parameter sets
plant = ParameterSet(plant_params)
soil = ParameterSet(soil_params)
hydrology = ParameterSet(hydrology_params)
water_column = ParameterSet(water_column_params)
benthic = ParameterSet(benthic_params)

sfac = 0.274
soil.n_increments = soil.increments_1 + soil.increments_2
soil.delta_x = np.array([soil.surface_dx] + [soil.layer_dx] * (soil.n_increments - 1))

"""
Values are from Table F1 of TR-55 (tr_55.csv), interpolated values are included to make arrays same size
type column is rainfall parameter (corresponds to IREG in PRZM5 manual) found in met_data.csv
rainfall is based on Figure 3.3 from PRZM5 manual (Young and Fry, 2016), digitized and joined with weather grid ID
Map source: Appendix B in USDA (1986). Urban Hydrology for Small Watersheds, USDA TR-55.
Used in the model to calculate time of concentration of peak flow for use in erosion estimation.
met_data.csv comes from Table 4.1 in the PRZM5 Manual (Young and Fry, 2016)
"""
types = pd.read_csv(types_path).set_index('type')

# Turn application matrix into a numerical array for faster array processing
# JCH - column orders to fields_and_qc.csv
input_indices = {'event': ['plant', 'harvest', 'emergence', 'bloom', 'maturity'],
                 'dist': ['ground', 'foliar'],
                 'method': ['uniform', 'step'],
                 'application': ['crop', 'event', 'offset', 'dist', 'method', 'window1', 'pct1', 'window2', 'pct2',
                                 'effic', 'rate']}

heading_lookup = {'runoff_conc': 'RunoffConc(ug/L)',
                  'local_runoff': 'LocalRunoff(m3)',
                  'total_runoff': 'TotalRunoff(m3)',
                  'local_mass': 'LocalMass(m3)',
                  'total_flow': 'TotalFlow(m3)',
                  'baseflow': 'Baseflow(m3)',
                  'total_conc': 'TotalConc(ug/L)',
                  'total_mass': 'TotalMass(kg)',
                  'wc_conc': 'WC_Conc(ug/L)',
                  'erosion': 'Erosion(kg)',
                  'erosion_mass': 'ErodedMass(kg)',
                  'runoff_mass': 'RunoffMass(kg)',
                  'benthic_conc': 'BenthicConc(ug/L)'}

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

# All states
nhd_regions = sorted(states_nhd.keys())
states = sorted(set().union(*states_nhd.values()))
