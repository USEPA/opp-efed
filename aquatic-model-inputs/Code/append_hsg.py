import pandas as pd
import os
import read
import modify

from parameters import nhd_regions


old_scenarios = r"C:\Users\Jhook\Desktop\PwcScenarios\r{}_1_Corn.csv"
new_scenarios = os.path.join(os.path.dirname(old_scenarios), "corn_fix", "r{}_1_Corn_fix.csv")
for region in nhd_regions:
    print(region)
    scenario_file = old_scenarios.format(region)
    new_file = new_scenarios.format(region)
    hsg, _ = modify.soils(read.soils(region))
    hsg = hsg[['soil_id', 'hydro_group']]
    scenarios = pd.read_csv(scenario_file)
    scenarios = scenarios.merge(hsg, how='left', on='soil_id')
    scenarios.to_csv(new_file, index=None)
    print(new_file)




