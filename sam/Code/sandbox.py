import arcpy

flowline_path = r"J:\opp-efed\aquatic-model-inputs\bin\Input\NHDPlusV21\NHDPlusMS\NHDPlus07\NHDSnapshot\Hydrography\NHDFlowline.shp"
missing_path = r'C:\users\jhook\desktop\missing_uns.csv'

arcpy.MakeFeatureLayer_management(flowline_path, "flowlines")
arcpy.MakeTableView_management(missing_path, "missing")

arcpy.AddJoin_management("flowlines", "COMID", "missing", "comid", "KEEP_COMMON")
arcpy.CopyFeatures_management("flowlines", r"C:\Users\Jhook\Desktop\bogeys.shp")