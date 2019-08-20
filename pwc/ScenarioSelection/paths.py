import os

# Root directories
input_dir = "Input"
output_dir = "Output"

# Input files
pwc_batch_file = os.path.join(input_dir, "Samples", "corn_60k.csv")  # the table used as input for the pwc run
pwc_summary_file = os.path.join(input_dir, "Samples", "Summary_SW_60k.txt")  # the output file from the pwc run

# Output files
output_dir = "Output"  # where output data will be written
summary_outfile = os.path.join(output_dir, "test_summary.csv")
selected_scenario_outfile = os.path.join(output_dir, "test_selection.csv")
plot_outfile = os.path.join(output_dir, "plot_{}")