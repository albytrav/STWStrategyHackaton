# Tutorial on how to interact with WORC using SimpleWORC

# Import neccesary packages
from WORC import SimpleWORC
import os

# These packages are only used in analysing the results
import pandas as pd
import json
import fastr
import glob


# Define the folder this script is in, so we can easily find the example data
script_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_path, 'Data')

# Tutorial on how to interact with WORC using SimpleWORC

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

# File in which your features are stored
feature_xlsx = os.path.join(data_path, 'Features.xlsx')

# File in which the labels (i.e. outcome you want to predict) is stated
# Again, change this accordingly if you use your own data.
label_file = os.path.join(data_path, 'pinfo_2y.csv')

# Name of the label you want to predict
label_name = '2ysurv'

# Determine whether we want to do a coarse quick experiment, or a full lengthy
# one. Again, change this accordingly if you use your own data.
coarse = True

# Give your experiment a name
experiment_name = 'Example_Lung'

# Instead of the default tempdir, let's but the temporary output in a subfolder
# in the same folder as this script
tmpdir = os.path.join(script_path, 'WORC_' + experiment_name)

# Override some fields in the config
config_overrides = {
    "Imputation" : {  # Since we have missing features, we are gonna use imputation
        "use" : 'True'
    },
    "Featsel":{  # With >4000 features and nfeat >> npat, feature selection seems neccesary.
                # Also saves a lot of time: vs 1.5 hr in classify
        "StatisticalTestUse": "False",  # You can switch on a statistic threshold feature selection method if you want with changing this
            # but other options are also possible. See
            # https://worc.readthedocs.io/en/latest/static/configuration.html#featsel
        "StatisticalTestMetric": "MannWhitneyU",
        "StatisticalTestThreshold": "-2.0, 1.0"
    }
}

# ---------------------------------------------------------------------------
# The actual experiment
# ---------------------------------------------------------------------------

# Create a Simple WORC object
experiment = SimpleWORC(experiment_name)

# Set the input data according to the variables we defined earlier
experiment.features_from_radiomix_xlsx(feature_xlsx)

experiment.labels_from_this_file(label_file)
experiment.predict_labels([label_name])

# Use the standard workflow for binary classification
experiment.binary_classification(coarse=coarse)

# Add our manual config overrides
experiment.add_config_overrides(config_overrides)

# Set the temporary directory
experiment.set_tmpdir(tmpdir)

# Run the experiment!
experiment.execute()

# ---------------------------------------------------------------------------
# Analysis of results
# ---------------------------------------------------------------------------

# There are two main outputs: the features for each patient/object, and the overall
# performance. These are stored as .hdf5 and .json files, respectively. By
# default, they are saved in the so-called "fastr output mount", in a subfolder
# named after your experiment name.

# Locate output folder
outputfolder = fastr.config.mounts['output']
experiment_folder = os.path.join(outputfolder, 'WORC_' + experiment_name)

print(f"Your output is stored in {experiment_folder}.")

# Read the overall peformance
performance_file = os.path.join(experiment_folder, 'performance_all_0.json')
with open(performance_file, 'r') as fp:
    performance = json.load(fp)

# Print the output performance
print("\n Performance:")
stats = performance['Statistics']
del stats['Percentages']  # Omitted for brevity
for k, v in stats.items():
    print(f"\t {k} {v}.")

# NOTE: the performance is probably horrible, which is expected as we ran
# the experiment on coarse settings. These settings are recommended to only
# use for testing: see also below.

# ---------------------------------------------------------------------------
# Tips and Tricks
# ---------------------------------------------------------------------------

# For tips and tricks on running a full experiment instead of this simple
# example, adding more evaluation options, debuggin a crashed network etcetera,
# please go to https://worc.readthedocs.io/en/latest/static/user_manual.html

# Some things we would advice to always do:
#   - Run actual experiments on the full settings (coarse=False):
#       coarse = False
#       experiment.binary_classification(coarse=coarse)
#   Note: this will result in more computation time. We therefore recommmend
#   to run this script on either a cluster or high performance PC. If so,
#   you may change the execution to use multiple cores to speed up computation
#   just before before experiment.execute():
#       experiment.set_multicore_execution()
#
#   - Add extensive evaluation: experiment.add_evaluation() before experiment.execute():
#       experiment.add_evaluation()
