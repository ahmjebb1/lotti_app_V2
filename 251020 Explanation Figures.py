## Objectives

#Long-tailed tit
    # Make a figure that explains facets of the dataset including:
        # sex distribution
        # year distribution
        # helper distribution
        # distribution across individuals
        # how distribution across individuals changes from raw, mid, to high quality.

############################# Data Set-up ################################

# Imports
    # data organisation
import pandas as pd
import os

    # plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Set directory path
base_dir = "C:/Users/bi1ahj/Desktop/Data Analysis/GitHub/lotti_app"

# Read in .csv files
noisedf = pd.read_csv(os.path.join(base_dir, "data_filtering_classification.csv"))
birddf = pd.read_excel(os.path.join(base_dir, "Data_Quantity_Lotti.xlsx"), sheet_name = 0)

# Check correct reading
print(noisedf.head())
print(birddf.head())

# Merge the two dataframes
rawdf = pd.merge(noisedf, birddf, on = "ID", how = "left")

# Check dataset structure
print(rawdf.dtypes)

# Correct column structure
rawdf = rawdf.astype({"database_ID": object, "year": object})

# Subset data for mid and high datasets
middf = rawdf[
    (rawdf["class_attrition"].notna()) &
    (rawdf["class_noise"].notna()) &
    (rawdf["class_attrition"] != "high") &
    (rawdf["class_noise"] != "high")
]

highdf = middf[
    (middf["artefact"].notna()) &
    (middf["artefact"] != "Y")
]

# Check how subsetting altered the datasets
print(rawdf.shape)
print(middf.shape)
print(highdf.shape)

#List of colours
#bda34e yellow (subtract noise and attrition - mid)
#6988a0 light blue (all data - raw)
#158f88 teal green (subtract artefacts - high)


############################# Raw Plotting the Data ################################

# Count rows per unique ID
raw_counts = rawdf["ID"].value_counts().reset_index()
raw_counts.columns = ["ID", "count"]
raw_counts["dataset"] = "All"

# Sort counts
raw_counts = raw_counts.sort_values("ID")

# Plot
plt.figure(figsize = (12, 10))

# Plot raw data bars with conditional colors
sns.barplot(data = raw_counts, x = "ID", y = "count", color = "#6988a0", alpha = 1.0, width = 0.6)

# Remove axis titles
plt.xlabel("")
plt.ylabel("")

# Specify axis tick labels
plt.xticks(rotation = 90, ha = 'right', fontsize = 25, color = "#e2ddd5")
plt.yticks(fontsize = 25, color = "#e2ddd5")

# Remove top and right spines (plot outline), keep bottom and left axes and recolour
sns.despine(top = True, right = True)

ax = plt.gca()  # get the current axes
ax.spines['bottom'].set_color("#e2ddd5")
ax.spines['left'].set_color("#e2ddd5")

# Tight layout to prevent clipping
plt.tight_layout()

# Save plot with transparent background
output_path = os.path.join(base_dir, "ID_counts.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print(f"✅ Plot saved to: {output_path}")

############################# Raw and Mid Plots of the Data ################################

# Count rows per unique ID in each dataset
middf_counts = middf["ID"].value_counts().reset_index()
middf_counts.columns = ["ID", "count"]
middf_counts["dataset"] = "Filtered"

# Sort counts
middf_counts = middf_counts.sort_values("ID")

# Check dataset
print(raw_counts)

# Check dataset
print(middf_counts)

# Define colors for filtered data
raw_colors = ["#A03A45" if val < 50 else "#6988a0" for val in raw_counts["count"]]
mid_colors = ["#A03A45" if val < 50 else "#158f88" for val in middf_counts["count"]]

plt.figure(figsize = (12, 10))

# Plot raw data bars with conditional colors
sns.barplot(data = raw_counts, x = "ID", y = "count", palette = raw_colors, alpha = 0.5, width = 0.6)

# Plot filtered data bars with conditional colors
sns.barplot(data = middf_counts, x ="ID", y ="count", palette = mid_colors, alpha = 1.0, width = 0.6)

# Remove axis titles
plt.xlabel("")
plt.ylabel("")

# Specify axis tick labels
plt.xticks(rotation = 90, ha = 'right', fontsize = 25, color = "#e2ddd5")
plt.yticks([])

# Remove top and right spines (plot outline), keep bottom and left axes and recolour
sns.despine(top = True, right = True, left = True)

ax = plt.gca()  # get the current axes
ax.spines['bottom'].set_color("#e2ddd5")

# Tight layout to prevent clipping
plt.tight_layout()

# Save plot
output_path = os.path.join(base_dir, "ID_counts_overlay_rawmid.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print(f"✅ Plot saved to: {output_path}")

############################# Mid and High Plots of the Data ################################

# Count rows per unique ID in each dataset
highdf_counts = highdf["ID"].value_counts().reset_index()
highdf_counts.columns = ["ID", "count"]
highdf_counts["dataset"] = "Filtered"

# Sort counts
highdf_counts = highdf_counts.sort_values("ID")

# Check dataset
print(highdf_counts)

# Define colors based on count for raw data
mid_colors = ["#A03A45" if val < 50 else "#158f88" for val in middf_counts["count"]]

# Define colors for filtered data
high_colors = ["#A03A45" if val < 50 else "#bda34e" for val in highdf_counts["count"]]

plt.figure(figsize = (12, 10))

# Plot raw data bars with conditional colors
sns.barplot(data = middf_counts, x = "ID", y = "count", palette = mid_colors, alpha = 0.5, width = 0.6)

# Plot filtered data bars with conditional colors
sns.barplot(data = highdf_counts, x="ID", y="count", palette = high_colors, alpha = 1.0, width = 0.6)

# Remove axis titles
plt.xlabel("")
plt.ylabel("")

# Specify axis tick labels
plt.xticks(rotation = 90, ha = 'right', fontsize = 25, color = "#e2ddd5")
plt.yticks([])

# Remove top and right spines (plot outline), keep bottom and left axes and recolour
sns.despine(top = True, right = True, left = True)

ax = plt.gca()  # get the current axes
ax.spines['bottom'].set_color("#e2ddd5")

# Tight layout to prevent clipping
plt.tight_layout()

# Save plot
output_path = os.path.join(base_dir, "ID_counts_overlay_midhigh.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)
plt.close()

print(f"✅ Plot saved to: {output_path}")

############################# Counting ID data ################################

#count by sex and year
sex_year_counts = birddf.groupby(["year", "sex", "recorded_as_helper"]).size().reset_index(name = "count")
print(sex_year_counts)


