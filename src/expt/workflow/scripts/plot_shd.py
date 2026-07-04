import pandas as pd
import seaborn as sns


# input
plot_data_path = snakemake.input[0]

# make plot
plot_data = pd.read_csv(plot_data_path)
plot_data["hue"] = plot_data["method"] + ", n=" + plot_data["n"].astype(str)
sns.set(font_scale=1.25)
sns.set_style("white")
sns.set_style({"legend.frameon": False})
g = sns.boxplot(data=plot_data, x="p", y="shd", hue="hue")

g.set(
    title="",
    xlabel="number of variables ($p$)",
    ylabel="SHD (LDAG)",
)

g.legend_.set_title("")
g.legend_.set_frame_on(False)

g.figure.tight_layout()

# output
g.figure.savefig(snakemake.output[0])
