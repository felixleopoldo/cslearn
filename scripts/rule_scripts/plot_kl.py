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
g = sns.boxplot(data=plot_data, x="p", y="kl_div", hue="method")

g.set(
    title=f"",
    xlabel="number of variables ($p$)",
    ylabel="KL-divergence",
)

g.legend_.set_title("")
g.legend_.set_frame_on(False)
# Labels can be saved i the CSV files
# new_labels = [
#     "constraint-based phase (PC)",
#     "Gibbs sampler and exact search",
#     "total",
# ]
# for t, l in zip(g.legend_.texts, new_labels):
#     t.set_text(l)

g.figure.tight_layout()

# output
g.figure.savefig(snakemake.output[0])
