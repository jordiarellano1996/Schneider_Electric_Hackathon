import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(df, col_name, label):
    sns.set(style="darkgrid")
    sns.histplot(data=df, x=col_name, color="skyblue", label=col_name+"_"+label, kde=True)
    plt.legend()
    plt.show()


def correlation_table(df):
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), cbar=True, annot=True, cmap='Blues')
    plt.show()
