import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset("penguins")
sns.jointplot(data=df, x="flipper_length_mm", y="bill_length_mm", hue="species")
plt.show()
