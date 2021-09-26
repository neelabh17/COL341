import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
plt.plot([10,100,1000,10000, 90000], [439,112,90,73,72])
# plt.legend()
plt.xlabel("Batch Size")
plt.ylabel("Time")
plt.tight_layout()
plt.savefig("bs_time.jpg")