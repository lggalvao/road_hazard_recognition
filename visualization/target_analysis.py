import matplotlib.pyplot as plt

def plot_hazard_balance(df):
    df['hazard_flag'].value_counts().plot(kind='bar')
    plt.title("Hazard vs Non-Hazard")
    plt.ylabel("Count")
    plt.show()
