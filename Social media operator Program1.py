import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "mobile_usage_behavioral_analysis.csv"
df = pd.read_csv(r"C:\Users\HP\Desktop\python project\Project 2\mobile_usage_behavioral_analysis.csv");

# Convert 'hour' column to integer
df['hour'] = df['hour'].astype(int)

# 1. Peak Usage Hours
plt.figure(figsize=(10,5))
sns.histplot(df['hour'], bins=24, kde=True)
plt.title('User Activity by Hour of the Day')
plt.xlabel('Hour of the Day')
plt.ylabel('Number of Users')
plt.show()

# 2. Engagement Patterns by Age and Gender
plt.figure(figsize=(12,5))
sns.boxplot(x='Age', y='Daily_Screen_Time_Hours', hue='Gender', data=df)
plt.title('Daily Screen Time by Age and Gender')
plt.xlabel('Age')
plt.ylabel('Daily Screen Time (Hours)')
plt.legend()
plt.show()

# 3. App Usage Trends (Correlation between App Usage and Gaming Time)
plt.figure(figsize=(8,5))
sns.scatterplot(x='Total_App_Usage_Hours', y='Gaming_App_Usage_Hours', hue='Gender', data=df)
plt.title('Correlation between Total App Usage and Gaming')
plt.xlabel('Total App Usage (Hours)')
plt.ylabel('Gaming App Usage (Hours)')
plt.show()

# 4. Weekly Engagement Trends
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['day_of_week'] = pd.Categorical(df['day_of_week'], categories=day_order, ordered=True)
weekly_usage = df.groupby('day_of_week')['Daily_Screen_Time_Hours'].mean()

plt.figure(figsize=(10,5))
weekly_usage.plot(kind='bar', color='skyblue')
plt.title('Average Daily Screen Time by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Screen Time (Hours)')
plt.show()

# 5. Location-based Engagement
plt.figure(figsize=(12,5))
sns.boxplot(x='Location', y='Daily_Screen_Time_Hours', data=df)
plt.xticks(rotation=45)
plt.title('Daily Screen Time Across Locations')
plt.xlabel('Location')
plt.ylabel('Daily Screen Time (Hours)')
plt.show()

print("User behavior analysis complete.")
