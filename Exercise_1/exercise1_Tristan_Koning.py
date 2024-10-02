import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.loadtxt('ironman.txt')
total_rank = data[:, 0]
year_of_birth = data[:, 1]
age = 2010 - year_of_birth
total_time = data[:, 2]
swimming_time = data[:, 3]
swimming_rank = data[:, 4]
cycling_time = data[:, 5]
cycling_rank = data[:, 6]
running_time = data[:, 7]
running_rank = data[:, 8]

# (a) scatter plots
plt.figure()
plt.plot(total_time, total_rank, '.')
plt.ylabel('Total Rank')
plt.xlabel('Total Time (min)')
plt.title('total rank vs total time')
plt.legend()
plt.savefig('total_rank_vs_total_time.pdf')
plt.close()

plt.figure()
plt.plot(total_time, age, '.')
plt.ylabel('Age of Participant (years)')
plt.xlabel('Total Time (min)')
plt.title('Age of Participant vs Total Time')
plt.legend()
plt.savefig('age_vs_total_time.pdf')
plt.close()

plt.figure()
plt.plot(swimming_time, running_time, '.')
plt.ylabel('Running Time (min)')
plt.xlabel('Swimming Time (min)')
plt.title('Running Time vs Swimming Time')
plt.legend()
plt.savefig('running_time_vs_swimming_time.pdf')
plt.close()

plt.figure()
plt.plot(total_time, swimming_time, '.')
plt.ylabel('Swimming Time (min)')
plt.xlabel('Total Time (min)')
plt.title('Swimming Time vs Total Time')
plt.legend()
plt.savefig('swimming_time_vs_total_time.pdf')
plt.close()

plt.figure()
plt.plot(total_time, cycling_time, '.')
plt.ylabel('Cycling Time (min)')
plt.xlabel('Total Time (min)')
plt.title('Cycling Time vs Total Time')
plt.legend()
plt.savefig('cycling_time_vs_total_time.pdf')
plt.close()

plt.figure()
plt.plot(total_time, running_time, '.')
plt.ylabel('Running Time (min)')
plt.xlabel('Total Time (min)')
plt.title('Running Time vs Total Time')
plt.legend()
plt.savefig('running_time_vs_total_time.pdf')
plt.close()

# (b) Histograms
plt.figure()
plt.hist(total_time, range=(min(total_time), max(total_time)), bins=20)
plt.xlabel('Total Time (min)')
plt.ylabel('Frequency')
plt.title('Histogram of Total Time')
plt.legend()
plt.savefig('histogram_of_total_time.pdf')
plt.close()

plt.figure()
plt.hist(age, range=(min(age), max(age)), bins=20)
plt.xlabel('Age of Participant (years)')
plt.ylabel('Frequency')
plt.title('Histogram of Age')
plt.legend()
plt.savefig('histogram_of_age.pdf')
plt.close()
