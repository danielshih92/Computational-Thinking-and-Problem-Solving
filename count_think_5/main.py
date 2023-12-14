import matplotlib.pyplot as plt
import calendar

# Function to read the temperature data
def getTemperatureData(fileName, cities):
    dataFile = open(fileName, 'r')
    TotalYearTemperature = []
    for i in range(cities):
        city = dataFile.readline()
        cityTotalYearTemperature = [city[:-2]]  # filter ":" and '\n'
        for year in range(10):
            cityYearTemperature = []
            line = dataFile.readline()[:-1]  # filter '\n'
            yearTemperature = line.split(',')
            tempData = [int(yearTemperature[0])]  # get the exact year
            for term in yearTemperature[1:]:  # yearTemperature[1:]-> (only temperature)
                tempData.append(float(term))
                # print(tempData)
            cityYearTemperature = tempData  # the city temperature for 1 year
            cityTotalYearTemperature.append(cityYearTemperature)  # the city temperature for 10 years
        TotalYearTemperature.append(cityTotalYearTemperature)  # including three cities
        print(cityYearTemperature)
        print(cityTotalYearTemperature)
        print(TotalYearTemperature)

    dataFile.close()
    return TotalYearTemperature

# Reading the data
txt_file_path = 'TemperatureofThreecities.txt'
temperature_data = getTemperatureData(txt_file_path, 3)

# Extracting Tainan's temperature data
tainan_data = temperature_data[0][1:]  # Exclude the city name from the data
years = [year_data[0] for year_data in tainan_data]  # Extract the years
monthly_data = [year_data[1:] for year_data in tainan_data]  # Extract the monthly temperatures
# Extracting Taipei's temperature data
taipei_data = temperature_data[1][1:]  # Exclude the city name from the data
monthly_data1 = [year_data[1:] for year_data in taipei_data]  # Extract the monthly temperatures
# Extracting Kaoshiung's temperature data
kaoshiung_data = temperature_data[2][1:]  # Exclude the city name from the data
monthly_data2 = [year_data[1:] for year_data in kaoshiung_data]  # Extract the monthly temperatures
# ********************************************** Tainan *****************************************************
# Plot 1: Tainan Monthly Mean Temperature From 2013 To 2022
plt.figure(figsize=(16, 8))
for i, year in enumerate(years):
    plt.plot(range(1, 13), monthly_data[i], label=str(year))
plt.title('Tainan Monthly Mean Temperature From 2013 To 2022', fontsize=25)
plt.xlabel('Month', fontsize=20)
plt.ylabel('Temperature in Degree C',fontsize=20)
plt.xticks(range(1, 13))
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(title='Month', loc='upper right')
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plt.grid(True)
plt.savefig('figure1.png')
plt.show()

# Calculate the mean temperature for each month of the 10 years
mean_monthly_temps = [sum(month) / len(month) for month in zip(*monthly_data)]
# Calculate the overall mean temperature for all months of the 10 years
overall_mean = sum(mean_monthly_temps) / len(mean_monthly_temps)

# Plot 2: Tainan Monthly Mean Temperature Of 2013 To 2022
mean_monthly_temps = [sum(month) / len(month) for month in zip(*monthly_data)]
last_year_data = tainan_data[-1][1:]  # Data for the year 2022
plt.figure(figsize=(16, 8))
plt.plot(range(1, 13), mean_monthly_temps, 'b-')
plt.plot(range(1, 13), mean_monthly_temps, marker='o', color='r', linestyle='')

for i, temp in enumerate(mean_monthly_temps):
    plt.annotate(f'{temp:.2f}', xy=(i+1, temp), textcoords="offset points", xytext=(0,10), ha='center')
plt.axhline(y=overall_mean, color='r', linestyle='--', label='Mean of 10 Years')
plt.text(1, overall_mean, f'{overall_mean:.2f}', ha='left', va='bottom')

plt.title('Tainan Monthly Mean Temperature Of 2013 To 2022',fontsize=25)
plt.xlabel('Month',fontsize=20)
plt.ylabel('Temperature in Degree C',fontsize=20)
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True)
plt.savefig('figure2.png')
plt.show()

# Plot 3: Tainan Mean Temperature of Month and Total Year Mean Of 2013 To 2022
monthly_means = [sum(month) / len(month) for month in zip(*monthly_data)]  # Average for each month over the years
yearly_means = [sum(year[1:]) / len(year[1:]) for year in tainan_data]  # Average for each year
plt.figure(figsize=(16, 8))
for i in range(12):
    monthly_temps = [year[i+1] for year in tainan_data]
    plt.plot(years, monthly_temps, label=f'Mean of {calendar.month_abbr[i+1]}')
plt.axhline(y=overall_mean, color='r', linestyle='--', label='Mean of the Years')
plt.text(2013, overall_mean, f'{overall_mean:.2f}', ha='left', va='bottom')
plt.title('Tainan Mean Temperature of Month and Total Year Mean Of 2013 To 2022',fontsize=23)
plt.xlabel('Years',fontsize=20)
plt.ylabel('Temperature in Degree C',fontsize=20)
plt.xticks(years)
# plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(title='Month', loc='upper right')
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plt.grid(True)
plt.savefig('figure3.png')
plt.show()
# ********************************************** Taipei *****************************************************
# Plot 1: Taipei Monthly Mean Temperature From 2013 To 2022
plt.figure(figsize=(16, 8))
for i, year in enumerate(years):
    plt.plot(range(1, 13), monthly_data1[i], label=str(year))
plt.title('Taipei Monthly Mean Temperature From 2013 To 2022', fontsize=25)
plt.xlabel('Month', fontsize=20)
plt.ylabel('Temperature in Degree C',fontsize=20)
plt.xticks(range(1, 13))
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(title='Month', loc='upper right')
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plt.grid(True)
plt.savefig('figure4.png')
plt.show()

# Calculate the mean temperature for each month of the 10 years
mean_monthly_temps = [sum(month) / len(month) for month in zip(*monthly_data1)]
# Calculate the overall mean temperature for all months of the 10 years
overall_mean = sum(mean_monthly_temps) / len(mean_monthly_temps)

# Plot 2: Taipei Monthly Mean Temperature Of 2013 To 2022
mean_monthly_temps = [sum(month) / len(month) for month in zip(*monthly_data1)]
last_year_data = taipei_data[-1][1:]  # Data for the year 2022
plt.figure(figsize=(16, 8))
plt.plot(range(1, 13), mean_monthly_temps, 'b-')
plt.plot(range(1, 13), mean_monthly_temps, marker='o', color='r', linestyle='')

for i, temp in enumerate(mean_monthly_temps):
    plt.annotate(f'{temp:.2f}', xy=(i+1, temp), textcoords="offset points", xytext=(0,10), ha='center')
plt.axhline(y=overall_mean, color='r', linestyle='--', label='Mean of 10 Years')
plt.text(1, overall_mean, f'{overall_mean:.2f}', ha='left', va='bottom')

plt.title('Taipei Monthly Mean Temperature Of 2013 To 2022',fontsize=25)
plt.xlabel('Month',fontsize=20)
plt.ylabel('Temperature in Degree C',fontsize=20)
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True)
plt.savefig('figure5.png')
plt.show()

# Plot 3: Taipei Mean Temperature of Month and Total Year Mean Of 2013 To 2022
monthly_means = [sum(month) / len(month) for month in zip(*monthly_data1)]  # Average for each month over the years
yearly_means = [sum(year[1:]) / len(year[1:]) for year in taipei_data]  # Average for each year
plt.figure(figsize=(16, 8))
for i in range(12):
    monthly_temps = [year[i+1] for year in taipei_data]
    plt.plot(years, monthly_temps, label=f'Mean of {calendar.month_abbr[i+1]}')
plt.axhline(y=overall_mean, color='r', linestyle='--', label='Mean of the Years')
plt.text(2013, overall_mean, f'{overall_mean:.2f}', ha='left', va='bottom')
plt.title('Taipei Mean Temperature of Month and Total Year Mean Of 2013 To 2022',fontsize=23)
plt.xlabel('Years',fontsize=20)
plt.ylabel('Temperature in Degree C',fontsize=20)
plt.xticks(years)
# plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(title='Month', loc='upper right')
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plt.grid(True)
plt.savefig('figure6.png')
plt.show()
# ********************************************** Kaoshiung *****************************************************
# Plot 1: Kaohsiung Monthly Mean Temperature From 2013 To 2022
plt.figure(figsize=(16, 8))
for i, year in enumerate(years):
    plt.plot(range(1, 13), monthly_data2[i], label=str(year))
plt.title('Kaohsiung Monthly Mean Temperature From 2013 To 2022', fontsize=25)
plt.xlabel('Month', fontsize=20)
plt.ylabel('Temperature in Degree C',fontsize=20)
plt.xticks(range(1, 13))
plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(title='Month', loc='upper right')
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plt.grid(True)
plt.savefig('figure7.png')
plt.show()

# Calculate the mean temperature for each month of the 10 years
mean_monthly_temps = [sum(month) / len(month) for month in zip(*monthly_data2)]
# Calculate the overall mean temperature for all months of the 10 years
overall_mean = sum(mean_monthly_temps) / len(mean_monthly_temps)

# Plot 2: Kaohsiung Monthly Mean Temperature Of 2013 To 2022
mean_monthly_temps = [sum(month) / len(month) for month in zip(*monthly_data2)]
last_year_data = kaoshiung_data[-1][1:]  # Data for the year 2022
plt.figure(figsize=(16, 8))
plt.plot(range(1, 13), mean_monthly_temps, 'b-')
plt.plot(range(1, 13), mean_monthly_temps, marker='o', color='r', linestyle='')

for i, temp in enumerate(mean_monthly_temps):
    plt.annotate(f'{temp:.2f}', xy=(i+1, temp), textcoords="offset points", xytext=(0,10), ha='center')
plt.axhline(y=overall_mean, color='r', linestyle='--', label='Mean of 10 Years')
plt.text(1, overall_mean, f'{overall_mean:.2f}', ha='left', va='bottom')

plt.title('Kaohsiung Monthly Mean Temperature Of 2013 To 2022',fontsize=25)
plt.xlabel('Month',fontsize=20)
plt.ylabel('Temperature in Degree C',fontsize=20)
plt.xticks(range(1, 13))
plt.legend()
plt.grid(True)
plt.savefig('figure8.png')
plt.show()

# Plot 3: Kaohsiung Mean Temperature of Month and Total Year Mean Of 2013 To 2022
monthly_means = [sum(month) / len(month) for month in zip(*monthly_data2)]  # Average for each month over the years
yearly_means = [sum(year[1:]) / len(year[1:]) for year in kaoshiung_data]  # Average for each year
plt.figure(figsize=(16, 8))
for i in range(12):
    monthly_temps = [year[i+1] for year in kaoshiung_data]
    plt.plot(years, monthly_temps, label=f'Mean of {calendar.month_abbr[i+1]}')
plt.axhline(y=overall_mean, color='r', linestyle='--', label='Mean of the Years')
plt.text(2013, overall_mean, f'{overall_mean:.2f}', ha='left', va='bottom')
plt.title('Kaohsiung Mean Temperature of Month and Total Year Mean Of 2013 To 2022',fontsize=23)
plt.xlabel('Years',fontsize=20)
plt.ylabel('Temperature in Degree C',fontsize=20)
plt.xticks(years)
# plt.legend(title='Month', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.legend(title='Month', loc='upper right')
plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
plt.grid(True)
plt.savefig('figure9.png')
plt.show()








