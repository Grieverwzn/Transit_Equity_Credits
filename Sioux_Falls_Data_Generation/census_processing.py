import pandas as pd

census_df = pd.read_csv('zone_information/census_income_data.csv')
# reduce the geography column so it only show the number after US
census_df['GEOID'] = census_df['Geography'].str.split('US').str[1]
census_df['GEOID'] = census_df['GEOID'].str.lstrip('0')
census_df['state_id'] = census_df['GEOID'].str[:2]
census_df['county_id'] = census_df['GEOID'].str[0:5]
census_df = census_df[census_df['county_id'] == '46099']

# Estimate!!Households!!Mean income (dollars) to hminc
census_df['Estimate!!Households!!Mean income (dollars)'] = pd.to_numeric(
    census_df['Estimate!!Households!!Mean income (dollars)'], errors='coerce')
census_df['Estimate!!Households!!Total'] = pd.to_numeric(census_df['Estimate!!Households!!Total'], errors='coerce')
census_df['hhminc'] = census_df['Estimate!!Households!!Mean income (dollars)']
census_df['totHh'] = census_df['Estimate!!Households!!Total']
# drop
census_df.drop(['Geography', 'Estimate!!Households!!Mean income (dollars)'], axis=1, inplace=True)

census_list = []  # list of dict
# pop ratio 4
census_df['$100k_to_$149,999'] = pd.to_numeric(census_df['Estimate!!Households!!Total!!$100,000 to $149,999'],
                                               errors='coerce')
census_df['$150k_to_$199,999'] = pd.to_numeric(census_df['Estimate!!Households!!Total!!$150,000 to $199,999'],
                                               errors='coerce')
census_df['$200k_or_more'] = pd.to_numeric(census_df['Estimate!!Households!!Total!!$200,000 or more'], errors='coerce')
census_df['minc>100k'] = census_df['$100k_to_$149,999'] + census_df['$150k_to_$199,999'] + census_df['$200k_or_more']
# if the value is null, replace it with the mean value of minc>100k, estimation_flag is 1
census_df['minc>100k_estimation_flag'] = census_df['minc>100k'].apply(lambda x: 1 if pd.isnull(x) else 0)
mean_minc100k = census_df[census_df['minc>100k'].notnull()]['minc>100k'].mean()
census_df['minc>100k'] = census_df['minc>100k'].fillna(mean_minc100k)
# replace 0 with 0.0001 to avoid division by zero
census_df['minc>100k'] = census_df['minc>100k'].replace(0, 0.0001)
census_df.drop(['Estimate!!Households!!Total!!$100,000 to $149,999',
                'Estimate!!Households!!Total!!$150,000 to $199,999',
                'Estimate!!Households!!Total!!$200,000 or more'], axis=1, inplace=True)

group_id = 4
group_name = 'minc>100k'
group_annual_income = (200000 - 100000) / 2 + 100000
group_hourly_salary = group_annual_income / 2080
group_vot = 0.5 * group_hourly_salary
census_list.append(
    {'[pop_group]': '', 'group_id': group_id, 'group_name': group_name, 'annual_income': group_annual_income,
     'hourly_salary': group_hourly_salary, 'value_of_time': group_vot})
# convert dict to dataframe


# pop ratio 3
census_df['$50k_to_$74,999'] = pd.to_numeric(census_df['Estimate!!Households!!Total!!$50,000 to $74,999'],
                                             errors='coerce')
census_df['$75k_to_$99,999'] = pd.to_numeric(census_df['Estimate!!Households!!Total!!$75,000 to $99,999'],
                                             errors='coerce')
census_df['minc50k_100k'] = census_df['$50k_to_$74,999'] + census_df['$75k_to_$99,999']
# calculate the mean value of minc50k_100k excluding nan
census_df['minc50k_100k_estimation_flag'] = census_df['minc50k_100k'].apply(lambda x: 1 if pd.isnull(x) else 0)
mean_minc50k_100k = census_df[census_df['minc50k_100k'].notnull()]['minc50k_100k'].mean()
census_df['minc50k_100k'] = census_df['minc50k_100k'].fillna(mean_minc50k_100k)
census_df['minc50k_100k'] = census_df['minc50k_100k'].replace(0, 0.0001)
census_df.drop(['Estimate!!Households!!Total!!$50,000 to $74,999', 'Estimate!!Households!!Total!!$75,000 to $99,999'],
               axis=1, inplace=True)

group_id = 3
group_name = 'minc50k_100k'
group_annual_income = (100000 - 50000) / 2 + 50000
group_hourly_salary = group_annual_income / 2080
group_vot = 0.5 * group_hourly_salary
census_list.append({'[pop_group]': '', 'group_id': group_id, 'group_name': group_name,
                    'annual_income': group_annual_income, 'hourly_salary': group_hourly_salary, 'value_of_time': group_vot})

# pop ratio 2
census_df['$25k_to_$34,999'] = pd.to_numeric(census_df['Estimate!!Households!!Total!!$25,000 to $34,999'],
                                             errors='coerce')
census_df['$35k_to_$49,999'] = pd.to_numeric(census_df['Estimate!!Households!!Total!!$35,000 to $49,999'],
                                             errors='coerce')
census_df['minc25k_50k'] = census_df['$25k_to_$34,999'] + census_df['$35k_to_$49,999']
census_df['minc25k_50k_estimation_flag'] = census_df['minc25k_50k'].apply(lambda x: 1 if pd.isnull(x) else 0)
# calculate the mean value of minc25k_50k excluding nan
mean_minc25k_50k = census_df[census_df['minc25k_50k'].notnull()]['minc25k_50k'].mean()
census_df['minc25k_50k'] = census_df['minc25k_50k'].fillna(mean_minc25k_50k)
census_df['minc25k_50k'] = census_df['minc25k_50k'].replace(0, 0.0001)
census_df.drop(['Estimate!!Households!!Total!!$25,000 to $34,999', 'Estimate!!Households!!Total!!$35,000 to $49,999'],
               axis=1, inplace=True)

group_id = 2
group_name = 'minc25k_50k'
group_annual_income = (50000 - 25000) / 2 + 25000
group_hourly_salary = group_annual_income / 2080
group_vot = 0.5 * group_hourly_salary
census_list.append({'[pop_group]': '', 'group_id': group_id, 'group_name': group_name,
                    'annual_income': group_annual_income, 'hourly_salary': group_hourly_salary, 'value_of_time': group_vot})

# pop ratio 1
census_df['Less_than_$10,000'] = pd.to_numeric(census_df['Estimate!!Households!!Total!!Less than $10,000'],
                                               errors='coerce')
census_df['$10k_to_$14,999'] = pd.to_numeric(census_df['Estimate!!Households!!Total!!$10,000 to $14,999'],
                                             errors='coerce')
census_df['$15k_to_$24,999'] = pd.to_numeric(census_df['Estimate!!Households!!Total!!$15,000 to $24,999'],
                                             errors='coerce')

census_df['minc<25k'] = census_df['Less_than_$10,000'] + census_df['$10k_to_$14,999'] + census_df['$15k_to_$24,999']
census_df['minc<25k_estimation_flag'] = census_df['minc<25k'].apply(lambda x: 1 if pd.isnull(x) else 0)

mean_minc25k = census_df[census_df['minc<25k'].notnull()]['minc<25k'].mean()
census_df['minc<25k'] = census_df['minc<25k'].fillna(mean_minc25k)
census_df['minc<25k'] = census_df['minc<25k'].replace(0, 0.0001)
census_df.drop(['Estimate!!Households!!Total!!Less than $10,000',
                'Estimate!!Households!!Total!!$10,000 to $14,999',
                'Estimate!!Households!!Total!!$15,000 to $24,999'], axis=1, inplace=True)

group_id = 1
group_name = 'minc<25k'
group_annual_income = (25000 - 0) / 2 + 0
group_hourly_salary = group_annual_income / 2080
group_vot = 0.5 * group_hourly_salary
census_list.append({'[pop_group]': '', 'group_id': group_id, 'group_name': group_name,
                    'annual_income': group_annual_income, 'hourly_salary': group_hourly_salary, 'value_of_time': group_vot})

# only keep pop_ratio columns
census_df = census_df[['GEOID', 'minc<25k', 'minc25k_50k', 'minc50k_100k', 'minc>100k', 'minc<25k_estimation_flag',
                       'minc25k_50k_estimation_flag', 'minc50k_100k_estimation_flag', 'minc>100k_estimation_flag',
                       'totHh', 'hhminc']]

census_df.to_csv('zone_information/zone_income_pop_ratio.csv', index=False)
settings_df = pd.DataFrame(census_list)
settings_df.to_csv('zone_information/settings.csv', index=False)
