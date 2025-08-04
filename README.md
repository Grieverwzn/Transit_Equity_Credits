# Transit_Equity_Credits
 Transit_Equity_Credits models
Teclite is an open-sourcelightweight, and fast Python engine for transit equity credits optimization model for subsidization. 
Input files: 
zone.csv 
•	zone_id	  int 
•	pop float (population) 
•	pop_ratio_1: float ( the population ratio of group 1) 
•	pop_ratio_2: float ( the population ratio of group 2)
•	pop_ratio_3: float ( the population ratio of group 3)
•	agency_id: int 

demand.csv
•	demand_id	int
•	from_zone_id	int 
•	to_zone_id	int 
•	travel_time_auto float	
•	travel_time_transit float
•	travel_cost_auto float
•	travel_cost_transit float
•	trips	float
•	agency_name int 

setting.csv
[pop_group] 
•	Group id int 
•	Group name string 
•	annual_income float (optional)
•	hourly_salary float 
•	value_of_time float 

[agency]
•	agency_id int 
•	agency_name string
Output files: 
 

