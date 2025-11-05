# Aoristic_Analysis
Preform Aoristic Analysis on any data set with a data time field
This tool is a simple way to preform an Aoristic Analysis to identify when is the most likely day of week and time of day an incident is likely to occur. This tool allows you to import an excel file with a data/time field and preform a weighted analysis. the output will include an expanded excel file and a aoristic time of day/day of week chart.


<img width="4324" height="2214" alt="heatmap" src="https://github.com/user-attachments/assets/ad9428ca-bdd5-472c-91fc-7f85cb3133eb" />

Aoristic analysis is a temporal analysis method that uses a weighting system to estimate the most likely time of occurrence of events. By weighting each hour within the possible time span of an incident proportionally, the analysis produces an estimated distribution of when incidents are most likely to have occurred.

Example:
An incident occurs overnight with no known time of occurrence, but the possible time window is between 00:00 and 03:00 — a three-hour period.
The total probability of occurrence is 1, distributed evenly across the three hours.
Thus, the probability for each hour is 1 ÷ 3 = 0.33.
Each of these hours is therefore assigned a weight of 0.33.

If an incident has a known exact time of occurrence, that hour is assigned a weight of 1.

With enough incidents, the individual hourly weights can be summed to develop a cumulative probability distribution showing the most likely times of occurrence.

This tool autmoates this process.

Follow these steps for simple install. (Python requiered)

Create a venv

      python -m venv venv

Activate the venv

      venv\Scripts\activate

Install dependencies (individually or with requirements.txt)

      pip install -r requirements.txt 

   Or
      
      pip install pandas
      pip install numpy
      pip install matplotlib
      pip install seaborn
      pip install openpyxl

Run the Python script

      aoristic_analysis2.0_with_weighted_analysis.py
