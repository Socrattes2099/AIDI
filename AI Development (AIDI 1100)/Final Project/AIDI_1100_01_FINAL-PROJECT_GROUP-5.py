# Group #5
# Members:
# - Garaye, Amanpreet Kaur
# - Ndiweni, Cliford
# - Guevara Calderon, Bayron Jose
# - Manchanda, Robin
#
# Course code: AIDI 1100 Section 01
# Submission Date: April 9th
# Description: This program will extract stock symbols present in the news of www.prnewswire.com, 
#              then, it will query the Yahoo Finance service and plot the Close Prices and Volume
#              of the last 10 days. It will continue scanning on the background for new symbols at
#              the specified time interval.

## INSTRUCTIONS ###
# Before running the code, the following Python packages need to be installed: 
# - schedule
# - yfinance 
# - bs4

import time  # to retrieve the time 
from datetime import datetime  # to convert time into readable format
import schedule  # to perform periodic scan
import stock_finder_plotter_module as mymodule


# Scheduler for running periodically the main logic in the background
def scheduler(period, epochs):
    """ This function will help to scan the website periodically
    period: it's the time difference between two loops
    epochs: it's the number of time you want to scan the given website"""
    
    start_time = time.time()# to get the start time of scrapping
    sec_date = lambda x: datetime.fromtimestamp(x).strftime("%B %d, %Y %I:%M")# function to convert time (in seconds) to a more readable format.
    
    
    j = 0
    schedule.every(period).minutes.do(do_main)# to run the parser function every "period" minutes
    
    while j < epochs*period: # this loop will run for total time i.e "epochs*period"        
        print(sec_date(time.time()))
        print('This code will run for another {} minutes'.format(epochs*period - j))# time remaining
        
        schedule.run_pending()# to run the function which is pending with the schedule
        time.sleep(60*period)# this will help to save the computational power, and hault the execution for "60*period" minutes

        j+=period# incrementing j with "period"
        
    end_time = time.time()


# Main function logic
stock_plotter = mymodule.StockFinderAndPlotter()
def do_main():
    stock_plotter.new_parser() # Parse news, extracting headline and paragraph
    stock_plotter.save_parsed_news() # Save parsed news into csv file
    stock_symbol = stock_plotter.search_symbol() # Search for stock symbols into the parsed news
    symbs = stock_plotter.save_symbol_occurence(stock_symbol) # calculate symbols frequency and save it to a file
    stock_plotter.plot_symbols_charts(symbs) # Plot found symbols

# Execute the main logic one time right away
do_main()
# Schedule to run main logic every 1 minute, 6 times 
scheduler(1,6)
