import re # regular expression library
import matplotlib.pyplot as plt
import yfinance as yf
import requests  # to retrieve the html source code of the website
from bs4 import BeautifulSoup  # to create beautiful soup object from html source code
import pandas as pd  # to work with dataframes

class StockFinderAndPlotter:
    def __init__(self):
        # make these class attributes, so it persists across multiple schedules (loops)
        self.web_dict = {'Headline': [], 'Paragraph': []} # dictionary to store headlines and paragraph of the website.
        self.a = 0
        self.plotted_symb = []
        
    
    # Parse the news using BeautifulSoup, extract headline and paragraph
    # Specify the number of the first and the last page to parse
    def new_parser(self, pages=1, first=10):
        """This function will extract the headlines and paragraph from the given website. 
            First run, when a=0, it will scan pages from 2 to 10."""
        
        
        if self.a != 0:# For other runs, a!=0, it will scan periodically the first two pages. 
            first = pages
            pages = 0
            
    
        print("Scanning web pages from {} to {}".format(pages, first))
        
        for i in range(pages, first):# loop to scan through number of pages of website
    
            website = f'https://www.prnewswire.com/news-releases/news-releases-list/?page={i+1}&pagesize=100'
    
            source = requests.get(website).text# to retrieve the html code
            soup = BeautifulSoup(source, 'lxml')# creating the beautiful soup, which represents the document nested as a data structure.
    
            articles = soup.find('div', class_="col-md-8 col-sm-8 card-list card-list-hr")# find div tag with given class
    
            for article in articles.find_all('div', class_ = 'row'):# loop for all the div tag with given class 
                head = article.a.h3.text# to extract the headline
                extract = lambda x: x.split('ET')[1].strip()# to grasp the main portion, and to remove the time
                headline = extract(head)
                paragraph = article.a.p.text# to extract the paragraph corresponds to the heading
    
                self.web_dict['Headline'].append(headline)# append the heading to the heading list in web_dict
                self.web_dict['Paragraph'].append(paragraph)# append the heading to the heading list in web_dict
    
                self.web_dict['Headline'] = list(set(self.web_dict['Headline']))# Set is used to remove duplicacy, and then converted to list. 
                self.web_dict['Paragraph'] = list(set(self.web_dict['Paragraph']))
        self.a=1    
        print('Found {} news articles till now.'.format(len(self.web_dict['Headline'])))
        print(' ')
        
    # Collect the parsed new content and save them into a file
    def save_parsed_news(self):
        df = pd.concat([pd.DataFrame(self.web_dict['Headline']), pd.DataFrame(self.web_dict['Paragraph'])], ignore_index= True, axis = 1)# create dataframe of headings and paragraph scrapped from website.
        df.columns = self.web_dict.keys()# columns name
        df.head()
            
        df.to_csv('news1111.csv',index=False)# storing the dataframe into csv file.
    
    
    # Search for stock symbols inside the news' paragraphs
    def search_symbol(self): 
        stock_symbol = [] ## to store the found symbol
           
        for para in self.web_dict['Paragraph']: # to scrap the stock symbol from paragraphs
            
            if '(TSX:' in para:
                symbol = para.split('(TSX:')[1].split(')')[0].strip()
            
                # ensuring only stock symbol will be extracted, also validate symbol format 
                if len(symbol) < 9 and re.search(r"^[a-zA-Z]{1,4}|\d{1,3}(?=\.)|\d{4,}$", symbol) != None:
                    stock_symbol.append(symbol)# appending the stock symbol into list
        
        print("Found stock symbols", stock_symbol)
            
        return stock_symbol
    
    # to find the number of occurence of symbols
    def occurence(self, lis): 
        stock_symbol_dic = {'symbol':[], 'occurence':[]}# to store the found symbol and their occurence.
        
        for e in lis:
            stock_symbol_dic['symbol'].append(e)
            stock_symbol_dic['occurence'].append(lis.count(e))# appending the number of occurence corresponds to each symbol.
        
        return stock_symbol_dic
    
    # Save symbols and their frequencies into a CSV file
    def save_symbol_occurence(self, stock_symbol):
        stock_symbol_dic = self.occurence(stock_symbol)
        
        print(stock_symbol_dic)
        
        out_df = pd.DataFrame(stock_symbol_dic).sort_values(by = 'occurence', ascending = False) # creating data frame from stock_symbol_dic 
        out_df
        out_df.to_csv('symbol_occurence1111.csv', index=False)# storing the found symbol and their occurence into a csv file
        df = pd.read_csv('symbol_occurence1111.csv')
        symbs = df['symbol'][0:6]
    
        # If not symbol was found, use a default in order to plot something right away   
        if len(symbs) <= 0:
            print("No symbols found in news, using fallback stock symbol")
            print(' ')
            # Using Apple stock as fallback
            return ["AAPL"]
        
        return symbs
    
    # Use PyPlot to create a chart for the volume and close price of every stock symbol
    def plot_symbols_charts(self, symbs):
        
        for symb in symbs:
            
            if symb not in self.plotted_symb:# only plot new symbols that found in current epoch
                t = yf.Ticker(symb) # Create ticket object for the symbol
                h = t.history(period="1mo") # Get 1 month of historical data, 1 day intervals
                if h.empty: # Do not plot symbols that couldn't be found by Yahoo Finance
                    continue
                # h is a pandas dataframe. 
                # We will use integer indexing to get the most recent 10 days
                d10 = h.iloc[-10:]
            
                # Plot the Volume data
                fig = plt.figure()
                plt.plot(d10.Volume, marker="o")
                plt.title(symb + " Volumes")
                plt.xlabel("Date")
                plt.ylabel("Volume")
                # The xticks should be the dates for each data point
                # In the data, the indexes are dates
                plt.xticks(ticks=d10.index, rotation='vertical')
                plt.grid() # enable grid display
                fig.savefig(symb+"_Volumes.png", dpi=400, transparent=True, bbox_inches='tight')
                plt.show()
            
                # Plot the Close data
                fig = plt.figure()
                plt.plot(d10.Close, marker="o")
                plt.title(symb + " Close Price")
                plt.xlabel("Date")
                plt.ylabel("Close Price ($)")
                # The xticks should be the dates for each data point
                # In the data, the indexes are dates
                plt.xticks(ticks=d10.index, rotation='vertical')
                plt.grid() # enable grid display
                fig.savefig(symb+"_Close.png", dpi=400, transparent=True, bbox_inches='tight')
                plt.show()
                
                self.plotted_symb.append(symb)