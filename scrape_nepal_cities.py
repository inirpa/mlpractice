import bs4 as bs
import urllib.request
import pandas as pd
from IPython.display import display_html

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/List_of_cities_in_Nepal')
soup = bs.BeautifulSoup(source, 'lxml')
tables = soup.find('table',{'class':'wikitable sortable'})

# print(tables)
# display_html(tables, raw=True)
# for table in tables:
dfs = pd.read_html(tables)
print(dfs)
# df = pd.DataFrame()
# for table in tables:
#     table_rows = table.find_all('tr')
#     for tr in table_rows:
#         td = tr.find_all('td')
#         for i in td:
#             # print(i.text)
#             df = pd.DataFrame([i.text])
#         df.to_excel('new.xls')