import bs4 as bs
import urllib.request
import pandas as pd

source = urllib.request.urlopen('https://en.wikipedia.org/wiki/List_of_cities_in_Nepal')
soup = bs.BeautifulSoup(source, 'lxml')
tables = soup.find_all('table',{'class':'wikitable sortable'})

rows = [['Rank', 'Name', 'Nepali', 'District', 'Province', 'Population (2011)', 'Area', 'Website']]
for table in tables:
	table_rows = table.find_all('tr')
	for tr in table_rows:
		td = tr.find_all('td')
		cols = []
		for i in td:
			cols.append(i.text)
		rows.append(cols)

df = pd.DataFrame(rows)
df.to_excel('cities_of_nepal.xls')