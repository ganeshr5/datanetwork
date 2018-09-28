
# import the required packages
import numpy as np
import pandas as pd
import arxivscraper
import matplotlib.pyplot as plt
import networkx as nx
import collections


### Part 1: Scraping the data from arxiv
# Create scraper, scrape arxiv database, store output as Pandas data frame

scraper = arxivscraper.Scraper(category='physics:astro-ph', date_from='2017-04-24',date_until='2017-05-05')
output = scraper.scrape()
cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
df = pd.DataFrame(output, columns = cols)
#df.head()


### Part 2: Create the network
G = nx.Graph()
df2 = df[['authors']]
#df2.head()
m=0
n=0
G.clear()
numrows = df2.shape[0]

# add each author name as a node
for i in range(numrows):
    x = (df2.iloc[i][0])
    
    # add nodes
    for j in (df2.iloc[i][0]):
        m=m+1
        G.add_node(j)
        
    # add edges
    for k in range(len(x)):
        for l in range(k, len(x) - 1):
            n=n+1
            G.add_edge(x[l], x[l+1])    

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

# To draw the network
pos = nx.spring_layout(G)
plt.figure(0)
nx.draw(G, pos)
nx.draw_networkx_nodes(G, pos)
nx.draw_networkx_edges(G, pos)
# To save the file
plt.savefig("proj1graph.png")     


### Part 3 - Calculate Network properties

# Degree Distribution
degree_distrib = dict(nx.degree(G))
plt.figure(1)
degree_seq = sorted([d for n, d in G.degree()], reverse = True)
degree_count = collections.Counter(degree_seq)
d, c = zip(*degree_count.items())
plt.bar(d, c, width = 0.5, color='g')
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Histogram for Degree Distribution")
plt.savefig("proj1deg.png")     

# Clustering Coefficient
clust_coeff = nx.clustering(G)
plt.figure(2)
cval = clust_coeff.values()
cval = sorted(cval)
cnt = collections.Counter(cval)
plt.xlim((0.0, 1.0))
plt.bar(cnt.keys(), cnt.values(), width = 0.5, color = 'b')
plt.xlabel("Clustering Coefficient")
plt.ylabel("Frequency")
plt.title("Histogram for Clustering Coefficient")
#plt.show()
plt.savefig("proj1clust.png")

# Average Clustering
avg_clust = nx.average_clustering(G)

# Diameter
con = nx.is_connected(G)
# The graph is not connected

# HITS and pagerank
hits_G = nx.hits(G)[0]
page_rank = nx.pagerank(G)

# closeness and betweenness
cl = nx.closeness_centrality(G)
bet = nx.betweenness_centrality(G)
