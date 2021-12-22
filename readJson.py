#import json
import pandas as pd
df = pd.read_json(r'marker_gene_stats.tsv')
df.to_csv(r'marker_gene_stats.csv', index = None)

#f = open("marker_gene_stats.tsv", "r")
#content = f.read()
#jsondecoded = json.loads(content)

#for entity in jsondecoded["Classes"]:
#    print(entity)
