from os import listdir
from os.path import isfile, join
mypath = "./"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
id_set = set()
for f in onlyfiles:
  if f[-4:] != '.csv':
    continue
  print (f)
  with open(f) as fp: 
    lines = fp.readlines()
    count = 0 
    for line in lines[1:]:
      if count ==3:
        count = 0
        continue
      id_set.update(line.split(","))
      count +=1
with open("all_ids.text", "w") as nf:
  nf.write(",".join(list(id_set)))

