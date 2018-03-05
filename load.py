from BiokeyData import *
import credentials

data = BiokeyData(credentials.postgres)

print(data.get_dwells('28513a33-02a3-4b09-88f6-92c9fdb9dcdf'))