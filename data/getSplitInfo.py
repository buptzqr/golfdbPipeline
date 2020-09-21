from scipy.io import loadmat
import pandas as pd
import sys

if __name__ == "__main__":
    split = sys.argv[1]
    filename="./val_split_{}.pkl".format(split)
    df = pd.read_pickle(filename)
    info = pd.DataFrame()
    info['id'] = df['id']
    info['youtube_id'] = df['youtube_id']
    info['view'] = df['view']
    info['slow'] = df['slow']
    info['events'] = df['events']
    info.to_csv(path_or_buf="./val_split_{}_info.csv".format(split), sep=' ',columns=['id', 'youtube_id', 'view', 'slow', 'events'])
    

