def mean_normalization(df):
    frames = list(df)
    length = len(df)

    for i in range(length):
        frames[i] = (frames[i] - frames[i].mean()) / frames[i].std()

    return frames

def min_max_normalization(df):
    frames = list(df)
    length = len(frames)

    for i in range(length):
        frames[i] = (frames[i] - frames[i].min()) / (frames[i].max() - frames[i].min())
    return frames