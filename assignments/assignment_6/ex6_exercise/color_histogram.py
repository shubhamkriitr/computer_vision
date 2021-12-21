import numpy as np
from matplotlib import pyplot as plt

PIXEL_MIN_VAL = 0
PIXEL_MAX_VAL = 255

PIXEL_VALUE_RANGE = (PIXEL_MIN_VAL, PIXEL_MAX_VAL) # used in histogram creation

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    frame_patch = frame[ymin:ymax, xmin:xmax, :] # NOTE: y is first dim

    hists = []

    for i in range(frame_patch.shape[2]):
        channel_hist, bins_ = np.histogram(frame_patch[:, :, i],
                                    range=PIXEL_VALUE_RANGE,
                                    bins=hist_bin)
        hists.append(channel_hist)

    hist = np.concatenate(hists, axis=0)
    hist = hist/np.sum(hist) # normalize

    return hist