import numpy as np

PIXEL_MIN_VAL = 0
PIXEL_MAX_VAL = 255

PIXEL_VALUE_RANGE = (PIXEL_MIN_VAL, PIXEL_MAX_VAL) # used in histogram creation

def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):
    frame_patch = frame[xmin:xmax, ymin:ymax, :]

    hists = []

    for i in range(frame_patch.shape[2]):
        channel_hist = np.histogram(frame_patch[:, :, i],
                                    range=PIXEL_VALUE_RANGE,
                                    bins=hist_bin)
        hists.append(channel_hist)

    hist = np.concatenate(hists, axis=0)
    hist = hist/np.sum(hist) # normalize

    return hist