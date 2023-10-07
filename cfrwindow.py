import numpy as np
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt

# this code implements the slinding window fits to the data

# data copied from Steve's spreadsheet
# just did this a parallel lists to make my life easy
week = ["5/24/2020", "5/31/2020", "6/7/2020", "6/14/2020", "6/21/2020", "6/28/2020", "7/5/2020", "7/12/2020", "7/19/2020", "7/26/2020", "8/2/2020", "8/9/2020", "8/16/2020", "8/23/2020", "8/30/2020", "9/6/2020", "9/13/2020", "9/20/2020", "9/27/2020", "10/4/2020", "10/11/2020", "10/18/2020", "10/25/2020", "11/1/2020", "11/8/2020", "11/15/2020", "11/22/2020", "11/29/2020", "12/6/2020", "12/13/2020", "12/20/2020", "12/27/2020", "1/3/2021", "1/10/2021", "1/17/2021", "1/24/2021", "1/31/2021", "2/7/2021", "2/14/2021", "2/21/2021", "2/28/2021", "3/7/2021", "3/14/2021", "3/21/2021", "3/28/2021", "4/4/2021", "4/11/2021", "4/18/2021", "4/25/2021", "5/2/2021", "5/9/2021", "5/16/2021", "5/23/2021", "5/30/2021", "6/6/2021", "6/13/2021", "6/20/2021", "6/27/2021", "7/4/2021", "7/11/2021", "7/18/2021", "7/25/2021", "8/1/2021", "8/8/2021", "8/15/2021", "8/22/2021", "8/29/2021", "9/5/2021", "9/12/2021", "9/19/2021", "9/26/2021", "10/3/2021", "10/10/2021", "10/17/2021", "10/24/2021", "10/31/2021", "11/7/2021", "11/14/2021", "11/21/2021", "11/28/2021", "12/5/2021", "12/12/2021", "12/19/2021",
        "12/26/2021", "1/2/2022", "1/9/2022", "1/16/2022", "1/23/2022", "1/30/2022", "2/6/2022", "2/13/2022", "2/20/2022", "2/27/2022", "3/6/2022", "3/13/2022", "3/20/2022", "3/27/2022", "4/3/2022", "4/10/2022", "4/17/2022", "4/24/2022", "5/1/2022", "5/8/2022", "5/15/2022", "5/22/2022", "5/29/2022", "6/5/2022", "6/12/2022", "6/19/2022", "6/26/2022", "7/3/2022", "7/10/2022", "7/17/2022", "7/24/2022", "7/31/2022", "8/7/2022", "8/14/2022", "8/21/2022", "8/28/2022", "9/4/2022", "9/11/2022", "9/18/2022", "9/25/2022", "10/2/2022", "10/9/2022", "10/16/2022", "10/23/2022", "10/30/2022", "11/6/2022", "11/13/2022", "11/20/2022", "11/27/2022", "12/4/2022", "12/11/2022", "12/18/2022", "12/25/2022", "1/1/2023", "1/8/2023", "1/15/2023", "1/22/2023", "1/29/2023", "2/5/2023", "2/12/2023", "2/19/2023", "2/26/2023", "3/5/2023", "3/12/2023", "3/19/2023", "3/26/2023", "4/2/2023", "4/9/2023", "4/16/2023", "4/23/2023", "4/30/2023", "5/7/2023", "5/14/2023", "5/21/2023", "5/28/2023", "6/4/2023", "6/11/2023", "6/18/2023", "6/25/2023", "7/2/2023", "7/9/2023", "7/16/2023", "7/23/2023", "7/30/2023", "8/6/2023"]
cases = [80151, 9858, 8535, 6204, 5689, 6301, 6776, 8574, 10181, 11387, 10681, 10566, 9176, 8362, 7967, 7429, 6677, 6994, 7117, 7451, 8051, 10431, 11944, 14177, 17585, 22616, 24062, 26281, 30657, 31901, 32209, 27540, 25581, 24427, 19848, 15834, 10630, 5871, 3705, 2050, 1545, 1261, 1032, 891, 839, 788, 907, 1043, 999, 1123, 957, 766, 722, 695, 463, 392, 331, 330, 460, 490, 792, 1441, 2192, 3128, 3903, 4533, 5078, 5078, 5192, 5005, 4390, 3866, 3798, 3506, 2925, 2861, 3138, 3778, 4143, 4310, 4564, 4380, 4045, 5952, 16647, 34715,
         45042, 46040, 35778, 23919, 15448, 9585, 5859, 3525, 2393, 1667, 1184, 1070, 1501, 2147, 2780, 4273, 5577, 6403, 8344, 8003, 8287, 7969, 8234, 8694, 10483, 11121, 12469, 13915, 13302, 13616, 12696, 11290, 10184, 9643, 8443, 8440, 8201, 7727, 8361, 9336, 9813, 11233, 10848, 10228, 10509, 13174, 17425, 18347, 17087, 15843, 18070, 19274, 14460, 12081, 11108, 10580, 10340, 10177, 10417, 9902, 8128, 7448, 6952, 6125, 5722, 5072, 4663, 4129, 3894, 3446, 2554, 2578, 2598, 2337, 2490, 2603, 2444, 2631, 2962, 3322, 3968, 4819]
deaths = [22212, 3216, 2501, 1816, 1533, 1258, 1283, 1361, 1569, 1951, 1980, 1974, 1796, 1667, 1498, 1346, 1201, 1159, 1142, 1173, 1223, 1464, 1778, 2022, 2469, 2922, 3686, 4365, 4785, 5475, 5744, 5326, 5225, 4858, 4382, 3717, 2870, 2041, 1354, 783, 617, 410, 268, 209, 206, 127, 179, 138, 181, 195, 167, 151, 118, 103, 90, 72, 43, 55, 60, 67, 67, 84, 151, 224, 306, 394, 481, 511, 535, 593, 538, 517, 453, 464, 377, 360, 366, 398,
          357, 425, 456, 411, 434, 368, 450, 711, 1137, 1454, 1341, 1185, 826, 650, 411, 228, 169, 113, 82, 74, 52, 80, 71, 88, 116, 137, 139, 190, 198, 183, 173, 173, 198, 218, 238, 315, 283, 293, 260, 264, 234, 197, 209, 194, 188, 163, 170, 198, 236, 247, 259, 224, 205, 209, 247, 324, 326, 302, 327, 368, 337, 278, 254, 215, 193, 182, 201, 201, 190, 166, 176, 159, 138, 121, 127, 94, 99, 81, 82, 66, 38, 50, 46, 60, 58, 38, 51, 50, 65, 85]

# convert to numpy arrays
cases = np.array(cases, dtype=np.float64)
deaths = np.array(deaths, dtype=np.float64)


def do_lag(x, f):
    # given a probability distribution, f
    # and a time series, x
    # compute the sum of the lags up to lagmax
    # this is used in the fitting functions
    y = x*(f.cdf(1)-f.cdf(0))
    for lag in range(1, lagmax):
        y[lag:] += x[0:-lag]*(f.cdf(lag+1)-f.cdf(lag))
    return y


def lag_lognormal(x, s, scale, a):
    # apply the lag function above with the lognormal distribution
    return a*do_lag(x, scipy.stats.lognorm(s, loc=0, scale=scale))


def lag_weibull2(x, scale, a):
    # apply the lag function with a weibull set to '2' which is the typical survival function
    return a*do_lag(x, scipy.stats.weibull_min(2, loc=0, scale=scale))


# how much of a lag we're going to look at
lagmax = 4

# how big our window is for fitting.
# larger windows will get a better fit but the response will be slower
window = lagmax*3

# let's loop over the data and accumulate the coefficients
weekid = []
al = []
c0l = []
aw = []
c0w = []
p0l = None
p0w = None
for wstart in range(1, len(cases)-window):
    wstop = wstart+window

    weekid.append(week[wstop])

    # fit the data to a log normal with lag
    p0l, pcov = scipy.optimize.curve_fit(lag_lognormal,
                                         cases[wstart:wstop],
                                         deaths[wstart:wstop],
                                         # p0=p0l,
                                         bounds=([0, 0, 0], [2, 5, 1]))
    (s, scale, a) = p0l
    f = scipy.stats.lognorm(s, loc=0, scale=scale)

    # take the fit parameters and pull out the coefficients
    al.append(a)
    c0 = (f.cdf(1)-f.cdf(0))
    ct = c0
    for lag in range(1, lagmax):
        ct += (f.cdf(lag+1)-f.cdf(lag))
    c0l.append(c0/ct)

    # fit the data to a weibull with lag
    p0w, pcov = scipy.optimize.curve_fit(lag_weibull2,
                                         cases[wstart:wstop],
                                         deaths[wstart:wstop],
                                         #  p0=p0w,
                                         bounds=([0, 0], [3, 1]))
    (scale, a) = p0w
    f = scipy.stats.weibull_min(2, loc=0, scale=scale)

    # take the fit parameters and pull out the coefficients
    aw.append(a)
    c0 = (f.cdf(1)-f.cdf(0))
    ct = c0
    for lag in range(1, lagmax):
        ct += (f.cdf(lag+1)-f.cdf(lag))
    c0w.append(c0/ct)

plt.clf()
# plt.plot(weekid, al)
# plt.xticks(range(0, len(weekid), 8), rotation=30)
# plt.show()
plt.plot(al)
plt.savefig('al.png')

plt.clf()
# plt.plot(weekid, c0l)
# plt.xticks(range(0, len(weekid), 8), rotation=30)
# plt.show()
plt.plot(c0l)
plt.savefig('c0l.png')

plt.clf()
# plt.plot(weekid, aw)
# plt.xticks(range(0, len(weekid), 8), rotation=30)
# plt.show()
plt.plot(aw)
plt.savefig('aw.png')

plt.clf()
# plt.plot(weekid, c0w)
# plt.xticks(range(0, len(weekid), 8), rotation=30)
# plt.show()
plt.plot(c0w)
plt.savefig('c0w.png')

plt.clf()
# plt.plot(weekid, aw)
# plt.xticks(range(0, len(weekid), 8), rotation=30)
# plt.show()
plt.plot(np.array(aw)*np.array(c0w))
plt.savefig('awc0w.png')

with open('weibull-coef.csv', 'w') as f:
    print('date, a, S0, a x S0', file=f)
    for i in range(len(weekid)):
        print(weekid[i], ',', aw[i], ',', c0w[i], ',', aw[i]*c0w[i], file=f)
