#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:32:16 2024

@author: fkanngiesser

based on dust_extend_knn_classification.py

2025-07-28: code clean-up
"""

import sys
import io
import datetime
import math
import random
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def find_best_k(X_train, X_test, y_train, y_test, kmax):
    """
    find best  number of neighbours to be considered

    Parameters
    ----------
    X_train : float (array)
        predictor, training dataset.
    X_test : float (array)
        predictor, test dataset.
    y_train : array
        label/class, training dataset.
    y_test : array
        label/class, test dataset.
    kmax : integer
        maximum number of neighbours to be considered.

    Returns
    -------
    list
        [best k for distance weighting, best k for uniform weighting]

    """
    acc_d = []
    acc_u = []
    for k in range(1, kmax + 1):
        classifier1 = KNeighborsClassifier(p=2, weights="distance", n_neighbors=k)
        classifier1.fit(X_train, y_train)
        predicted1 = classifier1.predict(X_test)
        acc_d.append(accuracy_score(y_test, predicted1))

        classifier2 = KNeighborsClassifier(p=2, weights="uniform", n_neighbors=k)
        classifier2.fit(X_train, y_train)
        predicted2 = classifier2.predict(X_test)
        acc_u.append(accuracy_score(y_test, predicted2))

    try:  # not returning the first element, i.e. avoiding pure nearest-neighbour
        kbd = acc_d.index(max(acc_d), 1) + 1
    except ValueError:
        kbd = acc_d.index(max(acc_d), 0) + 1
    try:
        kbu = acc_u.index(max(acc_u), 1) + 1
    except ValueError:
        kbu = acc_u.index(max(acc_u), 0) + 1

    print("------------------- best k -------------------")
    print("distance-weighted: k_best=" + str(kbd) + " with acc=" + str(max(acc_d)))
    print("uniformly weighted: k_best=" + str(kbu) + " with acc=" + str(max(acc_u)))
    return [kbu, kbd]


def vincenty(coords1, coords2):
    """
    second (indirect) method of Vincenty's formulae, calculating the geographical distance
    between two points iteratively
    the method assumes the Earth to be an oblate spheroid, radius and flattening are taken
    from the 1984 version of the World Geodetic System (WGS 84)
    Caution: for nearly antipodal points the iteration may fail to converge

    for reference see either:
    https://en.wikipedia.org/wiki/Vincenty%27s_formulae
    Karney, C. F. F., Algorithms for geodesics, Journal of Geodesy, Springer, 2012, 87, 43-55

    Parameters
    ----------
    coords1 : list with two elements (float)
        latitude/longitude pair of first location
    coords2 : list with two elements (float)
        latitude/longitude pair of second location

    Returns
    -------
    s : float
        distance between the two coordinates in m.

    """
    a = 6378137.0  # length of semi-major axis of ellipsoid (radius at equator) i m
    f = 1 / 298.257223563  # flattenening of ellipsoid
    b = (1 - f) * a
    phi_1, L_1 = coords1
    phi_2, L_2 = coords2

    phi_1 = math.radians(phi_1)
    phi_2 = math.radians(phi_2)
    L_1 = math.radians(L_1)
    L_2 = math.radians(L_2)

    U_1 = math.atan((1 - f) * math.tan(phi_1))
    U_2 = math.atan((1 - f) * math.tan(phi_2))

    L = L_2 - L_1

    Lambda = L

    for i in range(0, 1000):
        sin_sigma = np.sqrt(
            (math.cos(U_2) * math.sin(Lambda)) ** 2
            + (
                math.cos(U_1) * math.sin(U_2)
                - math.sin(U_1) * math.cos(U_2) * math.cos(Lambda)
            )
            ** 2
        )

        cos_sigma = (math.sin(U_1) * math.sin(U_2)) + (
            math.cos(U_1) * math.cos(U_2) * math.cos(Lambda)
        )

        sigma = np.arctan2(sin_sigma, cos_sigma)

        sin_alpha = (math.cos(U_1) * math.cos(U_2) * math.sin(Lambda)) / sin_sigma
        cos_alpha = np.sqrt(1 - sin_alpha**2)
        cos_2sigm = cos_sigma - (2 * math.sin(U_1) * math.sin(U_2)) / (cos_alpha**2)

        C = f / 16 * cos_alpha**2 * (4 + f * (4 - 3 * cos_alpha**2))
        Lambda_prev = Lambda
        Lambda = L + (1 - C) * f * sin_alpha * (
            sigma
            + C * sin_sigma * (cos_2sigm + C * cos_sigma * (-1 + 2 * cos_2sigm**2))
        )
        if abs(Lambda - Lambda_prev) <= 10e-12:
            break

    u_sq = cos_alpha**2 * (a**2 - b**2) / b**2
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))

    delta_sigma = (
        B
        * sin_sigma
        * (
            cos_2sigm
            + 0.25
            * B
            * (
                cos_sigma * (-1 + 2 * cos_2sigm**2)
                - B
                / 6
                * cos_2sigm
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos_2sigm**2)
            )
        )
    )

    s = b * A * (sigma - delta_sigma)
    return s


def check_train_test(X_train, X_test, y_train, y_test):

    random.seed(10)  # to ensure reproducibility

    X_train_ = X_train.to_numpy()
    y_train_ = y_train.to_numpy()
    X_test_ = X_test.to_numpy()
    y_test_ = y_test.to_numpy()
    to_swap = []
    for i in range(0, len(y_test_)):
        dist = []
        for j in range(0, len(y_train_)):
            dist.append(vincenty(X_train_[i, :], X_train_[j, :]))
        dindex = np.where(np.asarray(dist) < 75000)
        dindex = dindex[0]
        if len(dindex) > 0:
            for n in range(0, len(dindex)):
                num = 0
                ind = dindex[n]
                if y_train_[ind] != y_train_[i]:
                    num += 1
            if num > 0 and num > len(dindex) - 2:
                to_swap.append(i)
    print(
        str(len(to_swap))
        + " elements of the test data set are exchanged with training data set"
    )

    if to_swap:
        for i in to_swap:
            new_index = random.randrange(0, len(y_train))
            while y_train_[new_index] != y_test_[i]:
                new_index += 1

            X_aux0 = X_test_[i, 0]
            X_aux1 = X_test_[i, 1]
            X_test_[i, 0] = X_train_[new_index, 0]
            X_test_[i, 1] = X_train_[new_index, 1]
            X_train_[new_index, 0] = X_aux0
            X_train_[new_index, 1] = X_aux1

        X_train = pd.DataFrame(X_train_, columns=["lat", "lon"])
        X_test = pd.DataFrame(X_test_, columns=["lat", "lon"])

    return X_train, X_test


time_of_interest = datetime.datetime(2022, 3, 15, 12, 0)

DIR_PATH_INDICATORS = "{insert_path_to_indicator_directory}"
DIR_PATH_OUTPUT = "{insert_path_to_output_directory}"


old_stdout = sys.stdout
new_stdout = io.StringIO()
sys.stdout = new_stdout

df1 = pd.read_csv(
    DIR_PATH_INDICATORS
    + "alc_station_aerosol_layer_indicator_"
    + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
    + ".csv"
)
df2 = pd.read_csv(
    DIR_PATH_INDICATORS
    + "aeronet_station_aerosol_layer_indicator"
    + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
    + ".csv"
)
df3 = pd.read_csv(
    DIR_PATH_INDICATORS
    + "eea_aerosol_layer_indicator"
    + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
    + ".csv"
)
df4 = pd.read_csv(
    DIR_PATH_INDICATORS
    + "isd_aerosol_layer_indicator"
    + time_of_interest.strftime("%Y-%m-%dT%H--%M--%S")
    + ".csv"
)

df_sat = pd.read_csv(
    DIR_PATH_INDICATORS
    + "SEVIRI_aerosol_layer_indicator"
    + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
    + ".csv"
)

# indicate combinations of datasets to be used
# valid CASE_FLAGs are: all, alc, aer, alc_aer, alc_aer_isd, alc_aer_eea, isd_eea
CASE_FLAG = "aer"

if CASE_FLAG == "all":
    df_gnd = pd.concat([df1, df2, df3, df4], ignore_index=True)
elif CASE_FLAG == "alc":
    df_gnd = df1
elif CASE_FLAG == "aer":
    df_gnd = df2
elif CASE_FLAG == "alc_aer":
    df_gnd = pd.concat([df1, df2], ignore_index=True)
elif CASE_FLAG == "alc":
    df_gnd = df1
elif CASE_FLAG == "alc_aer_isd":
    df_gnd = pd.concat([df1, df2, df4], ignore_index=True)
elif CASE_FLAG == "alc_aer_eea":
    df_gnd = pd.concat([df1, df2, df3], ignore_index=True)
elif CASE_FLAG == "isd_eea":
    df_gnd = pd.concat([df3, df4], ignore_index=True)

df_gnd = df_gnd.round(6)


df = pd.concat([df_gnd, df_sat], ignore_index=True)

X = df[["lat", "lon"]]

y = df["indicator"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)


X_train, X_test = check_train_test(X_train, X_test, y_train, y_test)


k_best = find_best_k(X_train, X_test, y_train, y_test, 75)

weights = ["uniform", "distance"]

for i in range(0, len(weights)):
    # Creating a classifier object in sklearn
    clf = KNeighborsClassifier(p=2, weights=weights[i], n_neighbors=k_best[i])

    # Fitting our model
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    print(accuracy_score(y_test, predictions))

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(-20, 52, num=128, endpoint=True),
        np.linspace(24, 60, num=128, endpoint=True),
    )

    flat_grid = np.c_[grid_lat.ravel(), grid_lon.ravel()]

    df_grid = pd.DataFrame(flat_grid, columns=["lat", "lon"])

    predict_grid = clf.predict(df_grid)

    predict_grid = predict_grid.reshape((grid_lat.shape))


    np.save(
        DIR_PATH_OUTPUT
        + "predicted_dust_SEVIRI_"
        + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
        + "UTC_"
        + weights[i]
        + "_"
        + CASE_FLAG
        + ".npy",
        predict_grid,
    )


output = new_stdout.getvalue()
sys.stdout = old_stdout
with open(
    DIR_PATH_OUTPUT
    + "/log_knn_classification"
    + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
    + "_"
    + CASE_FLAG
    + ".txt",
    "w",
) as logfile:
    logfile.write(output)
