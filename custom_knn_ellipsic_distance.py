#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 11:15:12 2022 by Franz Kanngiesser


based on:
https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

2025-07-16 removing unused code
2025-07-28 cleaning up code
"""

import sys
import io
import math
import datetime
import random
import pandas as pd
import numpy as np
from scipy import spatial
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def euclidean_distance(row1, row2):
    """
    calculates eclidean distance between two vectors

    Parameters
    ----------
    row1 : 1D array
    row2 : 1D array

    Returns
    -------
    float

    """
    distance = 0.0
    for i in range(2):
        distance += (row1[i] - row2[i]) ** 2
    return np.sqrt(distance)


def ellipsic_distance(u, v, **kwargs):
    """
    calculates the distance using the semi-latus rectum of an ellipse

    Parameters
    ----------
    u : 1D array
    v : 1D array

    Returns
    -------
    p : float
        semi-latus rectum of ellipse, used as distance measure.

    """

    xdist = u[0] - v[0]
    ydist = u[1] - v[1]

    euclidian_dist = np.sqrt(xdist**2 + ydist**2)
    angle = np.arctan2(ydist, xdist)

    if kwargs:
        if kwargs["amv_dir"].any():
            meteo_angle = kwargs["amv_dir"]
            if np.isnan(meteo_angle).any():
                meteo_angle = 270.0
            math_angle = (270 - meteo_angle) % 360

            math_angle = np.radians(math_angle)

            p = euclidian_dist * (1 - 0.75 * np.cos(angle - math_angle))
            # if np.isnan(np.cos(angle - math_angle)):
            #    print(meteo_angle)

    else:
        p = euclidian_dist * (1 - 0.75 * np.cos(angle))

    return p


# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors, **kwargs):

    distances = []
    for train_row in train:
        # dist = euclidean_distance(test_row, train_row)
        dist = ellipsic_distance(test_row, train_row, **kwargs)
        distances.append((train_row, dist))

    distances.sort(key=lambda tup: tup[1])

    if kwargs:
        try:
            weight = kwargs["weight"]
            dists = []
            neighbors = []
            for i in range(num_neighbors):
                neighbors.append(distances[i][0])
                dists.append(distances[i][1])
            return neighbors, dists
        except:
            neighbors = []
            for i in range(num_neighbors):
                neighbors.append(distances[i][0])
            return neighbors

    else:
        neighbors = []
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors


# Make a classification prediction with neighbors
def predict_classification(train, test_row, num_neighbors, **kwargs):

    if kwargs:
        try:
            weight = kwargs["weight"]
            neighbors, distances = get_neighbors(
                train, test_row, num_neighbors, **kwargs
            )
            output_values = [row[-1] for row in neighbors]
            distances = np.asarray(distances)
            if min(distances) == 0:
                izero = np.where(distances == 0)
                izero = izero[0][0]
                prediction = neighbors[izero][-1]
                return prediction

            prediction = np.average(output_values, weights=1 / distances**2)
            prediction = np.around(prediction)

        except:
            neighbors = get_neighbors(train, test_row, num_neighbors, **kwargs)
            output_values = [row[-1] for row in neighbors]
            prediction = max(set(output_values), key=output_values.count)

    else:
        neighbors = get_neighbors(train, test_row, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)

    return prediction


def get_prediction(train, to_predict, num_neighbors, **kwargs):

    prediction_vector = np.full(to_predict.shape[0], np.nan)
    if kwargs["amv_dir"].any():
        for i in range(0, len(prediction_vector)):
            try:
                prediction_vector[i] = predict_classification(
                    train,
                    to_predict[i, :],
                    num_neighbors,
                    weight=kwargs["weight"],
                    amv_dir=kwargs["amv_dir"][i],
                )
            except:
                prediction_vector[i] = predict_classification(
                    train, to_predict[i, :], num_neighbors, amv_dir=kwargs["amv_dir"][i]
                )

    if not kwargs["amv_dir"].any():
        for i in range(0, len(prediction_vector)):
            prediction_vector[i] = predict_classification(
                train, to_predict[i, :], num_neighbors, **kwargs
            )

    return prediction_vector


def find_best_k(X_train, X_test, y_train, y_test, kmax, **kwargs):
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
        predicted1 = get_prediction(X_train, X_test, k, weight="dist", **kwargs)

        acc_d.append(accuracy_score(y_test, predicted1))

        predicted2 = get_prediction(X_train, X_test, k, **kwargs)
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


def get_amv_on_test_coords(test, amv_data):
    """


    based on https://github.com/blaylockbk/pyBKB_v3/blob/master/demo/KDTree_nearest_neighbor.ipynb

    Parameters
    ----------
    test : TYPE
        test dara.
    amv_data : TYPE
        DESCRIPTION.

    Returns
    -------
    amv_test : TYPE
        Atmospheric Motoin Vector (AMV) on test data.

    """

    grid_lon, grid_lat = np.meshgrid(
        np.linspace(-20, 52, num=amv_data.shape[0], endpoint=True),
        np.linspace(24, 60, num=amv_data.shape[1], endpoint=True),
    )

    tree = spatial.KDTree(np.column_stack([grid_lat.ravel(), grid_lon.ravel()]))

    dist, idx = tree.query(test)
    amv_test = amv_data.ravel()[idx]

    return amv_test


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
                - B / 6 * cos_2sigm * (-3 + 4 * sin_sigma**2) * (-3 + 4 * cos_2sigm**2)
            )
        )
    )

    s = b * A * (sigma - delta_sigma)
    return s


def check_train_test(X_train, X_test, y_train, y_test):
    """
    replace points from training data set, which have a unique class within 75 km
    radius, with points from test data set of the same class.
    this aims to ensure, that points indicating edges etc. are not missing from
    training data set
    """


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

            y_aux0 = y_test_[i]
            y_test_[i] = y_train_[new_index]
            y_train_[new_index] = y_aux0

        X_train = pd.DataFrame(X_train_, columns=["lat", "lon"])
        X_test = pd.DataFrame(X_test_, columns=["lat", "lon"])

    return X_train, X_test, y_train, y_test


time_of_interest = datetime.datetime(2022, 3, 15, 12, 0)


AMV_source_flag = "ERA5"
# pressure level for ERA5 AMVs
plevel = 700

DIR_PATH_INDICATORS = "{insert_path_to_indicator_directory}"
DIR_PATH_DIRECTION_INFO = "{inset_path_to_direction_directory}"
DIR_PATH_OUTPUT = "{insert_path_to_output_directory}"


old_stdout = sys.stdout
new_stdout = io.StringIO()
sys.stdout = new_stdout


df1 = pd.read_csv(
    DIR_PATH_INDICATORS
    + "alc_station_aerosol_layer_indicator"
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
    + "_2.csv"
)
df4 = pd.read_csv(
    DIR_PATH_INDICATORS
    + "isd_aerosol_layer_indicator"
    + time_of_interest.strftime("%Y-%m-%dT%H--%M--%S")
    + ".csv"
)

df_sat = pd.read_csv(
    DIR_PATH_INDICATORS
    + "SEVIRI_dust_layers/SEVIRI_aerosol_layer_indicator"
    + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
    + ".csv"
)

# indicate combinations of datasets to be used
# valid flags are: all, alc, aer, alc_aer, alc_aer_isd, alc_aer_eea, isd_eea
flag = "aer"

if flag == "all":
    df_gnd = pd.concat([df1, df2, df3, df4], ignore_index=True)
elif flag == "alc":
    df_gnd = df1
elif flag == "aer":
    df_gnd = df2
elif flag == "alc_aer":
    df_gnd = pd.concat([df1, df2], ignore_index=True)
elif flag == "alc_aer_isd":
    df_gnd = pd.concat([df1, df2, df4], ignore_index=True)
elif flag == "alc_aer_eea":
    df_gnd = pd.concat([df1, df2, df3], ignore_index=True)
elif flag == "isd_eea":
    df_gnd = pd.concat([df3, df4], ignore_index=True)

df_gnd = df_gnd.round(6)


df = pd.concat([df_gnd, df_sat], ignore_index=True)

X = df[["lat", "lon"]]

y = df["indicator"]

if AMV_source_flag == "ERA5":
    amv_dir1 = np.load(
        DIR_PATH_DIRECTION_INFO
        + "ERA5_directions/"
        + "ERA5_dir_"
        + (time_of_interest).strftime("%Y%m%d-%H%M%S")
        + "_"
        + str(plevel)
        + "hPa.npy"
    )
else:
    amv_dir1 = np.load(
        DIR_PATH_DIRECTION_INFO
        + "SEVIRI_AMV/AVM_dir_start_time"
        + (time_of_interest - datetime.timedelta(hours=1)).strftime("%Y%m%d%H%M")
        + "_linear.npy"
    )

amv_dir1 = amv_dir1[::-1]
amv_d = np.reshape(amv_dir1, 128 * 128)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=100)

X_train, X_test, y_train, y_test = check_train_test(X_train, X_test, y_train, y_test)

training_data = pd.concat([X_train, y_train], axis=1)

X_test = X_test.to_numpy()

amv_test = get_amv_on_test_coords(X_test, amv_dir1)

training_data = training_data.to_numpy()
k1, k2 = find_best_k(training_data, X_test, y_train, y_test, 75, amv_dir=amv_test)

grid_lon, grid_lat = np.meshgrid(
    np.linspace(-20, 52, num=128, endpoint=True),
    np.linspace(24, 60, num=128, endpoint=True),
)

flat_grid = np.c_[grid_lat.ravel(), grid_lon.ravel()]

prediction = get_prediction(training_data, flat_grid, k2, weight="dist", amv_dir=amv_d)

prediction = np.reshape(prediction, (128, 128))

if AMV_source_flag == "ERA5":
    np.save(
        DIR_PATH_OUTPUT
        + "/predicted_dust_SEVIRI_"
        + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
        + "UTC_ellipsicERA_"
        + flag
        + "_"
        + str(plevel)
        + "hPa.npy",
        prediction,
    )
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    with open(
        DIR_PATH_OUTPUT
        + "/log_knn_classification"
        + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
        + "_ellipsicERA_"
        + flag
        + "_"
        + str(plevel)
        + "hPa.txt",
        "w",
    ) as logfile:
        logfile.write(output)
else:
    np.save(
        DIR_PATH_OUTPUT
        + "/predicted_dust_SEVIRI_"
        + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
        + "UTC_ellipsic_"
        + flag
        + ".npy",
        prediction,
    )
    output = new_stdout.getvalue()
    sys.stdout = old_stdout
    with open(
        DIR_PATH_OUTPUT
        + "/log_knn_classification"
        + time_of_interest.strftime("%Y-%m-%d__%H--%M--%S")
        + "_ellipsic_"
        + ".txt",
        "w",
    ) as logfile:
        logfile.write(output)
