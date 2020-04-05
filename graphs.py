import numpy as np
from collections import Counter
from itertools import product, chain
import math
from scipy.signal import argrelextrema
import pickle
import networkx as nx


class Graph:
    def __init__(self, parameter, lyap_exp, transition_graph, pattern_label, clustering_coeff, out_strength, time_apparence):
        print('Processing...')
        self.parameter = parameter      #the parameter of the rossler oscillator
        
        self.lyap_exp = lyap_exp        #the lyapunov exponent of the oscillator
        
        self.transition_graph = transition_graph    #the adjacency matrix of the graph
        
        self.pattern_label = pattern_label          #the labels of the patterns/nodes
        
        self.clustering_coeff = clustering_coeff    #the clustering coefficient of each node
        
        self.out_strength = out_strength            #the out_strength of each node
        
        self.time_apparence = time_apparence        #the matrix of shape = (node, time) with 
                                                    #entries the tells you if that node 
                                                    #appears in ceartain time (1) or does not (0)

def calc_MI(X,Y,bins):

   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))
    return H

def find_lag(data, stepSize, tau_max = 200, bins =200, plot = False ):
    #find usable time delay via mutual information
    mis = []

    N = len(data)
    for tau in range(1, tau_max):
        M = N - tau
        unlagged = data[0:M]
        lagged = data[tau:N]
        mis.append(calc_MI(unlagged, lagged, bins = bins))
        best_tau = 0
        # mis.append(mutual_information_2d(unlagged, lagged, normalized=True))
    for i in mis:
        if i < 1/math.e:
            best_tau = mis.index(i)
            break
    if best_tau != 0:
        print('criterio e')
        pass
    else:
        #print('criterio minimo')
        mis = np.array(mis)

        minimun = argrelextrema(mis, np.less, order = int(1/stepSize)) 

        #print(minimun)
        best_tau = minimun[0][0] + 1

    if plot == False:
        return best_tau

    elif plot == True:

        tau_points = np.arange(1, tau_max)
        #plot time delay embedding
        fig = plt.plot(tau_points, mis), plt.xlabel('tau'), plt.ylabel('Mutual Information')
        return best_tau, fig

def mean_derivative(data, stepSize):
    num_points = len(data)
    p = []
    for point in range(num_points - 1):
        p.append((data[point + 1] - data[point])/stepSize)

    return p

def M_p(p):
    
    """M_p is the threshold for p"""
    
    p = np.array(p)
    return np.average(np.abs(p))

def symbolize_point(p, M_p):
    
    """It symbolize the time series with the criterion show below """
    
    p = np.array(p)
    symb = []
    for val in p:
        if val >= M_p:
            symb.append('R')
        elif val > 0 and val < M_p:
            symb.append('r')
        elif val == 0:
            symb.append('e')
        elif val < 0 and val > - M_p:
            symb.append('d')
        elif val <= -M_p:
            symb.append('D')

    return symb

def delay_embedding(data, emb_dim, delay):
    """It creats the embbeding phase space using the delay 
    
    and the embbeding dimmesion passed"""
   
    N = len(data)
    M = N - (emb_dim - 1)*delay
    delay_vec = []
    for i in range(emb_dim):
        for time in range(M):
            delay_vec[time][i] = data[time + i*delay]

    return delay_vec


def symbolize_vector(symb_points, emb_dim, delay):
    N = len(symb_points)
    M = N - (emb_dim - 1)*delay
    symb_vec = []
    time_points = dict()
    for time in range(M):
        temp_vector = []
        for i in range(emb_dim):
            temp_vector.append(symb_points[time + i*delay])
        symb_vec.append(''.join(temp_vector))
        if not symb_vec[time] in time_points:
            time_points[symb_vec[time]] = [time]
        else:
            time_points[symb_vec[time]].append(time)

    return symb_vec, time_points, M

def trasitional_graph(data_all, stepSize, emb_dim):

    data = data_all
    lag = find_lag(data, stepSize, tau_max = 50, bins = 200, plot = False)
    delay = lag
    print('delay = ', delay)

    histogram = []

    p = mean_derivative(data, stepSize)
    M = M_p(p)

    symb = symbolize_point(p, M)


    symb_vector, time_points, num_timePoints = symbolize_vector(symb, emb_dim,delay)


    all_edges = [(symb_vector[i], symb_vector[i + 1] ) for i in range(len(symb_vector)-1)]

    numEdges = Counter(all_edges)

    u, indices = np.unique(np.array(symb_vector),return_index=True)

    num_upatters = len(u)


    #possible edges with the unique patters
    possible_edges = list(product(u ,repeat = 2))

    possible_edges_dict = dict()
    for key in possible_edges:
        if key in numEdges:
            possible_edges_dict[key] = numEdges[key]
        else:
            possible_edges_dict[key] = 0


    matrix = np.zeros(shape=(num_upatters, num_upatters),dtype= int)
    time_apparence = np.zeros(shape= (num_upatters, num_timePoints ), dtype=int)

    count_coulumn = 0
    count_raw = 0
    patterns =[]
    for key in possible_edges_dict:
        if count_coulumn <  num_upatters - 1:
            matrix[count_raw, count_coulumn] = possible_edges_dict[key]
            count_coulumn += 1
            if count_raw == 0:
                patterns.append(key[1])

            else:
                pass

        elif count_coulumn == num_upatters - 1:
            matrix[count_raw, count_coulumn] = possible_edges_dict[key]
            if count_raw == 0:
                patterns.append(key[1])
            else:
                pass
            count_coulumn = 0
            count_raw += 1
    clustering_coeff = []
    G = nx.from_numpy_matrix(matrix)
    clustering = nx.clustering(G, weight= 'weight')

    for key in clustering:
        clustering_coeff.append(clustering[key])

    count_pattern = 0
    for pattern in patterns:
        time = time_points[pattern]
        for t in time:

            time_apparence[count_pattern, t ] = 1
        count_pattern += 1

    out_strength = np.sum(matrix, axis = 1, dtype= int)

    return matrix, patterns, clustering_coeff, out_strength, time_apparence

def degree_freq(data_all, stepSize, emb_dim):
    # delays = []
    # for node in range(90):
    #     data = data_all[:, node]
    #     lag = find_lag(data, stepSize= downsample, tau_max = 20, bins = 200, plot = False)
    #     delays.append(lag)
    
    # delay = int(np.average(np.array(delays)))
    delay = 3
    #print('delay = ', delay)
    
    histogram = []
    for node in range(90):
        data = data_all[:, node]
        #lag = find_lag(data, stepSize= downsample, tau_max = 20, bins = 200, plot = False)
        
        
        p = mean_derivative(data, stepSize)
        M = M_p(p)
    
        symb = symbolize_point(p, M)

    
        symb_vector = symbolize_vector(symb, emb_dim, delay = delay)
        
        
        #the last element has not an out-edge
        symb_vector.remove(symb_vector[-1])
    
        freq = dict()
    
        u, indices = np.unique(np.array(symb_vector),return_index=True)
        for pattern in u:
            freq[pattern] = symb_vector.count(pattern)
    
       
        for key in freq:
            histogram.append(freq[key])

        # all_edges = [(symb_vector[i], symb_vector[i + 1] ) for i in range(len(symb_vector)-1)]
    
        # numEdges = Counter(all_edges)
    
        # u, indices = np.unique(np.array(symb_vector),return_index=True)
    
        # num_upatters = len(u)
    
    
        # #possible edges with the unique patters
        # possible_edges = list(product(u ,repeat = 2))
    
        # possible_edges_dict = dict()
        # for key in possible_edges:
        #     if key in numEdges:
        #         possible_edges_dict[key] = numEdges[key]
        #     else:
        #         possible_edges_dict[key] = 0
    
    
        # matrix = np.zeros(shape=(num_upatters, num_upatters))
    
    
        # count_coulumn = 0
        # count_raw = 0
        # for key in possible_edges_dict:
        #     if count_coulumn < num_upatters - 1:
        #         matrix[count_raw, count_coulumn] = possible_edges_dict[key]
        #         count_coulumn += 1
    
        #     elif count_coulumn == num_upatters - 1:
        #         matrix[count_raw, count_coulumn] = possible_edges_dict[key]
        #         count_coulumn = 0
        #         count_raw += 1
    
        # np.fill_diagonal(matrix, 0)
        # weight = np.sum(matrix, axis=0)
        # weight = weight/np.max(weight)

        # for i in range(len(weight)):
        #     frequencies.append(weight[i])
        
    return np.array(histogram)/max(histogram)

def positive_values(data):
     """It transforms the points from the time series in
     positive values neccesary to create the visibility graph"""

     minimun = np.min(data)
     data_positive = np.array(data) + abs(minimun) + 1

     return data_positive

def horizontal_vg(data):

    numPoints = len(data)
    hvg = np.zeros(shape= (numPoints, numPoints))

    data = positive_values(data)

    for i in range(numPoints - 1):
        neighbor = []

        hvg[i, i + 1], hvg[i + 1, i] = 1, 1
        neighbor.append(data[i + 1 ])
        for j in range(i + 2, numPoints):
            if data[i] > max(neighbor):
                if data[j] > max(neighbor):
                    hvg[i, j], hvg[j, i] = 1, 1
                    neighbor.append(data[j])
                else:
                    neighbor.append(data[j])
                    pass
            else:
                break
    return hvg


def hvg_extended(t_series, time_step):
    
    downsample = time_step
    num_nodes = len(t_series[0])
    
    histogram = []
    for node in range(num_nodes):
        data = t_series[:, node]
        hvg = horizontal_vg(data)
        
        weight = np.sum(hvg, axis = 0)

        histogram.append(weight)
        
    histogram = np.array(list(chain.from_iterable(histogram)), dtype=int)
    
    
    
    return histogram/max(histogram)