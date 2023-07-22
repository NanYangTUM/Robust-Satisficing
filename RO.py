from gurobipy import *
# from gurobipy import Env
import numpy as np
import pandas as pd
from numpy import array
from numpy.linalg import norm
from itertools import product
import datetime
import random
#import multiprocess as mp
import warnings
import multiprocessing as mp

# global print settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.options.mode.chained_assignment = None

# hyperparameter settings
T,L,IT = 7,14,3

elective=['ELECTIVE', 'SURGICAL SAME DAY ADMISSION', 'SURGICAL SAME DAY ADMISSION']
emergency=['AMBULATORY OBSERVATION', 'DIRECT EMER.', 'URGENT', 'EW EMER.', 'DIRECT OBSERVATION', 'EU OBSERVATION', 'OBSERVATION ADMIT']

# MIMIC-IV includes 10 years data, however after shift becomes arround 100 years data.
year_number=10
year_remain=1
combine_year=int(year_number/year_remain)

week_number=110

admissions=pd.read_csv('admissions.csv')
df=admissions.copy()
df=df.drop(['subject_id','hadm_id','deathtime','admission_location','discharge_location',
         'insurance','language','marital_status','ethnicity','edregtime','edouttime','hospital_expire_flag'], axis=1)
df['admittime']=pd.to_datetime(df['admittime'], format='%Y-%m-%d', errors='coerce')
df['dischtime']=pd.to_datetime(df['dischtime'], format='%Y-%m-%d', errors='coerce')
df=df.sort_values('admittime')

df['losi']=(df['dischtime']-df['admittime']).round('1d')
df["los"] = (df["losi"]).dt.days

df['admitday']=df['admittime'].dt.weekday
df=df.drop('losi', axis=1)

index=[i for i, n in enumerate(list(df['admitday'])) if n == 0][0]
start_date = df.iloc[index,0]

end_date = start_date + datetime.timedelta(days=7*week_number*combine_year)
# end_date = start_date + datetime.timedelta(days=7)

data = df.loc[(df['admittime'] >= start_date) & (df['admittime'] < end_date)]
data = data.loc[df['admission_type'].isin(elective) | df['admission_type'].isin(emergency)]

data = data.loc[data.los>=0]
data.loc[data.los>L,'los']=L

data['week']=''
#data['year']=''
#data=data.drop(['admittime','dischtime'])

#len(data)

initial_week, initial_year = 0, 0
data.iloc[0, 5] = initial_week
# data.iloc[0,6]=initial_year
week, year = [], []

for i in range(1, len(data)):
    if data.iloc[i, 4] - data.iloc[i - 1, 4] == -6:
        initial_week += 1
        initial_week %= week_number
    data.iloc[i, 5] = initial_week
#     if data.iloc[i,5]-data.iloc[i-1,5]==-51:
#         initial_year+=1
#     data.iloc[i,6]=initial_year

data.loc[data.admission_type.isin(elective), 'admission_type'] = 'elective'
data.loc[data.admission_type.isin(emergency), 'admission_type'] = 'emergency'

data = data.sort_values(['week', 'admitday'], ascending=[True, True])

# df=df.drop('year', axis=1)

data.head(10)

## Daily and weekly - emergency and elective arrival split

week = 0
emergency_week = np.zeros((week_number, T))
elective_week = np.zeros((week_number, T))
for i in range(len(data) - 1):
    t = data.iloc[i, 4]
    if data.iloc[i, 2] == 'elective':
        elective_week[int(week), t] += 1
    if data.iloc[i, 2] == 'emergency':
        emergency_week[int(week), t] += 1
    if t == 6 and data.iloc[i + 1, 4] != 6:
        week += 1

# print('\n', 'daily_total_admission:', np.mean(emergency_week, axis=0) + np.mean(elective_week, axis=0))
# print('\n', 'daily_emergency_arrival:', np.mean(emergency_week, axis=0), '\n', '\n', 'daily_elective_arrival:',
#       np.mean(elective_week, axis=0))
# print('\n', 'weekly_emergency_arrival:', np.sum(emergency_week, axis=1), '\n', '\n', 'weekly_elective_arrival:',
#       np.sum(elective_week, axis=1))
t_0 = 55

data['time_window'] = ''
data.time_window = (data.week - t_0) * 7 + data.admitday
data['time_window'] = data['time_window'].astype(int)

S = 50

def solve_RO(index):
    with Env() as env0, Model(env=env0) as model0:

        input_data = parameter_RO[index]
        T = input_data[0]
        c = input_data[1]
        min_weekquota = input_data[2]
        mu0 = input_data[3]

        eta_equal = int(min_weekquota / 7)

        T_minus = list(range(-L + 1, 0))
        T_plus = list(range(T))
        T_list = list(range(-L + 1, T))

        L_list = []
        for t in T_list:
            L_list_t = list(range(max(1, 1 - t), min(L, T - t) + 1))
            L_list.append(L_list_t)

        J_list_aux = []
        for t in T_list:
            J_list_t_aux = []
            for i in L_list[t + L - 1]:
                J_list_t_aux.append((t, i))
            J_list_aux.append(J_list_t_aux)

        J_list = []
        for t in T_list:
            for i in L_list[t + L - 1]:
                J_list.append((t, i))

        L_list_plus = []
        for t in T_list:
            L_list_t = list(range(max(1, 1 - t), min(L, T - t) + 2))
            L_list_plus.append(L_list_t)

        J_list_plus = []
        for t in T_list:
            for i in L_list_plus[t + L - 1]:
                J_list_plus.append((t, i))

        z0 = np.zeros((T, T + L - 1, L))
        for t in range(T):
            for tau, l in product(range(T + L - 1), range(L)):
                if tau - L + 1 + l == t:
                    z0[t, tau, l] = 1

        LOS = list(range(1, L + 1))
        LOS_plus = list(range(1, L + 2))

        mul = int((T + L) // 7)

        min_week_number = int(7 - mul)
        alpha_plus = np.ones(T)

        S_list = list(range(S))
        omega = np.ones(T)
        time_min = min(data_new0[index].time_window)

        p = np.zeros((S, T + L - 1, L + 1))
        a = np.zeros((S, T + L - 1, L + 1))
        alpha = np.zeros((S, T + L - 1, L + 1))
        # eta_0 = np.zeros((S, T))

        for s in S_list:
            time_current = time_min + (s + min_week_number) * 7

            # print(time_current)
            for tau, l in J_list:
                a[s, tau + L - 1, l - 1] = len(data_new0[index].loc[(data_new0[index].admission_type == 'elective') &
                                                                   (data_new0[
                                                                        index].time_window == time_current + tau) & (
                                                                           data_new0[index].los >= l)])
                p[s, tau + L - 1, l - 1] = len(data_new0[index].loc[(data_new0[index].admission_type == 'emergency') &
                                                                   (data_new0[
                                                                        index].time_window == time_current + tau) & (
                                                                           data_new0[index].los >= l)])

                alpha[s, tau + L - 1, l - 1] = a[s, tau + L - 1, l - 1] / len(
                    data_new0[index].loc[(data_new0[index].admission_type == 'elective') &
                                        (data_new0[index].time_window == time_current + tau)])
        #     print('iter:',it,'\n','p0_bar:',p0_bar,'alpha0_bar:',alpha0_bar,'eta_bar:',eta_bar)
        p0_bar = np.mean(p, axis=0)
        alpha0_bar = np.mean(alpha, axis=0)

        result_BOR_RO, final_BOR_RO, total_eta_RO = [], [], []

        for it in range(IT):

            eta_minus0, p_minus, alpha_minus0 = -np.ones(L - 1), -np.ones(L - 1), -np.ones(L - 1)

            for t in T_minus:
                eta_minus0[t + L - 1] = len(
                    data_new0[index].loc[
                        (data_new0[index].time_window == t) & (data_new0[index].admission_type == 'elective')])
                p_minus[t + L - 1] = len(
                    data_new0[index].loc[
                        (data_new0[index].time_window == t) & (data_new0[index].admission_type == 'emergency') &
                        (data_new0[index].los + data_new0[index].time_window >= 0)])

                alpha_minus0[t + L - 1] = len(
                    data_new0[index].loc[
                        (data_new0[index].time_window == t) & (data_new0[index].admission_type == 'elective') &
                        (data_new0[index].los + data_new0[index].time_window >= 0)]) / eta_minus0[t + L - 1]

            ############################################################### RO

            model0.Params.MIPFocus = 1
            model0.Params.Cuts = 2
            model0.Params.timeLimit = 150

            rho0 = model0.addVar(lb=-GRB.INFINITY, vtype="C")
            eta0 = model0.addVars(T_plus, lb=5, ub=80, vtype="C")
            s0 = model0.addVars(J_list, lb=-100, vtype="C")
            u0 = model0.addVars(J_list, lb=0, vtype="C")
            v0 = model0.addVars(J_list, lb=-100, vtype="C")
            w0 = model0.addVars(J_list, lb=0, vtype="C")
            # some larger than 0 GRB.INFINITY
            lamdba0_tp = model0.addVars(T_plus, J_list_plus, lb=0, vtype="C")
            lamdba0_ta = model0.addVars(T_plus, J_list_plus, lb=0, vtype="C")
            #
            y0_tp = model0.addVars(T_plus, J_list, lb=0, vtype="C")
            y0_ta = model0.addVars(T_plus, J_list, lb=0, vtype="C")

            p_constraint = model0.addVars(T_plus, J_list, lb=-GRB.INFINITY, vtype="C")
            a_constraint = model0.addVars(T_plus, J_list, lb=-GRB.INFINITY, vtype="C")

            # model0.update()

            ### constraint

            model0.addConstrs(
                (quicksum(eta0[t] for t in range(7 * i, 7 * (i + 1))) == min_weekquota for i in range(T // 7)))

            model0.addConstrs(quicksum(y0_tp[t, tau, l] + y0_ta[t, tau, l] for tau, l in J_list) +
                              quicksum(
                                  (p_minus[tau + L - 1] * lamdba0_tp[t, tau, 1 - tau] + alpha_minus0[tau + L - 1] *
                                   lamdba0_ta[t, tau, 1 - tau]) for tau in T_minus) +
                              quicksum((
                                  max_p0_count[tau % 7] * lamdba0_tp[t, tau, 1] + lamdba0_ta[t, tau, 1] for tau in
                              T_plus)) <= rho0 +
                              c for t in T_plus)

            model0.addConstrs(
                p_constraint[t, tau, l] == - s0[tau, l] + z0[t, tau + L - 1, l - 1] - lamdba0_tp[t, tau, l] + lamdba0_tp[
                    t, tau, l + 1] for t in T_plus for tau, l in J_list)

            model0.addConstrs(
                4 * u0[tau, l] * y0_tp[t, tau, l] >= p_constraint[t, tau, l] * p_constraint[t, tau, l] for t in T_plus for
                tau, l in J_list)

            model0.addConstrs(
                a_constraint[t, tau, l] == v0[tau, l] - eta_minus0[tau + L - 1] * z0[t, tau + L - 1, l - 1] +
                lamdba0_ta[t, tau, l] - lamdba0_ta[t, tau, l + 1] for t in T_plus for tau, l in J_list if tau < 0)
            # print('(tau,l):',(tau,l),'eta_minus0[tau + L - 1]:',eta_minus0[tau + L - 1])
            model0.addConstrs(
                a_constraint[t, tau, l] == v0[tau, l] - eta0[tau] * z0[t, tau + L - 1, l - 1] + lamdba0_ta[
                    t, tau, l] - lamdba0_ta[t, tau, l + 1] for t in T_plus for tau, l in J_list if tau >= 0)

            model0.addConstrs(
                4 * w0[tau, l] * y0_ta[t, tau, l] >= a_constraint[t, tau, l] * a_constraint[t, tau, l] for t in T_plus for
                tau, l in J_list)

            model0.setObjective(rho0 + quicksum(
                p0_bar[tau + L - 1, l - 1] * s0[tau, l] + p0_bar[tau + L - 1, l - 1] * p0_bar[tau + L - 1, l - 1] * (
                        1 + mu0 * mu0) * u0[tau, l] +
                alpha0_bar[tau + L - 1, l - 1] * v0[tau, l] + alpha0_bar[tau + L - 1, l - 1] * alpha0_bar[
                    tau + L - 1, l - 1] * (1 + mu0 * mu0) * w0[tau, l]
                for tau, l in J_list), GRB.MINIMIZE)

            model0.optimize()

            status = model0.status

            # retrieve data from model

            try:
                eta_sol0 = [round(i.X) for i in eta0.values()]
                # obj_RO[index].append(model0.ObjVal)
            except:
                eta_sol0 = [eta_equal] * T

            total_eta_RO.append(eta_sol0)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=(
                        ".*will attempt to set the values inplace instead of always setting a new array. "
                        "To retain the old behavior, use either.*"
                    ),
                )

                BOR0_RS, BOR0_RO, BOR0_equal = [], [], []
                for dow in range(7):
                    eta0_RO = round(eta_sol0[dow])
                    # eta0_RS = int(eta_sol0[dow])

                    data_0 = data.loc[(data.admission_type == 'elective') & (data.admitday == dow)].copy()
                    data_1 = data.loc[(data.admission_type == 'emergency') & (data.time_window == it * 7 + dow)].copy()

                    data_1.loc[:, 'week'] = t_0 + it
                    data_1.loc[:, 'time_window'] = dow

                    data_new0[index] = pd.concat([data_new0[index], data_1])

                    data_RO_ele = data_0.sample(n=eta0_RO, replace = True).copy()

                    data_RO_ele.loc[:, 'week'] = t_0 + it
                    data_RO_ele.loc[:, 'time_window'] = dow

                    data_new0[index] = pd.concat([data_new0[index], data_RO_ele])

                    BOR0_RO.append(len(
                        data_new0[index].loc[(data_new0[index].time_window <= dow) &
                                             (data_new0[index].los + data_new0[index].time_window >= dow + 1)]) - c)

                result_BOR_RO.append(BOR0_RO)
                total_BOR_RO = sum([max(item, 0) for item in BOR0_RO])
                final_BOR_RO.append(total_BOR_RO)

                data_new0[index].time_window = data_new0[index].time_window - 7
                data_new0[index].time_window = data_new0[index].time_window.astype(int)

            # data_new00.time_window = data_new00.time_window - 7
            # data_new00.time_window = data_new00.time_window.astype(int)

    return [index,total_eta_RO,data_new0[index],result_BOR_RO,final_BOR_RO]


def solve_RS(index):
    with Env() as env, Model(env=env) as model:

        input_data = parameter_RS[index]
        T = input_data[0]
        c = input_data[1]
        min_weekquota = input_data[2]
        eta_equal = int(min_weekquota / 7)

        #print(T,T_plus)
        T_minus = list(range(-L + 1, 0))
        T_plus = list(range(T))
        T_list = list(range(-L + 1, T))

        L_list = []
        for t in T_list:
            L_list_t = list(range(max(1, 1 - t), min(L, T - t) + 1))
            L_list.append(L_list_t)

        J_list_aux = []
        for t in T_list:
            J_list_t_aux = []
            for i in L_list[t + L - 1]:
                J_list_t_aux.append((t, i))
            J_list_aux.append(J_list_t_aux)

        J_list = []
        for t in T_list:
            for i in L_list[t + L - 1]:
                J_list.append((t, i))

        L_list_plus = []
        for t in T_list:
            L_list_t = list(range(max(1, 1 - t), min(L, T - t) + 2))
            L_list_plus.append(L_list_t)

        J_list_plus = []
        for t in T_list:
            for i in L_list_plus[t + L - 1]:
                J_list_plus.append((t, i))

        z0 = np.zeros((T, T + L - 1, L))
        for t in range(T):
            for tau, l in product(range(T + L - 1), range(L)):
                if tau - L + 1 + l == t:
                    z0[t, tau, l] = 1

        LOS = list(range(1, L + 1))
        LOS_plus = list(range(1, L + 2))

        mul = int((T + L) // 7)
        min_week_number = int(7-mul)
        alpha_plus = np.ones(T)

        S_list = list(range(S))
        omega = np.ones(T)
        time_min = min(data_new[0].time_window)

        p = np.zeros((S, T + L - 1, L + 1))
        a = np.zeros((S, T + L - 1, L + 1))
        alpha = np.zeros((S, T + L - 1, L + 1))
        # eta_0 = np.zeros((S, T))

        eta_0 = np.zeros((S, T))

        for s in S_list:
            time_current = time_min + (s + min_week_number) * 7

            # print(time_current)
            for tau, l in J_list:
                a[s, tau + L - 1, l - 1] = len(data_new[index].loc[(data_new[index].admission_type == 'elective') &
                                                               (data_new[index].time_window == time_current + tau) & (
                                                                       data_new[index].los >= l)])
                p[s, tau + L - 1, l - 1] = len(data_new[index].loc[(data_new[index].admission_type == 'emergency') &
                                                               (data_new[index].time_window == time_current + tau) & (
                                                                       data_new[index].los >= l)])

                alpha[s, tau + L - 1, l - 1] = a[s, tau + L - 1, l - 1] / len(
                    data_new[index].loc[(data_new[index].admission_type == 'elective') &
                                    (data_new[index].time_window == time_current + tau)])

        result_BOR_RS, final_BOR_RS, total_eta_RS, result_BOR_equal, final_BOR_equal = [], [], [], [], []

        for it in range(IT):

            eta_minus, p_minus, alpha_minus = -np.ones(L - 1), -np.ones(L - 1), -np.ones(L - 1)

            for t in T_minus:
                # eta_minus[t + L - 1] = len(
                #     data_new.loc[(data_new['time_window'] == t) & (data_new['admission_type'] == 'elective')])
                p_minus[t + L - 1] = len(
                    data_new[index].loc[
                        (data_new[index].time_window == t) & (data_new[index].admission_type == 'emergency') &
                        (data_new[index].los + data_new[index].time_window >= 0)])

                eta_minus[t + L - 1] = len(
                    data_new[index].loc[
                        (data_new[index].time_window == t) & (data_new[index].admission_type == 'elective')])

                alpha_minus[t + L - 1] = len(
                    data_new[index].loc[
                        (data_new[index].time_window == t) & (data_new[index].admission_type == 'elective') &
                        (data_new[index].los + data_new[index].time_window >= 0)]) / eta_minus[t + L - 1]

            for s in S_list:
                time_current = time_min + (s + min_week_number) * 7

                for tao in T_plus:
                    eta_0[s, tao] = len(data_new[index].loc[(data_new[index].admission_type == 'elective') &
                                                            (data_new[index].time_window == time_current + tao)])

            eta_bar = np.mean(eta_0, axis=0)

        #     print('iter:',it,'\n','p0_bar:',p0_bar,'alpha0_bar:',alpha0_bar,'eta_bar:',eta_bar)

        ############################################################### RO

            model.Params.MIPFocus = 0
            model.Params.Cuts = -1
            model.Params.timeLimit = 150

            # k = model.addVars(T_plus, lb=0, vtype="C")
            k = model.addVar(lb=0, vtype="C")

            phi = model.addVars(S_list, T_plus, lb=-c, vtype="C")

            eta = model.addVars(T_plus, lb=5, ub=80, vtype="C")

            # k_aux = model.addVars(T_plus, T_list, lb=0, vtype="C")
            # xi = model.addVars(S_list, T_plus, lb=0, vtype="C")

            # some larger than 0
            xi_alpha = model.addVars(S_list, T_plus, T_list, LOS_plus, lb=0, vtype="C")
            xi_p = model.addVars(S_list, T_plus, T_list, LOS_plus, lb=0, vtype="C")

            # model.update()

            ### constraint
            for i in range(T // 7):
                model.addConstr(quicksum(eta[t] for t in range(7 * i, 7 * (i + 1))) == min_weekquota)

            for t in T_plus:

                model.addConstr(quicksum(phi[s, t] for s in S_list) / S <= 0.95 * min_weekquota)

                for s in S_list:

                    model.addConstr(quicksum(
                        (eta_minus[tau + L - 1] * alpha[s, tau + L - 1, l - 1] + p[s, tau + L - 1, l - 1]) * z0[
                            t, tau + L - 1, l - 1] for tau, l in J_list if tau < 0) +
                                    quicksum((eta[tau] * alpha[s, tau + L - 1, l - 1] + p[s, tau + L - 1, l - 1]) * z0[
                                        t, tau + L - 1, l - 1] for tau, l in J_list if tau >= 0) +
                                    quicksum(xi_alpha[s, t, tau, 1 - tau] * (
                                            alpha_minus[tau + L - 1] - alpha[s, tau + L - 1, -tau]) +
                                             xi_p[s, t, tau, 1 - tau] * (p_minus[tau + L - 1] - p[s, tau + L - 1, -tau]) for
                                             tau
                                             in T_minus) +
                                    quicksum(xi_alpha[s, t, tau, 1] * (alpha_plus[tau % 7] - alpha[s, tau + L - 1, 0]) +
                                             xi_p[s, t, tau, 1] * (max_p0_count[tau % 7] - p[s, tau + L - 1, 0]) for tau in
                                             T_plus) +
                                    quicksum(
                                        xi_alpha[s, t, tau, l] * (alpha[s, tau + L - 1, l - 1] - alpha[s, tau + L - 1, l]) +
                                        xi_p[s, t, tau, l] * (p[s, tau + L - 1, l - 1] - p[s, tau + L - 1, l]) for tau, l in
                                        J_list) <= phi[s, t])

                    for tau, l in J_list:

                        model.addConstr(z0[t, tau + L - 1, l - 1] + xi_p[s, t, tau, l + 1] - xi_p[s, t, tau, l] >= -k)
                        model.addConstr(z0[t, tau + L - 1, l - 1] + xi_p[s, t, tau, l + 1] - xi_p[s, t, tau, l] <= k)

                        if tau < 0:

                            model.addConstr(
                                eta_minus[tau + L - 1] * z0[t, tau + L - 1, l - 1] + xi_alpha[s, t, tau, l + 1] - xi_alpha[
                                    s, t, tau, l] >= -k * eta_minus[tau + L - 1])
                            model.addConstr(
                                eta_minus[tau + L - 1] * z0[t, tau + L - 1, l - 1] + xi_alpha[s, t, tau, l + 1] - xi_alpha[
                                    s, t, tau, l] <= k * eta_minus[tau + L - 1])

                        else:

                            # model.addConstr(k_aux[t,tau]==k[t]*eta[tau])

                            model.addConstr(
                                eta[tau] * z0[t, tau + L - 1, l - 1] + xi_alpha[s, t, tau, l + 1] - xi_alpha[
                                    s, t, tau, l] >= -
                                k * eta_bar[tau])
                            model.addConstr(
                                eta[tau] * z0[t, tau + L - 1, l - 1] + xi_alpha[s, t, tau, l + 1] - xi_alpha[
                                    s, t, tau, l] <=
                                k * eta_bar[tau])

            # model.setObjective(quicksum(omega[t] * k for t in T_plus), GRB.MINIMIZE)
            model.setObjective(k, GRB.MINIMIZE)

            model.optimize()
            status = model.status

            # retrieve data from model
            try:
                eta_sol = [round(i.X) for i in eta.values()]
            except:
                eta_sol = [eta_equal] * T

            total_eta_RS.append(eta_sol)

            # try:
            #     phi_sol = [i.X for i in phi.values()]
            # except:
            #     phi_sol = []

            #eta_RS.append(eta_sol)
            # phi_sol0.append(phi_sol)

            # eta_RO[index].append(eta_sol0)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=(
                        ".*will attempt to set the values inplace instead of always setting a new array. "
                        "To retain the old behavior, use either.*"
                    ),
                )

                # data_RO = pd.DataFrame(columns=data.columns)

                BOR0_RS, BOR0_RO, BOR0_equal = [], [], []
                for dow in range(7):
                    eta0_RS = round(eta_sol[dow])

                    data_0 = data.loc[(data.admission_type == 'elective') & (data.admitday == dow)].copy()
                    data_1 = data.loc[(data.admission_type == 'emergency') & (data.time_window == it * 7 + dow)].copy()

                    data_1.loc[:, 'week'] = t_0 + it
                    data_1.loc[:, 'time_window'] = dow

                    data_new[index] = pd.concat([data_new[index], data_1])
                    # data_new = pd.concat([data_new, data_1])
                    data_new00[index] = pd.concat([data_new00[index], data_1])

                    # data_RO_ele = data_0.sample(n=eta0_RO, replace = True).copy()
                    data_RS_ele = data_0.sample(n=eta0_RS, replace=True).copy()
                    data_equal_ele = data_0.sample(n=eta_equal, replace=True).copy()

                    # data_RO_ele.loc[:, 'week'] = t_0 + it
                    # data_RO_ele.loc[:, 'time_window'] = dow

                    data_RS_ele.loc[:, 'week'] = t_0 + it
                    data_RS_ele.loc[:, 'time_window'] = dow

                    data_equal_ele.loc[:, 'week'] = t_0 + it
                    data_equal_ele.loc[:, 'time_window'] = dow

                    # data_new0[index] = pd.concat([data_new0[index], data_RO_ele])
                    data_new[index] = pd.concat([data_new[index], data_RS_ele])
                    data_new00[index] = pd.concat([data_new00[index], data_equal_ele])

                    BOR0_RS.append(len(
                        data_new[index].loc[(data_new[index].time_window <= dow) & (data_new[index].los + data_new[index].time_window >= dow + 1)]) - c)

                    # BOR0_RO.append(len(
                    #     data_new0[index].loc[(data_new0[index].time_window <= dow) & (data_new0[index].los + data_new0[index].time_window >= dow + 1)]) - c)

                    BOR0_equal.append(len(data_new00[index].loc[(data_new00[index].time_window <= dow) &
                                                         (data_new00[index].los + data_new00[index].time_window >= dow + 1)]) - c)

                # BOR_RO[index].append(BOR0_RO)
                result_BOR_RS.append(BOR0_RS)
                result_BOR_equal.append(BOR0_equal)
                # total_BOR_RO = sum([max(item, 0) for item in BOR0_RO])
                total_BOR_RS = sum([max(item, 0) for item in BOR0_RS])
                total_BOR_equal = sum([max(item, 0) for item in BOR0_equal])

                final_BOR_RS.append(total_BOR_RS)
                final_BOR_equal.append(total_BOR_equal)

                data_new[index].time_window = data_new[index].time_window - 7
                data_new[index].time_window = data_new[index].time_window.astype(int)

                data_new00[index].time_window = data_new00[index].time_window - 7
                data_new00[index].time_window = data_new00[index].time_window.astype(int)

    return [index,total_eta_RS,data_new[index],result_BOR_RS,final_BOR_RS,data_new00[index],result_BOR_equal,final_BOR_equal]

#c_list=[600,620,650]
#w_list=[201,245,301]
#mu_list=[0,0.01,0.05,0.1]

TT_list=[14,21]
c_list=[600,620]
w_list=[245,301]
mu_list=[0,0.01]

parameter0_RO=[TT_list,c_list,w_list,mu_list]
parameter0_RS=[TT_list,c_list,w_list]

parameter_RO = [element for element in product(*parameter0_RO)]
parameter_RS = [element for element in product(*parameter0_RS)]

I_RO = len(parameter_RO)
I_RS = len(parameter_RS)
# I_equal = len(parameter_equal)
max_p0_count = 200*np.ones(7)

eta_RS, eta_RO, obj_RS, obj_RO = [[] for i in range(I_RS)], [[] for i in range(I_RO)], [[] for i in range(I_RS)], \
                                 [[] for i in range(I_RO)]

data_new, data_new0, data_new00 = [data.loc[(data.week < t_0)].copy() for i in range(I_RS)], \
                                  [data.loc[(data.week < t_0)].copy() for i in range(I_RO)], \
                                  [data.loc[(data.week < t_0)].copy() for i in range(I_RS)]

BOR_RS, BOR_RO, BOR_equal = [[] for i in range(I_RS)], [[] for i in range(I_RO)], [[] for i in range(I_RS)]
total_BOR_RS,total_BOR_RO,total_BOR_equal = [[] for i in range(I_RS)], [[] for i in range(I_RO)], [[] for i in range(I_RS)]

if __name__ == '__main__':

    with mp.Pool(32) as pool:
        solution_RO = pool.map(solve_RO, list(range(I_RO)))
        solution_RS = pool.map(solve_RS, list(range(I_RS)))

    for line in solution_RO:
        i0 = int(line[0])

        eta_RO[i0].append(line[1])
        data_new0[i0]=line[2]
        BOR_RO[i0].append(line[3])
        total_BOR_RO[i0].append(line[4])

        data_new0[i0].to_csv('data_RO_'+str(i0)+'.csv')

    for line in solution_RS:
        i0 = int(line[0])

        eta_RS[i0].append(line[1])
        data_new[i0] = line[2]
        BOR_RS[i0].append(line[3])
        total_BOR_RS[i0].append(line[4])
        data_new00[i0] = line[5]
        BOR_equal[i0].append(line[6])
        total_BOR_equal[i0].append(line[7])

        data_new[i0].to_csv('data_RS_'+str(i0)+'.csv')

        data_new00[i0].to_csv('data_equal_' + str(i0) + '.csv')


    df_eta_RO=pd.DataFrame(eta_RO)
    df_eta_RO.to_csv('eta_RO.csv')

    df_eta_RS=pd.DataFrame(eta_RS)
    df_eta_RS.to_csv('eta_RS.csv')

    df_BOR_RO=pd.DataFrame(BOR_RO)
    df_BOR_RO.to_csv('BOR_RO.csv')

    df_BOR_RS=pd.DataFrame(BOR_RS)
    df_BOR_RS.to_csv('BOR_RS.csv')

    df_BOR_equal=pd.DataFrame(BOR_equal)
    df_BOR_equal.to_csv('BOR_equal.csv')

    df_total_BOR_RO=pd.DataFrame(total_BOR_RO)
    df_total_BOR_RO.to_csv('total_BOR_RO.csv')

    df_total_BOR_RS=pd.DataFrame(total_BOR_RS)
    df_total_BOR_RS.to_csv('total_BOR_RS.csv')

    df_total_BOR_equal=pd.DataFrame(total_BOR_equal)
    df_total_BOR_equal.to_csv('total_BOR_equal.csv')

