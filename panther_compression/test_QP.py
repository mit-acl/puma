
#!/usr/bin/env python

"""
Test QP constraints
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def plot_results(q, q_star, v, v_star, n):

    # subfigure 1: position
    fig, axes = plt.subplots(1, 3, sharex=True)
    axes[0].plot(np.linspace(0, n+1, n+1), q[0::3], 'b', label='x')
    axes[0].plot(np.linspace(0, n+1, n+1), q[1::3], 'r', label='y')
    axes[0].plot(np.linspace(0, n+1, n+1), q[2::3], 'g', label='z')
    axes[0].plot(np.linspace(0, n+1, n+1), q_star[0::3], 'b--', label='x*')
    axes[0].plot(np.linspace(0, n+1, n+1), q_star[1::3], 'r--', label='y*')
    axes[0].plot(np.linspace(0, n+1, n+1), q_star[2::3], 'g--', label='z*')
    axes[0].legend()

    # subfigure 2: velocity
    axes[1].plot(np.linspace(0, n, n), v[0::3], 'b', label='vx')
    axes[1].plot(np.linspace(0, n, n), v[1::3], 'r', label='vy')
    axes[1].plot(np.linspace(0, n, n), v[2::3], 'g', label='vz')
    axes[1].plot(np.linspace(0, n, n), v_star[0::3], 'b--', label='vx*')
    axes[1].plot(np.linspace(0, n, n), v_star[1::3], 'r--', label='vy*')
    axes[1].plot(np.linspace(0, n, n), v_star[2::3], 'g--', label='vz*')
    axes[1].legend()

    plt.show()

def get_vel_constr(p, n, knots, v_max=2):

    A_v = np.zeros((3*n, 3*(n+1)))
    for i in range(0, 3*n, 3):
        alpha_v_i = p / (knots[i//3+p+1] - knots[i//3+1])
        # x axis constraint
        A_v[i, i] = -alpha_v_i; A_v[i, i+3] = alpha_v_i
        # y axis constraint
        A_v[i+1, i+1] = -alpha_v_i; A_v[i+1, i+4] = alpha_v_i
        # z axis constraint
        A_v[i+2, i+2] = -alpha_v_i; A_v[i+2, i+5] = alpha_v_i

    b_v = np.repeat(v_max, 3*n)

    return A_v, b_v

def get_acc_constr_wrt_q(p, n, knots, a_max):

    A_a = np.zeros((3*(n-1), 3*(n+1)))
    for i in range(0, 3*(n-1), 3):
        alpha_a_1 = p*(p-1) / (knots[i//3+p+1] - knots[i//3+2])
        alpha_a_2 = 1 / (knots[i//3+p+2] - knots[i//3+2])
        alpha_a_3 = (knots[i//3+p+1] - knots[i//3+1] + knots[i//3+p+2] - knots[i//3+2]) / ( (knots[i//3+p+2] - knots[i//3+2]) * (knots[i//3+p+1] - knots[i//3+1]) )
        alpha_a_4 = 1 / (knots[i//3+p+1] - knots[i//3+1])
        # x axis constraint
        A_a[i, i] = alpha_a_1 * alpha_a_4; A_a[i, i+3] = -alpha_a_1 * alpha_a_3; A_a[i, i+6] = alpha_a_1 * alpha_a_2
        # y axis constraint
        A_a[i+1, i+1] = alpha_a_1 * alpha_a_4; A_a[i+1, i+4] = -alpha_a_1 * alpha_a_3; A_a[i+1, i+7] = alpha_a_1 * alpha_a_2
        # z axis constraint
        A_a[i+2, i+2] = alpha_a_1 * alpha_a_4; A_a[i+2, i+5] = -alpha_a_1 * alpha_a_3; A_a[i+2, i+8] = alpha_a_1 * alpha_a_2
    
    b_a = np.repeat(a_max, 3*(n-1))

    return A_a, b_a

def get_jerk_constr_wrt_q(p, n, knots, j_max):

    A_j = np.zeros((3*(n-2), 3*(n+1)))
    for i in range(0, 3*(n-2), 3):
        alpha_j_1 = p*(p-1)*(p-2) / ( (knots[i//3+p+1] - knots[i//3+3]) * (knots[i//3+p+1] - knots[i//3+2]) )
        alpha_j_2 = ( knots[i//3+p+1] - knots[i//3+2]) / ( (knots[i//3+p+2] - knots[i//3+3]) * (knots[i//3+p+3] - knots[i//3+3]) )
        alpha_j_3 = ( (knots[i//3+p+1] - knots[i//3+2]) * (knots[i//3+p+2] - knots[i//3+2]) * (knots[i//3+p+2] - knots[i//3+2] + knots[i//3+p+3] - knots[i//3+3]) + \
                        (knots[i//3+p+2] - knots[i//3+2]) * (knots[i//3+p+2] - knots[i//3+3]) * (knots[i//3+p+3] - knots[i//3+3]) ) * \
                        1 / ( (knots[i//3+p+2] - knots[i//3+2])**2 * (knots[i//3+p+2] - knots[i//3+3]) * (knots[i//3+p+3] - knots[i//3+3]) )
        alpha_j_4 = ( (knots[i//3+p+1] - knots[i//3+1]) * (knots[i//3+p+1] - knots[i//3+2]) + (knots[i//3+p+2] - knots[i//3+3]) * (knots[i//3+p+1] - knots[i//3+1] + knots[i//3+p+2] - knots[i//3+2]) ) * \
                        1 / ( (knots[i//3+p+1] - knots[i//3+1]) * (knots[i//3+p+2] - knots[i//3+2]) * (knots[i//3+p+2] - knots[i//3+3]) )
        alpha_j_5 = 1 / (knots[i//3+p+1] - knots[i//3+1])
        # x axis constraint
        A_j[i, i] = -alpha_j_1 * alpha_j_5; A_j[i, i+3] = alpha_j_1 * alpha_j_4; A_j[i, i+6] = -alpha_j_1 * alpha_j_3; A_j[i, i+9] = alpha_j_1 * alpha_j_2
        # y axis constraint
        A_j[i+1, i+1] = -alpha_j_1 * alpha_j_5; A_j[i+1, i+4] = alpha_j_1 * alpha_j_4; A_j[i+1, i+7] = -alpha_j_1 * alpha_j_3; A_j[i+1, i+10] = alpha_j_1 * alpha_j_2
        # z axis constraint
        A_j[i+2, i+2] = -alpha_j_1 * alpha_j_5; A_j[i+2, i+5] = alpha_j_1 * alpha_j_4; A_j[i+2, i+8] = -alpha_j_1 * alpha_j_3; A_j[i+2, i+11] = alpha_j_1 * alpha_j_2
    
    b_j = np.repeat(j_max, 3*(n-2))

    return A_j, b_j

def get_acc_constr_wrt_v(p, n, knots, a_max):

    A_a = np.zeros((3*(n-1), 3*n))
    for i in range(0, 3*(n-1), 3):
        alpha = (p-1) / (knots[i//3+p+1] - knots[i//3+2])
        # x axis constraint
        A_a[i, i] = -alpha; A_a[i, i+3] = alpha
        # y axis constraint
        A_a[i+1, i+1] = -alpha; A_a[i+1, i+4] = alpha
        # z axis constraint
        A_a[i+2, i+2] = -alpha; A_a[i+2, i+5] = alpha
    
    b_a = np.repeat(a_max, 3*(n-1))

    return A_a, b_a

def get_subset_vel(A_v, b_v, p, knots, v_max, q0):
    
    """
    This function is used to get the subset of velocity constraints for the p-th control point
    """

    A_v_sub = A_v[3*(p-1):, 3*p:]
    b_v_sub = b_v[3*p:]

    # add velocity inequality constraints from q_p to q_{p+1}
    alpha_v_2 = p / (knots[2//3+p+1] - knots[2//3+1]) # v2 is the first control point we have control over
    b_v_sub_prime = np.zeros(3)
    b_v_sub_prime[0] = v_max + alpha_v_2 * q0[6]
    b_v_sub_prime[1] = v_max + alpha_v_2 * q0[7]
    b_v_sub_prime[2] = v_max + alpha_v_2 * q0[8]
    b_v_sub = np.hstack((b_v_sub_prime, b_v_sub))

    return A_v_sub, b_v_sub

def get_subset_acc(A_a, b_a, p, knots, a_max, q0):

    A_a_sub = A_a[3*(p-2):, 3*p:]
    b_a_sub = b_a[3*p:]

    # add acceleration inequality constraints from q_{p-1} and q_p to q_{p+1}

    # used to constraint q_{1}
    alpha_a_1_1 = p*(p-1) / (knots[1//3+p+1] - knots[1//3+2]) # a1 is the first control point we have control over
    alpha_a_2_1 = 1 / (knots[1//3+p+2] - knots[1//3+2])
    alpha_a_3_1 = (knots[1//3+p+1] - knots[1//3+1] + knots[1//3+p+2] - knots[1//3+2]) / ( (knots[1//3+p+2] - knots[1//3+2]) * (knots[1//3+p+1] - knots[1//3+1]) )
    alpha_a_4_1 = 1 / (knots[1//3+p+1] - knots[1//3+1])

    # used to constraint q_{2}
    alpha_a_1_2 = p*(p-1) / (knots[2//3+p+1] - knots[2//3+2]) 
    alpha_a_2_2 = 1 / (knots[2//3+p+2] - knots[2//3+2])
    alpha_a_3_2 = (knots[2//3+p+1] - knots[2//3+1] + knots[2//3+p+2] - knots[2//3+2]) / ( (knots[2//3+p+2] - knots[2//3+2]) * (knots[2//3+p+1] - knots[2//3+1]) )
    alpha_a_4_2 = 1 / (knots[2//3+p+1] - knots[2//3+1])

    b_a_sub_prime = np.zeros(6)
    # constraint q_{3}
    b_a_sub_prime[0] = a_max + alpha_a_1_1 * alpha_a_3_1 * q0[6] - alpha_a_1_1 * alpha_a_4_1 * q0[3]
    b_a_sub_prime[1] = a_max + alpha_a_1_1 * alpha_a_3_1 * q0[7] - alpha_a_1_1 * alpha_a_4_1 * q0[4]
    b_a_sub_prime[2] = a_max + alpha_a_1_1 * alpha_a_3_1 * q0[8] - alpha_a_1_1 * alpha_a_4_1 * q0[5]
    # constraint q_{4}
    b_a_sub_prime[3] = a_max - alpha_a_1_2 * alpha_a_4_2 * q0[6]
    b_a_sub_prime[4] = a_max - alpha_a_1_2 * alpha_a_4_2 * q0[7]
    b_a_sub_prime[5] = a_max - alpha_a_1_2 * alpha_a_4_2 * q0[8]

    b_a_sub = np.hstack((b_a_sub_prime, b_a_sub))

    return A_a_sub, b_a_sub

def get_subset_jerk(A_j, b_j, p, knots, j_max, q):

    """
    This function is used to get the subset of jerk constraints for the p-th control point
    """

    A_j_sub = A_j[3*(p-3):, 3*p:]
    b_j_sub = b_j[3*p:]

    # add jerk inequality constraints from q_{p-2}, q_{p-1} and q_p to q_{p+1}

    # used to constraint q_{0} to q_{2}
    alpha_j = np.zeros((3, 5))
    for i in range(0, 3):
        alpha_j[i][4] = p*(p-1)*(p-2) / ( (knots[i//3+p+1] - knots[i//3+3]) * (knots[i//3+p+1] - knots[i//3+2]) )                               # constanst before everything
        alpha_j[i][3] = ( knots[i//3+p+1] - knots[i//3+2]) / ( (knots[i//3+p+2] - knots[i//3+3]) * (knots[i//3+p+3] - knots[i//3+3]) )          # constant before P_{i+3} # refer to the paper
        alpha_j[i][2] = ( (knots[i//3+p+1] - knots[i//3+2]) * (knots[i//3+p+2] - knots[i//3+2]) * (knots[i//3+p+2] - knots[i//3+2] + knots[i//3+p+3] - knots[i//3+3]) + \
                        (knots[i//3+p+2] - knots[i//3+2]) * (knots[i//3+p+2] - knots[i//3+3]) * (knots[i//3+p+3] - knots[i//3+3]) ) * \
                        1 / ( (knots[i//3+p+2] - knots[i//3+2])**2 * (knots[i//3+p+2] - knots[i//3+3]) * (knots[i//3+p+3] - knots[i//3+3]) )    # constant before P_{i+2} # refer to the paper
        alpha_j[i][1] = ( (knots[i//3+p+1] - knots[i//3+1]) * (knots[i//3+p+1] - knots[i//3+2]) + (knots[i//3+p+2] - knots[i//3+3]) * (knots[i//3+p+1] - knots[i//3+1] + knots[i//3+p+2] - knots[i//3+2]) ) * \
                        1 / ( (knots[i//3+p+1] - knots[i//3+1]) * (knots[i//3+p+2] - knots[i//3+2]) * (knots[i//3+p+2] - knots[i//3+3]) )       # constant before P_{i+1} # refer to the paper
        alpha_j[i][0] = 1 / (knots[i//3+p+1] - knots[i//3+1])                                                                                   # constant before P_{i}   # refer to the paper

    b_j_sub_prime = np.zeros(9)

    # constraint q_{3} (depends on q_{0}, q_{1}, q_{2}, where q_{0} = q[0:3], q_{1} = q[3:6], q_{2} = q[6:9])
    b_j_sub_prime[0] = j_max + alpha_j[0][4] * alpha_j[0][2] * q[6] - alpha_j[0][4] * alpha_j[0][1] * q[3] + alpha_j[0][4] * alpha_j[0][0] * q[0] #x
    b_j_sub_prime[1] = j_max + alpha_j[0][4] * alpha_j[0][2] * q[7] - alpha_j[0][4] * alpha_j[0][1] * q[4] + alpha_j[0][4] * alpha_j[0][0] * q[1] #y
    b_j_sub_prime[2] = j_max + alpha_j[0][4] * alpha_j[0][2] * q[8] - alpha_j[0][4] * alpha_j[0][1] * q[5] + alpha_j[0][4] * alpha_j[0][0] * q[2] #z

    # constraint q_{4} (depends on q_{1}, q_{2}, (and q_{3} but this is included in A_j_sub))
    b_j_sub_prime[3] = j_max - alpha_j[1][4] * alpha_j[1][1] * q[6] + alpha_j[1][4] * alpha_j[1][0] * q[3] #x
    b_j_sub_prime[4] = j_max - alpha_j[1][4] * alpha_j[1][1] * q[7] + alpha_j[1][4] * alpha_j[1][0] * q[4] #y
    b_j_sub_prime[5] = j_max - alpha_j[1][4] * alpha_j[1][1] * q[8] + alpha_j[1][4] * alpha_j[1][0] * q[5] #z

    # constraint q_{5} (depends on q_{2}, (and q_{3}, q_{4} but this is included in A_j_sub))
    b_j_sub_prime[6] = j_max + alpha_j[2][4] * alpha_j[2][0] * q[6] #x
    b_j_sub_prime[7] = j_max + alpha_j[2][4] * alpha_j[2][0] * q[7] #y
    b_j_sub_prime[8] = j_max + alpha_j[2][4] * alpha_j[2][0] * q[8] #z

    b_j_sub = np.hstack((b_j_sub_prime, b_j_sub))

    return A_j_sub, b_j_sub

def solve_QP(p, n, knots, v_max, a_max, j_max, q0):

    """
    Using scipy.optimize.minimize
    """

    # Get velocity constraint matrix A_v and vector b_v
    A_v, b_v = get_vel_constr(p, n, knots, v_max)

    # optimize only a subset of the control points
    q_sub = q0[3*p:]
    A_v_sub, b_v_sub = get_subset_vel(A_v, b_v, p, knots, v_max, q0)

    # Get acceleration constraint matrix A_a and vector b_a
    A_a, b_a = get_acc_constr_wrt_q(p, n, knots, a_max)

    # optimize only a subset of the control points
    q_sub = q0[3*p:]
    A_a_sub, b_a_sub = get_subset_acc(A_a, b_a, p, knots, a_max, q0)

    # Get jerk constraint matrix A_j and vector b_j
    A_j, b_j = get_jerk_constr_wrt_q(p, n, knots, j_max)
    A_j_sub, b_j_sub = get_subset_jerk(A_j, b_j, p, knots, j_max, q0)

    # Get q, A, and b
    q_orig = q0
    x = q_sub
    # A = np.vstack((A_v_sub, -A_v_sub, A_a_sub, -A_a_sub, A_j_sub, -A_j_sub))
    # b = np.hstack((b_v_sub, b_v_sub, b_a_sub, b_a_sub, b_j_sub, b_j_sub))

    A = np.vstack((A_v_sub, -A_v_sub, A_a_sub, -A_a_sub, A_j_sub, -A_j_sub))
    b = np.hstack((b_v_sub, b_v_sub, b_a_sub, b_a_sub, b_j_sub, b_j_sub))

    # Get H, c, and c0 (ref: https://stackoverflow.com/questions/17009774/quadratic-program-qp-solver-that-only-depends-on-numpy-scipy)
    H = np.eye(x.shape[0])
    c = - x.copy()
    c0 = 1/2 * x.T @ x
    x0 = q_orig[3*p:]

    def loss(x, sign=1.):
        return sign * (0.5 * np.dot(x.T, np.dot(H, x))+ np.dot(c, x) + c0)

    def jac(x, sign=1.):
        return sign * (np.dot(x.T, H) + c)
    
    ineq_cons = {'type':'ineq',
            'fun':lambda x: b - np.dot(A,x),
            'jac':lambda x: -A}

    # eq_cons = {'type':'eq',
    #         'fun':lambda x: q_orig[-3:] - x[-3:],
    #         'jac':lambda x: -np.eye(3, x.shape[0])}

    # cons = ([ineq_cons, eq_cons])

    cons = ([ineq_cons])

    opt = {'disp':False}

    res_cons = optimize.minimize(loss, x0, jac=jac,constraints=cons,
                                    method='SLSQP', options=opt)
    
    print("success", res_cons['success'])
    
    return res_cons['x']

def get_vel_star_wrt_q(p, n, knots, v_max, q):

    """
    This funciton is used to get the optimized velocity and acceleration spline at the same time for the minimum-deviation q trajectory
    (Not works well)
    
    """

    # Get velocity constraint matrix A_v and vector b_v
    A_v, b_v = get_vel_constr(p, n, knots, v_max)

    # optimize only a subset of the control points
    q_sub = q[3*p:]
    A_v_sub = A_v[3*(p-1):, 3*p:]
    b_v_sub = b_v[3*p:]

    # add velocity inequality constraints from q_p to q_{p+1}
    alpha_v_2 = p / (knots[2//3+p+1] - knots[2//3+1]) # v2 is the first control point we have control over
    b_v_sub_prime = np.zeros(3)
    b_v_sub_prime[0] = v_max + alpha_v_2 * q[6]
    b_v_sub_prime[1] = v_max + alpha_v_2 * q[7]
    b_v_sub_prime[2] = v_max + alpha_v_2 * q[8]
    b_v_sub = np.hstack((b_v_sub_prime, b_v_sub))

    # Get q, A, and b
    q_orig = q
    q = q_sub
    A = A_v_sub
    b = b_v_sub

    # Get the closed-form solution
    q_star = q + A.T @ np.linalg.inv(- A @ A.T) @ np.maximum(np.zeros(A.shape[0]), A @ q - b) # optimizzed all control points

    # Get velocity spline
    v = A_v @ q_orig
    v_star = A_v_sub @ q_star

    q_star = np.hstack((q_orig[0:3*p], q_star)) # concatenate the optimized control points with the rest
    v_star = np.hstack((v[0:3*(p-1)], v_star)) # concatenate the optimized control points with the rest

    return q_star, v_star, v

def get_accel_star_wrt_q(p, n, knots, a_max, q):

    """
    This funciton is used to get the optimized velocity and acceleration spline at the same time for the minimum-deviation q trajectory
    (Not works well)
    
    """

    # Get acceleration constraint matrix A_a and vector b_a
    A_a, b_a = get_acc_constr_wrt_q(p, n, knots, a_max)

    # optimize only a subset of the control points
    q_sub = q[3*p:]
    A_a_sub = A_a[3*(p-2):, 3*p:]
    b_a_sub = b_a[3*p:]

    # add acceleration inequality constraints from q_{p-1} and q_p to q_{p+1}

    # used to constraint q_{3}
    alpha_a_1_1 = p*(p-1) / (knots[1//3+p+1] - knots[1//3+2]) # a1 is the first control point we have control over
    alpha_a_2_1 = 1 / (knots[1//3+p+2] - knots[1//3+2])
    alpha_a_3_1 = (knots[1//3+p+1] - knots[1//3+1] + knots[1//3+p+2] - knots[1//3+2]) / ( (knots[1//3+p+2] - knots[1//3+2]) * (knots[1//3+p+1] - knots[1//3+1]) )
    alpha_a_4_1 = 1 / (knots[1//3+p+1] - knots[1//3+1])

    # used to constraint q_{4}
    alpha_a_1_2 = p*(p-1) / (knots[2//3+p+1] - knots[2//3+2]) 
    alpha_a_2_2 = 1 / (knots[2//3+p+2] - knots[2//3+2])
    alpha_a_3_2 = (knots[2//3+p+1] - knots[2//3+1] + knots[2//3+p+2] - knots[2//3+2]) / ( (knots[2//3+p+2] - knots[2//3+2]) * (knots[2//3+p+1] - knots[2//3+1]) )
    alpha_a_4_2 = 1 / (knots[2//3+p+1] - knots[2//3+1])

    b_a_sub_prime = np.zeros(6)
    # constraint q_{3}
    b_a_sub_prime[0] = a_max + alpha_a_1_1 * alpha_a_3_1 * q[6] - alpha_a_1_1 * alpha_a_4_1 * q[3]
    b_a_sub_prime[1] = a_max + alpha_a_1_1 * alpha_a_3_1 * q[7] - alpha_a_1_1 * alpha_a_4_1 * q[4]
    b_a_sub_prime[2] = a_max + alpha_a_1_1 * alpha_a_3_1 * q[8] - alpha_a_1_1 * alpha_a_4_1 * q[5]
    # constraint q_{4}
    b_a_sub_prime[3] = a_max - alpha_a_1_2 * alpha_a_4_2 * q[6]
    b_a_sub_prime[4] = a_max - alpha_a_1_2 * alpha_a_4_2 * q[7]
    b_a_sub_prime[5] = a_max - alpha_a_1_2 * alpha_a_4_2 * q[8]

    b_a_sub = np.hstack((b_a_sub_prime, b_a_sub))

    # Get q, A, and b
    q_orig = q
    q = q_sub
    A = A_a_sub
    b = b_a_sub

    # Get the closed-form solution
    q_star = q + A.T @ np.linalg.inv(- A @ A.T) @ np.maximum(np.zeros(A.shape[0]), A @ q - b) # optimizzed all control points

    # Get acceleration spline
    a = A_a @ q_orig
    a_star = A_a_sub @ q_star

    q_star = np.hstack((q_orig[0:3*p], q_star)) # concatenate the optimized control points with the rest
    a_star = np.hstack((a[0:3*(p-2)], a_star)) # concatenate the optimized control points with the rest

    print(q_star.shape)
    print(a_star.shape)

    return q_star, a_star, a

def get_vel_accel_star_wrt_q(p, n, knots, v_max, a_max, q):

    """
    This funciton is used to get the optimized velocity and acceleration spline at the same time for the minimum-deviation q trajectory
    (Not works well)
    
    """

    # Get velocity constraint matrix A_v and vector b_v
    A_v, b_v = get_vel_constr(p, n, knots, v_max)

    # Get acceleration constraint matrix A_a and vector b_a
    A_a, b_a = get_acc_constr_wrt_q(p, n, knots, a_max)

    # optimize only a subset of the control points
    q_sub = q[3*p:]
    A_v_sub = A_v[3*(p-1):, 3*p:]
    b_v_sub = b_v[3*p:]

    # add velocity inequality constraints from q_p to q_{p+1}
    alpha_v_2 = p / (knots[2//3+p+1] - knots[2//3+1]) # v2 is the first control point we have control over
    b_v_sub_prime = np.zeros(3)
    b_v_sub_prime[0] = v_max + alpha_v_2 * q[6]
    b_v_sub_prime[1] = v_max + alpha_v_2 * q[7]
    b_v_sub_prime[2] = v_max + alpha_v_2 * q[8]
    b_v_sub = np.hstack((b_v_sub_prime, b_v_sub))

    # optimize only a subset of the control points
    q_sub = q[3*p:]
    A_a_sub = A_a[3*(p-2):, 3*p:]
    b_a_sub = b_a[3*p:]

    # add acceleration inequality constraints from q_{p-1} and q_p to q_{p+1}

    # used to constraint q_{3}
    alpha_a_1_1 = p*(p-1) / (knots[1//3+p+1] - knots[1//3+2]) # a1 is the first control point we have control over
    alpha_a_2_1 = 1 / (knots[1//3+p+2] - knots[1//3+2])
    alpha_a_3_1 = (knots[1//3+p+1] - knots[1//3+1] + knots[1//3+p+2] - knots[1//3+2]) / ( (knots[1//3+p+2] - knots[1//3+2]) * (knots[1//3+p+1] - knots[1//3+1]) )
    alpha_a_4_1 = 1 / (knots[1//3+p+1] - knots[1//3+1])

    # used to constraint q_{4}
    alpha_a_1_2 = p*(p-1) / (knots[2//3+p+1] - knots[2//3+2]) 
    alpha_a_2_2 = 1 / (knots[2//3+p+2] - knots[2//3+2])
    alpha_a_3_2 = (knots[2//3+p+1] - knots[2//3+1] + knots[2//3+p+2] - knots[2//3+2]) / ( (knots[2//3+p+2] - knots[2//3+2]) * (knots[2//3+p+1] - knots[2//3+1]) )
    alpha_a_4_2 = 1 / (knots[2//3+p+1] - knots[2//3+1])

    b_a_sub_prime = np.zeros(6)
    # constraint q_{3}
    b_a_sub_prime[0] = a_max + alpha_a_1_1 * alpha_a_3_1 * q[6] - alpha_a_1_1 * alpha_a_4_1 * q[3]
    b_a_sub_prime[1] = a_max + alpha_a_1_1 * alpha_a_3_1 * q[7] - alpha_a_1_1 * alpha_a_4_1 * q[4]
    b_a_sub_prime[2] = a_max + alpha_a_1_1 * alpha_a_3_1 * q[8] - alpha_a_1_1 * alpha_a_4_1 * q[5]
    # constraint q_{4}
    b_a_sub_prime[3] = a_max - alpha_a_1_2 * alpha_a_4_2 * q[6]
    b_a_sub_prime[4] = a_max - alpha_a_1_2 * alpha_a_4_2 * q[7]
    b_a_sub_prime[5] = a_max - alpha_a_1_2 * alpha_a_4_2 * q[8]

    b_a_sub = np.hstack((b_a_sub_prime, b_a_sub))

    # Get q, A, and b
    q_orig = q
    q = q_sub
    # A = A_v_sub
    # b = b_v_sub
    # A = A_a_sub
    # b = b_a_sub
    A = np.vstack((A_v_sub, A_a_sub))
    b = np.hstack((b_v_sub, b_a_sub))

    # to get the minimum bound 
    # A = np.vstack((A, -A))
    # b = np.hstack((b, b))

    # Get the closed-form solution
    q_star = q + A.T @ np.linalg.inv(- A @ A.T) @ np.maximum(np.zeros(A.shape[0]), A @ q - b) # optimizzed all control points

    # Get velocity spline
    v = A_v @ q_orig
    v_star = A_v_sub @ q_star

    # Get acceleration spline
    a = A_a @ q_orig
    a_star = A_a_sub @ q_star

    q_star = np.hstack((q_orig[0:3*p], q_star)) # concatenate the optimized control points with the rest
    v_star = np.hstack((v[0:3*(p-1)], v_star)) # concatenate the optimized control points with the rest
    a_star = np.hstack((a[0:3*(p-2)], a_star)) # concatenate the optimized control points with the rest

    print(q_star.shape)
    print(v_star.shape)
    print(a_star.shape)

    return q_star, v_star, a_star

def get_vel_star(p, n, knots, v_max, q):

    """
    This funciton is used to get the optimized velocity spline
    """

    # Get velocity constraint matrix A_v and vector b_v
    A_v, b_v = get_vel_constr(p, n, knots, v_max)

    # optimize only a subset of the control points
    q_sub = q[3*p:]
    A_v_sub = A_v[3*(p-1):, 3*p:]
    b_v_sub = b_v[3*p:]

    # add velocity inequality constraints from q_p to q_{p+1}
    alpha_v_2 = p / (knots[2//3+p+1] - knots[2//3+1]) # v2 is the first control point we have control over
    b_v_sub_prime = np.zeros(3)
    b_v_sub_prime[0] = v_max + alpha_v_2 * q[6]
    b_v_sub_prime[1] = v_max + alpha_v_2 * q[7]
    b_v_sub_prime[2] = v_max + alpha_v_2 * q[8]
    b_v_sub = np.hstack((b_v_sub_prime, b_v_sub))

    # optimize only a subset of the control points
    q_sub = q[3*p:]

    # Get q, A, and b
    q_orig = q
    q = q_sub
    A = A_v_sub
    b = b_v_sub

    # Get the closed-form solution
    q_star = q + A.T @ np.linalg.inv(- A @ A.T) @ np.maximum(np.zeros(A.shape[0]), A @ q - b) # optimizzed all control points

    # Get velocity spline
    v = A_v @ q_orig
    v_star = A_v_sub @ q_star

    q_star = np.hstack((q_orig[0:3*p], q_star)) # concatenate the optimized control points with the rest
    v_star = np.hstack((v[0:3*(p-1)], v_star)) # concatenate the optimized control points with the rest

    return q_star, v_star, v

def get_acc_star_wrt_v(p, n, knots, a_max, v):

    # Get acceleration constraint matrix A_a and vector b_a
    A_a, b_a = get_acc_constr_wrt_v(p, n, knots, a_max)

    # optimize only a subset of the control points

    print(v.shape)
    v_sub = v[3*(p-1):]
    A_a_sub = A_a[3*(p-2):, 3*(p-1):]
    b_a_sub = b_a[3*(p-1):]

    # add acceleration inequality constraints from v_{p-1} to v_p
    alpha = (p-1) / (knots[2//3+p+1] - knots[2//3+2]) # v2 is the first control point we have control over
    b_a_sub_prime = np.zeros(3)
    b_a_sub_prime[0] = a_max + alpha * v[3]
    b_a_sub_prime[1] = a_max + alpha * v[4]
    b_a_sub_prime[2] = a_max + alpha * v[5]
    b_a_sub = np.hstack((b_a_sub_prime, b_a_sub))

    # Get q, A, and b
    v_orig = v
    v = v_sub
    A = A_a_sub
    b = b_a_sub

    # Get the closed-form solution
    v_star = v + A.T @ np.linalg.inv(- A @ A.T) @ np.maximum(np.zeros(A.shape[0]), A @ v - b) # optimizzed all control points

    # Get acceleration spline
    a = A_a @ v_orig
    a_star = A_a_sub @ v_star

    v_star = np.hstack((v_orig[0:3*(p-1)], v_star)) # concatenate the optimized control points with the rest
    a_star = np.hstack((a[0:3*(p-2)], a_star)) # concatenate the optimized control points with the rest

    return v_star, a_star, a

def main():

    # problem parameters
    v_max = 3 # m/s
    a_max = 5 # m/s^2
    j_max = 8 # m/s^3

    # Test example from main.m
    p = 3 # degree of b-spline
    n = 9 # n+1 control points
    m = n + p + 1 # number of knots
    tf = 3.53

    # Get q vector (position b-spline)
    # the first constrol point is the final position
    q = np.array([0, 0, 0, \
                    0, 0, 0, \
                    0, 0, 0, \
                    1.6468, -0.3788, 0.0001, \
                    2.8523, -1.0509, 0.0019, \
                    4.0578, -1.7163, 0.0029, \
                    5.7046, -2.0837, 0.0001, \
                    5.7046, -2.0837, 0.0001, \
                    5.7046, -2.0837, 0.0001, \
                    5.7046, -2.0837, 0.0011])

    # Get knots vector
    knots = np.zeros(m+1)
    knots[-p-1:] = tf
    for i in range(m-2*p):
        knots[p+i] = i * tf / (m-2*p)

    # Get q_star, v_star, and a_star
    q_star = solve_QP(p, n, knots, v_max, a_max, j_max, q)
    
    # put q_star into the original q vector size
    q_star = np.hstack((q[0:3*p], q_star))

    # get v and v_star
    A_v, b_v = get_vel_constr(p, n, knots, v_max)
    A_a, b_a = get_acc_constr_wrt_q(p, n, knots, a_max)
    A_j, b_j = get_jerk_constr_wrt_q(p, n, knots, j_max)

    v = A_v @ q
    v_star = A_v @ q_star
    a = A_a @ q
    a_star = A_a @ q_star
    j = A_j @ q
    j_star = A_j @ q_star

    # Plot the results
    # plot_results(q_orig, q_star, v, v_star, n)

    # subfigure 1: position
    fig, axes = plt.subplots(1, 4, sharex=True)
    axes[0].plot(np.linspace(0, n+1, n+1), q[0::3], 'b', label='x')
    axes[0].plot(np.linspace(0, n+1, n+1), q[1::3], 'r', label='y')
    axes[0].plot(np.linspace(0, n+1, n+1), q[2::3], 'g', label='z')
    axes[0].plot(np.linspace(0, n+1, n+1), q_star[0::3], 'b--', label='x*')
    axes[0].plot(np.linspace(0, n+1, n+1), q_star[1::3], 'r--', label='y*')
    axes[0].plot(np.linspace(0, n+1, n+1), q_star[2::3], 'g--', label='z*')
    axes[0].legend()

    # subfigure 2: velocity
    axes[1].plot(np.linspace(0, n, n), v[0::3], 'b', label='vx')
    axes[1].plot(np.linspace(0, n, n), v[1::3], 'r', label='vy')
    axes[1].plot(np.linspace(0, n, n), v[2::3], 'g', label='vz')
    axes[1].plot(np.linspace(0, n, n), v_star[0::3], 'b--', label='vx*')
    axes[1].plot(np.linspace(0, n, n), v_star[1::3], 'r--', label='vy*')
    axes[1].plot(np.linspace(0, n, n), v_star[2::3], 'g--', label='vz*')
    axes[1].axhline(y=v_max, color='k', linestyle='--')
    axes[1].axhline(y=-v_max, color='k', linestyle='--')
    axes[1].legend()

    # subfigure 3: acceleration
    axes[2].plot(np.linspace(0, n-1, n-1), a[0::3], 'b', label='ax')
    axes[2].plot(np.linspace(0, n-1, n-1), a[1::3], 'r', label='ay')
    axes[2].plot(np.linspace(0, n-1, n-1), a[2::3], 'g', label='az')
    axes[2].plot(np.linspace(0, n-1, n-1), a_star[0::3], 'b--', label='ax*')
    axes[2].plot(np.linspace(0, n-1, n-1), a_star[1::3], 'r--', label='ay*')
    axes[2].plot(np.linspace(0, n-1, n-1), a_star[2::3], 'g--', label='az*')
    axes[2].axhline(y=a_max, color='k', linestyle='--')
    axes[2].axhline(y=-a_max, color='k', linestyle='--')
    axes[2].legend()

    # subfigure 4: jerk
    axes[3].plot(np.linspace(0, n-2, n-2), j[0::3], 'b', label='jx')
    axes[3].plot(np.linspace(0, n-2, n-2), j[1::3], 'r', label='jy')
    axes[3].plot(np.linspace(0, n-2, n-2), j[2::3], 'g', label='jz')
    axes[3].plot(np.linspace(0, n-2, n-2), j_star[0::3], 'b--', label='jx*')
    axes[3].plot(np.linspace(0, n-2, n-2), j_star[1::3], 'r--', label='jy*')
    axes[3].plot(np.linspace(0, n-2, n-2), j_star[2::3], 'g--', label='jz*')
    axes[3].axhline(y=j_max, color='k', linestyle='--')
    axes[3].axhline(y=-j_max, color='k', linestyle='--')
    axes[3].legend()

    plt.show()
    exit()

    # If we want to impose equality constraints for the first p control points (initial points), we can use the following
    C_v = np.zeros((3*p, 3*(n+1)))
    d_v = np.zeros(3*p)
    for i in range(0, 3*p):
        C_v[i, i] = 1
        d_v[i] = q[i]
    
    print(C_v)

    # Get the closed-form solution
    A_v_hat = np.vstack((A_v, C_v))
    b_v_hat = np.hstack((b_v, d_v))

    print(b_v_hat)

    # get the rank of A_v_hat
    print(A_v_hat)
    print(A_v_hat.shape)
    print(np.linalg.inv(A_v_hat @ A_v_hat.T))
    exit()

    q_star_hat = q + A_v_hat.T @ np.linalg.inv(- A_v_hat @ A_v_hat.T) @ np.maximum(np.zeros(3*n), A_v_hat @ q - b_v_hat)

if __name__ == '__main__':
    main()