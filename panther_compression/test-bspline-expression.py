#!/usr/bin/env python3

import sympy as sp







def main():

    # define the symbols for V_i
    P_i_3 = sp.Symbol('P_i_3')
    P_i_2 = sp.Symbol('P_i_2')
    P_i_1 = sp.Symbol('P_i_1')
    P_i = sp.Symbol('P_i')
    p = sp.Symbol('p')
    t_i_p_1 = sp.Symbol('t_i_p_1')
    t_i_1 = sp.Symbol('t_i_1')
    V_i = p * (P_i_1 - P_i)/(t_i_p_1 - t_i_1)

    # define the symbols for A_i
    t_i_2 = sp.Symbol('t_i_2')
    t_i_p_2 = sp.Symbol('t_i_p_2')
    V_i_1 = p * (P_i_2 - P_i_1)/(t_i_p_2 - t_i_2)
    A_i = (p-1) / (t_i_p_1 - t_i_2) * (V_i - V_i_1)

    print(A_i)





















if __name__ == '__main__':
    main()