from math import exp, pi, sqrt, log
from scipy.optimize import fsolve
from sympy import *
import numpy as np
from dics import *
from functions import *

def evfPen(phi, ta, qa):
	"""Penman-Monteith transpiration (um/sec)"""
	# phi = solar radiation intensity (W/m2), ta = atmospheric temperature (K), qa = specific humidity (kg/kg)
	GAMMA_W = (P_ATM*CP_A)/(.622*LAMBDA_W) # Pa*J/kg/K/(J/kg) = Pa/k
	def delta_s(ta):
		""" Pa / C"""
		return esat(ta)*(C_SAT*B_SAT)/(C_SAT + ta -273)**2
	def drh(ta, qa):
		"""Unitless (Pa/Pa)"""
		return VPD(ta, qa)*.622/P_ATM 
	GS = 20. # mm/s...value for coniferous forests from Monteith and Unsworth (2003)
	GA = 324. # mm/s...assume much larger than gs (for now). This is the value calculated for doug fir.

	return ((LAMBDA_W*GAMMA_W*GA/1000.*RHO_A*drh(ta, qa) + delta_s(ta)*phi)*GS/1000.*1000000.)/ \
	(RHO_W*LAMBDA_W*(GAMMA_W*(GA/1000. + GS/1000.) + GS/1000.*delta_s(ta)))
