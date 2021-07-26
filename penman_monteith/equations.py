from math import exp, pi, sqrt, log
from Penman_Monteith.dics import *

# ASSUMPTIONS:
# Wind and humidity are measured at a height of 2 m
# Wind speed is always 2 m *alter if data are available*
# Eucalyptus canopy is 20 m tall
# Stomatal resistance is 100 s/m
# zero displacement height and roughness length governing momentum transfer can be estimated by d = 2/3h and z = 0.123h


def evfPen(phi, ta, qa):
	# phi = solar radiation intensity (W/m2), ta = atmospheric temperature (K), qa = specific humidity (kg/kg)
	"""Penman-Monteith transpiration (um/sec)"""

	GAMMA_W = (P_ATM*CP_A)/(.622*LAMBDA_W)

	def delta_s(ta):
		return esat(ta)*(C_SAT*B_SAT)/(C_SAT + ta - 273)**2

	def drh(ta, qa):
		return VPD(ta, qa)*.622/P_ATM

	# gs and ga equations here
	ra = log((2 - 2/3*20)/0.123/20)*log((2 - 2/3*0.12)/0.1/0.123/20)/(0.41**2)/2
	ga = 1/ra
	rs = 100/0.5/24/20
	gs = 1/rs

	return ((LAMBDA_W*GAMMA_W*ga/1000.*RHO_A*drh(ta, qa) + delta_s(ta)*phi)*(R*ta/P_ATM)*gs*1000000.) / \
		(RHO_W*LAMBDA_W*(GAMMA_W*(ga/1000. + (R*ta/P_ATM)*gs) + (R*ta/P_ATM)*gs*delta_s(ta)))


def steps(duration, timeStep):
	"""Change Duration of Simulation to to number of timesteps according to timestep value"""
	return (duration*24*60)//timeStep


def VPD(ta, qa):
	"""Vapor pressure deficit (Pa)"""
	return esat(ta) - (qa*P_ATM)/.622


def esat(ta):
	"""Saturated vapor pressure (Pa)"""
	return A_SAT*exp((B_SAT*(ta - 273.))/(C_SAT + ta - 273.))


def qaRh(rh, ta):
	"""Specific humidity (kg/kg), input of rh in %, ta in K"""
	return 0.622*rh/100.*esat(ta)/P_ATM  # needs to be in kg/kg
