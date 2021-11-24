from math import exp, log
from Penman_Monteith.dics import *

# MLT: The following code was taken from the Photo3 database and altered for use in this project.

# ASSUMPTIONS:
# Wind and humidity are measured at a height of 2 m
# Wind speed at any time stamp is equal to the monthly average at the location
# Canopy is 20 m tall
# Stomatal resistance is 100 s/m
# Zero displacement height and roughness length governing momentum transfer can be estimated by d = 2/3h and z = 0.123h


def evfPen(phi, ta, qa, u):
	# phi = solar radiation intensity (W/m2), ta = atmospheric temperature (K), qa = specific humidity (kg/kg), u = wind speed (m/s)
	"""Penman-Monteith transpiration (um/sec)"""

	GAMMA_W = (P_ATM*CP_A)/(.622*LAMBDA_W)

	def delta_s(ta):
		return esat(ta)*(C_SAT*B_SAT)/(C_SAT + ta - 273)**2

	def drh(ta, qa):
		return VPD(ta, qa)*.622/P_ATM

	# gs and ga equations here
	h = 20  # canopy height, m
	ra = log(abs(2 - 2/3*h)/0.123/h)*log(abs(2 - 2/3*h)/0.1/0.123/h)/(0.41**2)/u  # can't take absolute value - need to fix this
	ga = 1/ra
	rs = 100/0.5/24/20  # convert 100 s/m to units of ???
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
	return 0.622*rh/100.*esat(ta)/P_ATM  # kg/kg
