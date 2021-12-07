from math import exp, log
from Penman_Monteith.dics import *

# MLT: The following code was taken from the Photo3 code repository and altered for use in this project.

# ASSUMPTIONS:
# Stomatal conductance is 20 mm/s - value for coniferous forests from Monteith & Unsworth (2003)
# Air conductance is 324 mm/s, much larger than gs

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
	GS = 20. # mm/s
	GA = 324. # mm/s

	return ((LAMBDA_W*GAMMA_W*GA/1000.*RHO_A*drh(ta, qa) + delta_s(ta)*phi)*GS/1000.*1000000.)/ \
	(RHO_W*LAMBDA_W*(GAMMA_W*(GA/1000. + GS/1000.) + GS/1000.*delta_s(ta)))


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
