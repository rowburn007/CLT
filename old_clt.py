import numpy as np

"""
########################################################################################################################
INPUTS
########################################################################################################################
"""
# Layup parameters
t = 0.005
layup = [0, 90]
n_plies = len(layup)
h = n_plies * t
unit_correction = 6894.76  # For in to si
# unit_correction = 1  # For si

# Point stress/strain analysis parameters
ply_to_investigate = 2
ply_to_investigate = ply_to_investigate - 1
side_to_investigate = 'top'

# Global Lamina Stresses
NM = np.array([[0], [0], [0], [0], [0], [0]], dtype=np.float64)

# Thermal stresses
delta_T = -75

# Material parameters
Vf = 0.7
E1 = 26.25 * 10 ** 6
E2 = 1.49 * 10 ** 6
v12 = 0.28
G12 = 1.04 * 10 ** 6

# Strength properties
s1T = 217.56 * 10 ** 3
s1C = 217.56 * 10 ** 3
s2T = 5.802 * 10 ** 3
s2C = 35.68 * 10 ** 3
t12 = 9.863 * 10 ** 3

# Thermal coefficients
a1 = 0.00000002
a2 = 0.0000225
a12 = 0

# Numpy print settings
np.set_printoptions(precision=3)

"""
########################################################################################################################
Q & S Matrices
########################################################################################################################
"""

# Compliance Matrix (S) (isotropic)
S11 = 1 / E1
S12 = -v12 / E1
S22 = 1 / E2
S66 = 1 / G12

S = np.array([[S11, S12, 0], [S12, S22, 0], [0, 0, S66]], dtype=np.float64)

# Stiffness Matrix (Q) (isotropic)
Q11 = S22 / (S11 * S22 - S12 ** 2)
Q12 = -S12 / (S11 * S22 - S12 ** 2)
Q22 = S11 / (S11 * S22 - S12 ** 2)
Q66 = 1 / S66

Q = np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]], dtype=np.float64)

# Stress vector
stress_vector = np.array([[s1T], [s2T], [t12]], dtype=np.float64)

# Strain vector
strain_vector = np.dot(S, stress_vector)

"""
########################################################################################################################
FUNCTIONS
########################################################################################################################
"""
# TODO DONE
# Find bar function
def get_bars(ply):
    # Transformation matrices
    theta = np.deg2rad(ply)
    c = np.cos(theta)
    s = np.sin(theta)

    T = np.array([[c ** 2, s ** 2, 2 * s * c], [s ** 2, c ** 2, -2 * s * c],
                  [-s * c, s * c, (c ** 2) - (s ** 2)]], dtype=np.float64)
    Tinv = np.linalg.inv(T)

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.float64)
    Rinv = np.linalg.inv(R)

    # Local to global transformation
    Sbar = np.linalg.multi_dot([Tinv, S, R, T, Rinv]) * unit_correction
    Qbar = np.linalg.multi_dot([Tinv, Q, R, T, Rinv]) * unit_correction

    return Sbar, Qbar

# TODO DONE
# Iterates through each ply starting at k=1 and finds ABD matrices
def get_ABD():
    # ABD Matrices

    # Extensional stiffness matrix relating the resultant in-plane forces to the in-plane strains.
    A = np.zeros((3, 3),
                 dtype=np.float64)

    # Coupling stiffness matrix coupling the force and moment terms to the midplane strains and midplane curvatures.
    B = np.zeros((3, 3),
                 dtype=np.float64)

    # Bending stiffness matrix relating the resultant bending moments to the plate curvatures.
    D = np.zeros((3, 3),
                 dtype=np.float64)

    h0 = -t * n_plies / 2

    for k, ply in enumerate(layup, start=1):
        hk = h0 + (t * (k))
        h1 = h0 + (t * ((k) - 1))

        bars = get_bars(ply)
        A += bars[1] * (hk - h1)
        B += 0.5 * bars[1] * ((hk ** 2) - (h1 ** 2))
        D += (1 / 3) * bars[1] * ((hk ** 3) - (h1 ** 3))
    return A, B, D

# TODO DONE
# Global strains at position
def get_ply_global_stress_strain(full_midplane_vector, distance, angle):
    midplane_strain_vector = np.array(full_midplane_vector[:3], dtype=np.float64)
    midplane_curvature_vector = np.array(full_midplane_vector[3:], dtype=np.float64)

    q_bar_at_ply = get_bars(angle)[1]

    global_point_strain_vector = midplane_strain_vector + distance * midplane_curvature_vector
    global_point_stress_vector = np.dot(q_bar_at_ply, global_point_strain_vector)

    return global_point_stress_vector, global_point_strain_vector

# TODO DONE
# Finds local stress and strain at point in ply
def get_ply_local_stress_strain(full_midplane_vector, distance, angle):
    # Transformation matrices
    angle = np.deg2rad(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    T = np.array([[c ** 2, s ** 2, 2 * s * c], [s ** 2, c ** 2, -2 * s * c],
                  [-s * c, s * c, (c ** 2) - (s ** 2)]], dtype=np.float64)

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.float64)
    Rinv = np.linalg.inv(R)

    # Global to Local transformation
    midplane_strain_vector = np.array(full_midplane_vector[:3], dtype=np.float64)
    local_midplane_strain_vector = np.linalg.multi_dot([R, T, Rinv, midplane_strain_vector])

    midplane_curvature_vector = np.array(full_midplane_vector[3:], dtype=np.float64)
    local_midplane_curvature_vector = np.linalg.multi_dot([R, T, Rinv, midplane_curvature_vector])

    local_point_strain_vector = local_midplane_strain_vector + (distance * local_midplane_curvature_vector)
    local_point_stress_vector = np.dot(Q, local_point_strain_vector) * unit_correction

    return local_point_stress_vector, local_point_strain_vector

# TODO DONE
# Stresses/strain per ply
def get_ply_stresses_strains(desired_ply, side):
    # Take in the ply number you want and top or bottom, finds distance, angle, and stress/strains
    # symmetric = lambda ply_num : ply_num % 2 == 0
    bottom_z = -(n_plies * t) / 2

    # Fills in dictionary containing z distances for top and bottom of each ply
    dist_dict = {}
    for ply, angle in enumerate(layup):
        ply_top = bottom_z + (t * ply)
        ply_middle = ply_top + (t / 2)
        ply_bottom = ply_top + t

        top_bottom_dict = {}
        top_bottom_dict['bottom'] = ply_bottom
        top_bottom_dict['middle'] = ply_middle
        top_bottom_dict['top'] = ply_top
        dist_dict[ply] = top_bottom_dict

    # Z distance from midplane and ply angle
    distance = dist_dict[desired_ply][side]
    angle = layup[desired_ply]

    local_point_stress_vector, local_point_strain_vector = get_ply_local_stress_strain(midplane_vector, distance, angle)
    global_point_stress_vector, global_point_strain_vector = get_ply_global_stress_strain(midplane_vector, distance,
                                                                                          angle)

    return local_point_stress_vector, local_point_strain_vector, global_point_stress_vector, global_point_strain_vector


# Get all local strain/stress vectors and the average of top middle bottom for each layer
def get_all_ply_stress_strain():
    positions = ['bottom', 'middle', 'top']

    sum_local_point_stress_vector = np.zeros((3, 1))
    sum_local_point_strain_vector = np.zeros((3, 1))
    sum_global_point_stress_vector = np.zeros((3, 1))
    sum_global_point_strain_vector = np.zeros((3, 1))

    avg_local_stress_dict = {}
    avg_local_strain_dict = {}
    avg_global_stress_dict = {}
    avg_global_strain_dict = {}

    # Summation of all stress/strain vectors for top, middle, and bottom
    for ply, angle in enumerate(layup):
        for side in positions:
            local_point_stress_vector = get_ply_stresses_strains(ply, side)[0]
            local_point_strain_vector = get_ply_stresses_strains(ply, side)[1]
            global_point_stress_vector = get_ply_stresses_strains(ply, side)[2]
            global_point_strain_vector = get_ply_stresses_strains(ply, side)[3]
            sum_local_point_stress_vector += local_point_stress_vector
            sum_local_point_strain_vector += local_point_strain_vector
            sum_global_point_stress_vector += global_point_stress_vector
            sum_global_point_strain_vector += global_point_strain_vector

        # Summation of all stress/strain vectors for each ply in dicts
        avg_local_stress_dict[ply] = sum_local_point_stress_vector / len(positions)
        avg_local_strain_dict[ply] = sum_local_point_strain_vector / len(positions)
        avg_global_stress_dict[ply] = sum_global_point_stress_vector / len(positions)
        avg_global_strain_dict[ply] = sum_global_point_strain_vector / len(positions)

        # Resetting summation vectors
        sum_local_point_stress_vector = 0
        sum_local_point_strain_vector = 0
        sum_global_point_stress_vector = 0
        sum_global_point_strain_vector = 0

        print(f'Ply {ply + 1} local stress: \n {avg_global_stress_dict.get(ply)} \n \n')

    return avg_global_stress_dict, avg_global_strain_dict, avg_local_stress_dict, avg_local_strain_dict


# Given a dictionary of average values per ply return vector of averages for that ply
def get_average_vector(average_dict, ply):
    avg_vector = average_dict[ply]
    return avg_vector


# Returns load percentage taken by each ply given dictionaries containing average stress/strain per ply
def get_load_pcts(global_stress):
    global_N_dict = {}

    sum_load = 0
    load_percent_dict = {}

    for ply, angle in enumerate(layup):
        avg_global_stress_vect = get_average_vector(avg_global_stress_dict, ply)
        global_N_dict[ply] = avg_global_stress_vect * t

        # Summation of load take per ply
        sum_load += abs(avg_global_stress_vect * t)

    for ply, angle in enumerate(layup):
        try:
            load_pct = (get_average_vector(global_N_dict, ply) / sum_load) * 100
            load_percent_dict[ply] = (get_average_vector(global_N_dict, ply) / sum_load) * 100
            print(f'Ply {ply + 1} load %: \n {load_pct} \n \n')
        except Exception:
            print('Zero division occurred and handled')
            pass


# Returns force vector resulting from thermal expansion
def get_thermal_force():
    # Starting z value of laminate
    h0 = -t * n_plies / 2

    Nt = 0

    for k, angle in enumerate(layup, start=1):
        # Z values for iteration through laminate
        hk = h0 + (t * (k))
        h1 = h0 + (t * ((k) - 1))

        qbar = get_bars(angle)[1]

        # Converts angle to rads
        angle = np.deg2rad(angle)

        ax = a1 * np.cos(angle) + a2*np.sin(angle)
        ay = a2 * np.cos(angle) + a1*np.sin(angle)
        axy = a12

        a_vect = np.array([[ax], [ay], [axy]])

        Nt += np.linalg.multi_dot([qbar, a_vect]) * (hk - h1)

    Nt = Nt * delta_T
    return Nt


# Find coefficient of thermal expansion
def get_thermal_coef(Nt):
    axy_vect = np.linalg.multi_dot([np.linalg.inv(A), Nt])
    return axy_vect

"""
########################################################################################################################
CALCULATIONS
########################################################################################################################
"""

# ABD Matrices
A = get_ABD()[0]
B = get_ABD()[1]
D = get_ABD()[2]

# Governing equation of laminate
G = np.block([[A, B], [B, D]])
G_inv = np.linalg.inv(G)

# Midplane stresses and strains (kirchoff)
midplane_vector = np.dot(G_inv, NM)


# Dictionaries containing average stress/strain for each ply
avg_global_stress_dict, avg_global_strain_dict, avg_local_stress_dict, avg_local_strain_dict = get_all_ply_stress_strain()

# Load percent per ply
get_load_pcts(avg_global_stress_dict)

# Thermal forces
NT = get_thermal_force()

# Engineering Constants
Ex = 1 / (h * G_inv[0][0])
Ey = 1 / (h * G_inv[1][1])
Gxy = 1 / (h * G_inv[2][2])
vxy = -G_inv[0][1] / G_inv[0][0]
vyx = -G_inv[0][1] / G_inv[1][1]

Exf = 12 / ((h ** 3) * G_inv[3][3])
Eyf = 12 / ((h ** 3) * G_inv[4][4])
Gxyf = 12 / ((h ** 3) * G_inv[5][5])
vxyf = -G_inv[3][4] / G_inv[3][3]
vyxf = -G_inv[3][4] / G_inv[4][4]


print(f'ABD matrix: \n {G} \n \n')
print(f'ABD inv matrix: \n {G_inv} \n \n')
print(f'Midplane vector: \n {midplane_vector} \n \n')


# print(f'Ex: \n {Ex} \n \n')
# print(f'Ey: \n {Ey} \n \n')
# print(f'Exf/Ex: \n {Exf / Ex} \n \n')
# print(f'Gxy: \n {Gxy} \n \n')
# print(f'vxy: \n {vxy} \n \n')
# print(f'vyx: \n {vyx} \n \n')
#
# print(f'Exf: \n {Exf} \n \n')
# print(f'Eyf: \n {Eyf} \n \n')
# print(f'Gxyf: \n {Gxyf} \n \n')
# print(f'vxyf: \n {vxyf} \n \n')
# print(f'vyxf: \n {vyxf} \n \n')

print(f'NT: {NT}')