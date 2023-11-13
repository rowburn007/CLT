import numpy as np

# Numpy print settings
np.set_printoptions(precision=3)

"""
########################################################################################################################
PLY
########################################################################################################################
"""


class Ply:
    def __init__(self, angle, ply_num, material_parameters):
        self.ply_num = ply_num
        self.angle = angle

        # Material parameters
        self.Vf = material_parameters[0]
        self.E1 = material_parameters[1]
        self.E2 = material_parameters[2]
        self.v12 = material_parameters[3]
        self.G12 = material_parameters[4]

        # Strength properties
        self.s1T = material_parameters[5]
        self.s1C = material_parameters[6]
        self.s2T = material_parameters[7]
        self.s2C = material_parameters[8]
        self.t12 = material_parameters[9]

        # Thermal coefficients
        self.a1 = material_parameters[10]
        self.a2 = material_parameters[11]
        self.a12 = material_parameters[12]

        # Transformation
        theta = np.deg2rad(self.angle)  # Degree to radian
        c = np.cos(theta)
        s = np.sin(theta)

        T = np.array([[c ** 2, s ** 2, 2 * s * c], [s ** 2, c ** 2, -2 * s * c],
                      [-s * c, s * c, (c ** 2) - (s ** 2)]], dtype=np.float64)

        Tinv = np.linalg.inv(T)

        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.float64)
        Rinv = np.linalg.inv(R)

        # Composition of material properties per ply
        # Compliance Matrix (S) (isotropic)
        S11 = 1 / self.E1
        S12 = -self.v12 / self.E1
        S22 = 1 / self.E2
        S66 = 1 / self.G12

        # S matrix
        self.s_matrix = np.array([[S11, S12, 0], [S12, S22, 0], [0, 0, S66]], dtype=np.float64)

        # Stiffness Matrix (Q) (isotropic)
        Q11 = S22 / (S11 * S22 - S12 ** 2)
        Q12 = -S12 / (S11 * S22 - S12 ** 2)
        Q22 = S11 / (S11 * S22 - S12 ** 2)
        Q66 = 1 / S66

        self.q_matrix = np.array([[Q11, Q12, 0], [Q12, Q22, 0], [0, 0, Q66]], dtype=np.float64)

        # Stress vector
        self.stress_vector = np.array([[self.s1T], [self.s2T], [self.t12]], dtype=np.float64)

        # Strain vector
        self.strain_vector = np.dot(self.s_matrix, self.stress_vector)

        # Local to global transformation
        self.s_bar = np.linalg.multi_dot([Tinv, self.s_matrix, R, T, Rinv])
        self.q_bar = np.linalg.multi_dot([Tinv, self.q_matrix, R, T, Rinv])

        self.ax = self.a1 * np.cos(theta) ** 2 + self.a2 * np.sin(theta) ** 2
        self.ay = self.a2 * np.cos(theta) ** 2 + self.a1 * np.sin(theta) ** 2
        self.axy = 2 * np.cos(theta) * np.sin(theta) * (self.a1 - self.a2)

        self.a_vect = np.array([[self.ax], [self.ay], [self.axy]])


"""
########################################################################################################################
LAMINATE
########################################################################################################################
"""


class Laminate:
    def __init__(self, layup, material_parameters, thickness, delta_t, load):
        self.layup = layup
        self.ply_thickness = thickness
        self.n_plies = len(self.layup)
        self.unit_correction = 1
        self.abd_matrix = None
        self.abd_inv_matrix = None
        self.load = load
        self.midplane_strain_vector = None
        self.z_zero = -(self.ply_thickness * self.n_plies) / 2
        self.laminate_thickness = self.n_plies * self.ply_thickness
        self.delta_t = delta_t
        self.laminate = self.plies(layup, material_parameters)
        self.material_parameters = material_parameters

        # Extensional stiffness matrix relating the resultant in-plane forces to the in-plane strains.
        A = np.zeros((3, 3), dtype=np.float64)
        # Coupling stiffness matrix coupling the force and moment terms to the midplane strains and midplane curvatures.
        B = np.zeros((3, 3), dtype=np.float64)
        # Bending stiffness matrix relating the resultant bending moments to the plate curvatures.
        D = np.zeros((3, 3), dtype=np.float64)

        h0 = -self.ply_thickness * self.n_plies / 2

        for ply_num, angle in enumerate(self.layup, start=1):
            ply = Ply(angle, ply_num, material_parameters)
            q_bar = ply.q_bar

            hk = h0 + (self.ply_thickness * ply_num)
            h1 = h0 + (self.ply_thickness * (ply_num - 1))

            A += q_bar * (hk - h1)
            B += 0.5 * q_bar * ((hk ** 2) - (h1 ** 2))
            D += (1 / 3) * q_bar * ((hk ** 3) - (h1 ** 3))

        self.abd_matrix = np.block([[A, B], [B, D]]) * self.unit_correction
        self.abd_inv_matrix = np.linalg.inv(self.abd_matrix)
        self.midplane_strain_curvature_vector = np.linalg.multi_dot([self.abd_inv_matrix, self.load])

        # Engineering Constants
        self.Ex = 1 / (self.laminate_thickness * self.abd_inv_matrix[0][0])
        self.Ey = 1 / (self.laminate_thickness * self.abd_inv_matrix[1][1])
        self.Gxy = 1 / (self.laminate_thickness * self.abd_inv_matrix[2][2])
        self.vxy = -self.abd_inv_matrix[0][1] / self.abd_inv_matrix[0][0]
        self.vyx = -self.abd_inv_matrix[0][1] / self.abd_inv_matrix[1][1]

        self.Exf = 12 / ((self.laminate_thickness ** 3) * self.abd_inv_matrix[3][3])
        self.Eyf = 12 / ((self.laminate_thickness ** 3) * self.abd_inv_matrix[4][4])
        self.Gxyf = 12 / ((self.laminate_thickness ** 3) * self.abd_inv_matrix[5][5])
        self.vxyf = -self.abd_inv_matrix[3][4] / self.abd_inv_matrix[3][3]
        self.vyxf = -self.abd_inv_matrix[3][4] / self.abd_inv_matrix[4][4]

    # Stores all ply objects as a list for the laminate, updates E2 if needed
    @classmethod
    def plies(cls, layup, material_parameters):
        plies = []
        for ply_num, angle in enumerate(layup):
            ply = Ply(angle, ply_num, material_parameters)
            plies.append(ply)

        return plies

    # Updates E2 of given ply in laminate and updates laminate ply list
    def update(self, laminate, ply_num, angle, material_parameters):
        material_parameters[2] = material_parameters[2] * 0.75
        ply = Ply(angle, ply_num, material_parameters)
        laminate[ply_num] = ply

    # Returns z distance from midplane for given ply
    def get_z_distance(self, ply_num, side):
        z_distance = (self.z_zero + (ply_num * self.ply_thickness)) + (self.ply_thickness / 2)

        # Corrects z distance based on desired ply side
        if side == 'top':
            z_distance = z_distance - (self.ply_thickness / 2)
        elif side == 'bottom':
            z_distance = z_distance + (self.ply_thickness / 2)
        else:
            pass
        return z_distance

    # Returns local stress and strain vector at desired ply
    def global_stress_strain(self, angle, ply_num, side, print_enabled=False):
        ply = Ply(angle, ply_num, self.material_parameters)

        midplane_strain_vector = np.array(self.midplane_strain_curvature_vector[:3], dtype=np.float64)
        midplane_curvature_vector = np.array(self.midplane_strain_curvature_vector[3:], dtype=np.float64)

        q_bar_at_ply = ply.q_bar
        z_distance = self.get_z_distance(ply_num, side)

        global_point_strain_vector = midplane_strain_vector + z_distance * midplane_curvature_vector
        global_point_stress_vector = np.dot(q_bar_at_ply, global_point_strain_vector)
        global_point_stress_strain_vector = np.vstack([global_point_stress_vector, global_point_strain_vector])

        if print_enabled:
            print(f'Global strain at Ply: {ply_num}, '
                  f'Angle: {angle}, Side: {side}, \n {global_point_stress_strain_vector} \n')

        return global_point_stress_strain_vector

    # Finds local stress and strain at point in ply
    def local_stress_strain(self, angle, ply_num, side, print_enabled=False):
        ply = Ply(angle, ply_num, self.material_parameters)

        z_distance = self.get_z_distance(ply_num, side)

        # Transformation matrices
        theta = np.deg2rad(angle)
        c = np.cos(theta)
        s = np.sin(theta)

        T = np.array([[c ** 2, s ** 2, 2 * s * c], [s ** 2, c ** 2, -2 * s * c],
                      [-s * c, s * c, (c ** 2) - (s ** 2)]], dtype=np.float64)

        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.float64)
        Rinv = np.linalg.inv(R)

        # Global to Local transformation
        midplane_strain_vector = np.array(self.midplane_strain_curvature_vector[:3], dtype=np.float64)

        local_midplane_strain_vector = np.linalg.multi_dot([R, T, Rinv, midplane_strain_vector])
        local_midplane_stress_vector = np.linalg.multi_dot([ply.q_matrix, local_midplane_strain_vector])

        midplane_curvature_vector = np.array(self.midplane_strain_curvature_vector[3:], dtype=np.float64)
        local_midplane_curvature_vector = np.linalg.multi_dot([R, T, Rinv, midplane_curvature_vector])

        local_point_strain_vector = local_midplane_strain_vector + (z_distance * local_midplane_curvature_vector)
        local_point_stress_vector = np.dot(ply.q_matrix, local_point_strain_vector)
        local_point_stress_strain_vector = np.vstack([local_point_stress_vector, local_point_strain_vector])

        if print_enabled:
            print(f'Local midplane stress vector: \n {local_midplane_stress_vector} \n')
            print(f'Local stress/strain vector Ply: {ply_num}, '
                  f'Angle: {angle}, Side: {side}, \n {local_point_stress_strain_vector} \n')

        return local_point_stress_strain_vector

    # Finds load by ply and prints if print enabled is set to True
    def load_by_ply(self, print_enabled=False):
        sides = ['top', 'middle', 'bottom']
        average_stress = 0

        for ply_num, angle in enumerate(self.layup):
            for side in sides:
                average_stress += self.global_stress_strain(angle, ply_num, side=side)

            try:
                # Load taken in glabal direction by each ply
                load_Nx = ((average_stress[:3][0]) / self.n_plies) * self.ply_thickness
                load_Ny = ((average_stress[:3][1]) / self.n_plies) * self.ply_thickness
                load_Nxy = ((average_stress[:3][2]) / self.n_plies) * self.ply_thickness

                # Percent of load in direction taken by each ply
                pct_Nx = (load_Nx / self.load[0]) * 100
                pct_Ny = (load_Ny / self.load[1]) * 100
                pct_Nxy = (load_Nxy / self.load[2]) * 100
            except Exception as e:
                pass

            if print_enabled:
                # Print statements to enable / disable
                print(f'Ply: {ply_num} PCT Nx {pct_Nx}')
                print(f'Ply: {ply_num} PCT Ny {pct_Ny}')
                print(f'Ply: {ply_num} PCT Nxy {pct_Nxy}')

                print(f'Ply: {ply_num} Nx {load_Nx}')
                print(f'Ply: {ply_num} Ny {load_Ny}')
                print(f'Ply: {ply_num} Nxy {load_Nxy}')
                print(f'Ply {ply_num} Average Stress: \n {average_stress[:3]} \n')
                print(f'Ply {ply_num} Average Strain: \n {average_stress[3:]} \n')

            average_stress = 0

    # Returns the pct of moment carried by each layer
    def moment_by_ply(self, print_enabled=False):
        h0 = self.z_zero + self.ply_thickness
        Mx = 0
        My = 0
        Mxy = 0

        for ply_num, angle in enumerate(self.layup):
            hk = h0 + (self.ply_thickness * ply_num)
            h1 = h0 + (self.ply_thickness * (ply_num - 1))
            Mx = Mx + (self.global_stress_strain(angle, ply_num, 'middle')[0] * 0.5 * (hk ** 2 - h1 ** 2))
            My = My + (self.global_stress_strain(angle, ply_num, 'middle')[1] * 0.5 * (hk ** 2 - h1 ** 2))
            Mxy = Mxy + (self.global_stress_strain(angle, ply_num, 'middle')[2] * 0.5 * (hk ** 2 - h1 ** 2))

            try:
                pct_Mx = (Mx / self.load[3]) * 100
                pct_My = (My / self.load[4]) * 100
                pct_Mxy = (Mxy / self.load[5]) * 100

            except RuntimeWarning as e:
                pass

            if print_enabled:
                print(f'Ply: {ply_num}, Mx: {Mx}, My: {My}, Mxy: {Mxy}')
                print(f'Ply: {ply_num}, %Mx: {pct_Mx}, %My: {pct_My}, %Mxy: {pct_Mxy} \n')

            # Returns global midplane thermal strain vector, with print options

    def thermal_strain(self, print_enabled=False):
        # Starting z value of laminate
        h0 = -(self.ply_thickness * self.n_plies) / 2

        global_thermal_force_vector = 0
        global_thermal_moment_vector = 0

        for ply_num, angle in enumerate(self.layup, start=1):
            ply = Ply(angle, ply_num, self.material_parameters)

            # Z values for iteration through laminate
            hk = h0 + (self.ply_thickness * ply_num)
            h1 = h0 + (self.ply_thickness * (ply_num - 1))

            q_bar = ply.q_bar
            a_vect = ply.a_vect

            global_thermal_force_vector += np.linalg.multi_dot([q_bar, a_vect]) * (hk - h1)
            global_thermal_moment_vector += 0.5 * np.linalg.multi_dot([q_bar, a_vect]) * (hk ** 2 - h1 ** 2)

        global_thermal_force_vector = global_thermal_force_vector * self.delta_t
        global_thermal_moment_vector = global_thermal_moment_vector * self.delta_t

        thermal_force_moment_vector = np.vstack([global_thermal_force_vector, global_thermal_moment_vector])

        global_midplane_thermal_deformation_vector = (
            np.linalg.multi_dot([self.abd_inv_matrix, thermal_force_moment_vector]))

        if print_enabled:
            print(f'NxT: \n {global_thermal_force_vector} \n')
            print(f'MxT: \n {global_thermal_moment_vector} \n')
            print(f'Global Midplane Thermal Deformation Vector: \n {global_midplane_thermal_deformation_vector} \n')

        return global_midplane_thermal_deformation_vector

    # Finds the global and local thermal stress at specified side of laminate
    def thermal_stress(self, ply_num, side, print_enabled=False):
        angle = self.layup[ply_num]
        ply = Ply(angle, ply_num, self.material_parameters)

        global_midplane_thermal_deformation_vector = self.thermal_strain()

        z_distance = self.get_z_distance(ply_num, side)

        global_point_thermal_strain = (global_midplane_thermal_deformation_vector[:3] +
                                       z_distance * global_midplane_thermal_deformation_vector[3:])

        angle = self.layup[ply_num]

        plain_thermal_strain = ply.a_vect * self.delta_t
        plain_moment_thermal_strain = global_point_thermal_strain - plain_thermal_strain

        global_point_thermal_stress = np.linalg.multi_dot([ply.q_bar, plain_moment_thermal_strain])

        # Transformation matrices
        angle = np.deg2rad(angle)
        c = np.cos(angle)
        s = np.sin(angle)

        T = np.array([[c ** 2, s ** 2, 2 * s * c], [s ** 2, c ** 2, -2 * s * c],
                      [-s * c, s * c, (c ** 2) - (s ** 2)]], dtype=np.float64)

        R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=np.float64)
        Rinv = np.linalg.inv(R)
        local_point_thermal_stress = np.linalg.multi_dot([R, T, Rinv, global_point_thermal_stress])

        if print_enabled:
            print(f'Local thermal stress ply: {ply_num}, '
                  f'angle: {angle}, side: {side} \n {local_point_thermal_stress} \n')
            print(f'Global thermal stress ply: {ply_num}, '
                  f'angle: {angle}, side: {side} \n {global_point_thermal_stress} \n')

    # Returns tsai_wu criterion and factor of safety for a given ply
    def tsai_wu(self, angle, ply_num, print_enabled=False):
        ply = Ply(angle, ply_num, self.material_parameters)

        f1 = (1 / ply.s1T) + (1 / ply.s1C)
        f11 = -1 / (ply.s1T * ply.s1C)
        f2 = (1 / ply.s2T) + (1 / ply.s2C)
        f22 = -1 / (ply.s2T * ply.s2C)
        f12 = (-0.5) * (f11 * f22) ** 0.5
        S6 = ply.t12
        f66 = 1 / (S6 ** 2)

        local_stress_strain = self.local_stress_strain(angle, ply_num, 'middle')

        s1 = local_stress_strain[0]
        s2 = local_stress_strain[1]
        T12 = local_stress_strain[2]

        a = (f11 * s1 ** 2) + (f22 * s2 ** 2) + (f66 * T12 ** 2) - (f11 * s1 * s2)
        b = (f1 * s1) + (f2 * s2)
        c = -1

        try:
            fos = (1 / (2 * a)) * (np.sqrt(b ** 2 + (4 * a)) - b)
            tsai_wu = (f1 * s1) + (f2 * s2) + (f11 * s1 ** 2) + (f22 * s2 ** 2) + (f66 * T12 ** 2) + (2 * f12 * s1 * s2)
        except Exception as e:
            pass

        if print_enabled:
            print(f'Factor of safety: {fos}, Ply: {ply_num}, Angle: {angle}')
            print(f'Tsai-Wu criterion: {tsai_wu}, Ply: {ply_num}, Angle: {angle} \n')

        return tsai_wu, fos

    # Returns tsai-wu criterion by ply
    def ply_tsai_wu(self, print_enabled=False):
        for ply_num, angle in enumerate(self.layup):
            tsai_wu, fos = self.tsai_wu(angle, ply_num)

            if print_enabled:
                print(f'Factor of safety: {fos}, Ply: {ply_num}, Angle: {angle}')
                print(f'Tsai-Wu criterion: {tsai_wu}, Ply: {ply_num}, Angle: {angle} \n')


"""
########################################################################################################################
MATERIAL
########################################################################################################################
"""

# # CF inches
# # Material parameters
# Vf = 0.7
# E1 = 26.25 * 10 ** 6
# E2 = 1.49 * 10 ** 6
# v12 = 0.28
#
# G12 = 1.04 * 10 ** 6
#
# # Strength properties
# s1T = 217.56 * 10 ** 3
# s1C = 217.56 * 10 ** 3
# s2T = 5.802 * 10 ** 3
# s2C = 35.68 * 10 ** 3
# t12 = 9.863 * 10 ** 3
#
# # Thermal coefficients
# a1 = 0.00000002
# a2 = 0.0000225
# a12 = 0

# CFP meters
# Material parameters
Vf = 0.7
E1 = 181 * 10 ** 9
E2 = 10.30 * 10 ** 9
v12 = 0.28
G12 = 7.17 * 10 ** 9

# Strength properties
s1T = 1500 * 10 ** 6
s1C = 1500 * 10 ** 6
s2T = 40 * 10 ** 6
s2C = 246 * 10 ** 6
t12 = 68 * 10 ** 6

# Thermal coefficients
a1 = 0.00000002
a2 = 0.0000225
a12 = 0

# # Tsai Wu Pa
# # Material parameters
# Vf = 0.7
# E1 = 1.55 * 10 ** 11
# E2 = 1.21 * 10 ** 10
# v12 = 0.248
# G12 = 4.40 * 10 ** 9
#
# # Strength properties
# s1T = 1.50 * 10 ** 9
# s1C = -1.25 * 10 ** 9
# s2T = 5 * 10 ** 7
# s2C = -2 * 10 ** 8
# t12 = 1 * 10 ** 8
#
# # Thermal coefficients
# a1 = 0.00000002
# a2 = 0.0000225
# a12 = 0

# # CFP Inches Test
# # Material parameters
# Vf = 0.7
# E1 = 20 * 10 ** 6
# E2 = 1.2 * 10 ** 6
# v12 = 0.25
# G12 = 0.8 * 10 ** 6
#
# # Strength properties
# s1T = 300 * 10 ** 3
# s1C = -150 * 10 ** 3
# s2T = 7 * 10 ** 3
# s2C = -25 * 10 ** 3
# t12 = 14 * 10 ** 3
#
# # Thermal coefficients
# a1 = 0.00000002
# a2 = 0.0000225
# a12 = 0

"""
########################################################################################################################
FUNCTIONS
########################################################################################################################
"""

test_material_parameters = [Vf, E1, E2, v12, G12, s1T, s1C, s2T, s2C, t12, a1, a2, a12]

test_load = np.array([[0], [200000], [0], [0], [0], [0]], dtype=np.float64)
test_delta_t = 0

test_layup = [45, -45, 0, 0, -45, 45]
test_thickness = 0.00015

# TODO: FINALLY
# test_laminate_1 = Laminate(test_layup, test_material_parameters, test_thickness, test_delta_t, test_load)
# print(test_laminate_1.laminate[1].E2)
# test_laminate_1.update(test_laminate_1.laminate, 1, 45, test_material_parameters)
# test_laminate_1.update(test_laminate_1.laminate, 2, 45, test_material_parameters)
# print(test_laminate_1.laminate[1].E2)
# test_laminate_1.update(test_laminate_1.laminate, 1, 45, test_material_parameters)
# print(test_laminate_1.laminate[2].E2)

test_laminate = Laminate(test_layup, test_material_parameters, test_thickness, test_delta_t, test_load)

# test = test_laminate.local_stress_strain(30, 1, 'top', print_enabled=True)

# test_laminate.load_by_ply(print_enabled=True)
# test_laminate.thermal_strain(print_enabled=True)
# test_laminate.thermal_stress(1, 'bottom')
test_laminate.ply_tsai_wu(print_enabled=True)


# Laminate(test_layup, test_thickness, test_delta_t, test_load).local_stress_strain(30, 0, 'middle', print_enabled=True)
# Laminate(test_layup, test_thickness, test_delta_t, test_load).get_strength(test_load)
# Laminate(test_layup, test_thickness, test_delta_t, test_load).moment_by_ply()

# Determines the ply of first failure and the failure it fails at
def get_strength(laminate, loads_to_change, print_enabled=False):
    weakest_strength = 0
    weakest_ply = 0
    weakest_ply_angle = 0
    applied_load = laminate.load
    incrementing_value = applied_load.max() / 100
    # incrementing_value = 10000

    while weakest_strength <= 1:
        # Finds weakest ply and its corresponding tsai_wu criterion
        for ply_num, angle in enumerate(laminate.layup):
            tsai_wu, _ = laminate.tsai_wu(angle, ply_num)
            # tsai_wu = abs(tsai_wu)

            if tsai_wu > weakest_strength:
                weakest_ply = ply_num
                weakest_strength = tsai_wu
                weakest_ply_angle = angle

            if print_enabled:
                print(f'Ply: {ply_num} Tsai-Wu: {tsai_wu}')

        applied_load[loads_to_change[0]] += incrementing_value

    return weakest_ply, weakest_ply_angle, applied_load


weakest, angle, load = get_strength(test_laminate, [0], print_enabled=True)
print(f'Weakest Ply: {weakest}, Angle: {angle}, Load: \n {load}')


def progressive_failure(layup, material_parameters, thickness, delta_t, load, load_to_change):
    comparison = True
    while comparison:
        weakest_ply_num, weakest_ply_angle, load = get_strength(layup, material_parameters, thickness, delta_t, load,
                                                                load_to_change)
        laminate = Laminate(layup, material_parameters, thickness, delta_t, load)
        ply = Ply(weakest_ply_angle, weakest_ply_num, material_parameters)

        # Failure Mode check
        s1 = laminate.local_stress_strain(weakest_ply_angle, weakest_ply_num, 'middle')[0]
        s2 = laminate.local_stress_strain(weakest_ply_angle, weakest_ply_num, 'middle')[1]
        t6 = ply.t12

        f1 = (1 / ply.s1T) + (1 / ply.s1C)
        f2 = (1 / ply.s2T) + (1 / ply.s2C)
        f6 = t6

        comparison = (s2 / f2) ** 2 + (t6 / f6) ** 2 > (s1 / f1) ** 2

    print(f'Fiber Failure at ply: {weakest_ply_num}, load: \n {load} \n')
    print(ply.E2)

# progressive_failure(test_layup, test_material_parameters, test_thickness, test_delta_t, test_load, [0, 1])
