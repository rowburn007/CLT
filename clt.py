import numpy as np
from random import randint
import matplotlib.pyplot as plt

# Numpy print settings
np.set_printoptions(precision=3)

"""
########################################################################################################################
PLY
########################################################################################################################
"""


class Ply:
    def __init__(self, angle, ply_num, ply_material_parameters):
        self.ply_num = ply_num
        self.angle = angle

        # Material parameters
        self.Vf = ply_material_parameters[0]
        self.E1 = ply_material_parameters[1]
        self.E2 = ply_material_parameters[2]
        self.v12 = ply_material_parameters[3]
        self.G12 = ply_material_parameters[4]

        # Strength properties
        self.s1T = ply_material_parameters[5]
        self.s1C = ply_material_parameters[6]
        self.s2T = ply_material_parameters[7]
        self.s2C = ply_material_parameters[8]
        self.t12 = ply_material_parameters[9]

        # Thermal coefficients
        self.a1 = ply_material_parameters[10]
        self.a2 = ply_material_parameters[11]
        self.a12 = ply_material_parameters[12]

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
    def __init__(self, layup, laminate_material_parameters, ply_material_parameters, thickness, delta_t, load):
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
        self.plies = self.make_plies(layup, laminate_material_parameters)
        self.laminate_material_parameters = laminate_material_parameters
        self.ply_material_parameters = ply_material_parameters

        # Extensional stiffness matrix relating the resultant in-plane forces to the in-plane strains.
        A = np.zeros((3, 3), dtype=np.float64)
        # Coupling stiffness matrix coupling the force and moment terms to the midplane strains and midplane curvatures.
        B = np.zeros((3, 3), dtype=np.float64)
        # Bending stiffness matrix relating the resultant bending moments to the plate curvatures.
        D = np.zeros((3, 3), dtype=np.float64)

        h0 = -self.ply_thickness * self.n_plies / 2

        for ply_num, angle in enumerate(self.layup, start=1):
            ply = Ply(angle, ply_num, self.ply_material_parameters[ply_num - 1])
            q_bar = ply.q_bar

            hk = h0 + (self.ply_thickness * ply_num)
            h1 = h0 + (self.ply_thickness * (ply_num - 1))

            A += q_bar * (hk - h1)
            B += 0.5 * q_bar * ((hk ** 2) - (h1 ** 2))
            D += (1 / 3) * q_bar * ((hk ** 3) - (h1 ** 3))

        self.abd_matrix = np.rint(np.block([[A, B], [B, D]])) * self.unit_correction
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
    def make_plies(cls, layup, ply_material_parameters):
        plies = []
        for ply_num, angle in enumerate(layup):
            ply = Ply(angle, ply_num, ply_material_parameters)
            plies.append(ply)

        return plies

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
        ply = Ply(angle, ply_num, self.ply_material_parameters[ply_num])

        midplane_strain_vector = np.array(self.midplane_strain_curvature_vector[:3], dtype=np.float64)
        midplane_curvature_vector = np.array(self.midplane_strain_curvature_vector[3:], dtype=np.float64)

        q_bar_at_ply = ply.q_bar
        z_distance = self.get_z_distance(ply_num, side)

        global_point_strain_vector = midplane_strain_vector + z_distance * midplane_curvature_vector
        global_point_stress_vector = np.dot(q_bar_at_ply, global_point_strain_vector)

        # Combining global thermal and mechanical stress strain
        (local_point_thermal_stress_vector, global_point_thermal_stress_vector,
         local_point_thermal_strain_vector, global_point_thermal_strain_vector) = self.thermal_stress(ply_num, side)

        global_point_strain_vector = global_point_strain_vector + global_point_thermal_strain_vector
        global_point_stress_vector = global_point_stress_vector + global_point_thermal_stress_vector

        global_point_stress_strain_vector = np.vstack([global_point_stress_vector, global_point_strain_vector])

        if print_enabled:
            print(f'Global strain at Ply: {ply_num}, '
                  f'Angle: {angle}, Side: {side}, \n {global_point_stress_strain_vector} \n')

        return global_point_stress_strain_vector

    # Finds local stress and strain at point in ply
    def local_stress_strain(self, angle, ply_num, side, print_enabled=False):
        ply = Ply(angle, ply_num, self.ply_material_parameters[ply_num])

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

        # Combining thermal and mechanical stress
        (local_point_thermal_stress_vector, global_point_thermal_stress_vector,
         local_point_thermal_strain_vector, global_point_thermal_strain_vector) = self.thermal_stress(ply_num, side)

        local_point_stress_vector = local_point_stress_vector + local_point_thermal_stress_vector
        local_point_strain_vector = local_midplane_strain_vector + local_point_thermal_strain_vector

        # Combined local stress strain
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
        pct_Nx = 0
        pct_Ny = 0
        pct_Nxy = 0
        load_Nx = 0
        load_Ny = 0
        load_Nxy = 0

        for ply_num, angle in enumerate(self.layup):
            for side in sides:
                average_stress += self.global_stress_strain(angle, ply_num, side=side)

            try:
                # Load taken in global direction by each ply
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
        pct_Mx = 0
        pct_My = 0
        pct_Mxy = 0

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

        for ply_num, angle in enumerate(self.layup, start=0):
            ply = Ply(angle, ply_num, self.ply_material_parameters[ply_num])

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
        ply = Ply(angle, ply_num, self.ply_material_parameters[ply_num])

        global_midplane_thermal_deformation_vector = self.thermal_strain()

        z_distance = self.get_z_distance(ply_num, side)

        global_point_thermal_strain = (global_midplane_thermal_deformation_vector[:3] +
                                       z_distance * global_midplane_thermal_deformation_vector[3:])

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

        # COnverting to local thermal stress and strain
        local_point_thermal_stress = np.linalg.multi_dot([R, T, Rinv, global_point_thermal_stress])
        local_point_thermal_strain = np.linalg.multi_dot([R, T, Rinv, global_point_thermal_strain])

        if print_enabled:
            print(f'Local thermal stress ply: {ply_num}, '
                  f'angle: {angle}, side: {side} \n {local_point_thermal_stress} \n')
            print(f'Global thermal stress ply: {ply_num}, '
                  f'angle: {angle}, side: {side} \n {global_point_thermal_stress} \n')

        return (local_point_thermal_stress, global_point_thermal_stress,
                local_point_thermal_strain, global_point_thermal_strain)

    # Returns tsai_wu criterion and factor of safety for a given ply
    def tsai_wu(self, ply_num, angle, print_enabled=False):
        tsai_wu_criterion = 0
        fos = 0

        ply = Ply(angle, ply_num, self.ply_material_parameters[ply_num])

        f1 = (1 / ply.s1T) + (1 / ply.s1C)
        f11 = -1 / (ply.s1T * ply.s1C)
        f2 = (1 / ply.s2T) + (1 / ply.s2C)
        f22 = -1 / (ply.s2T * ply.s2C)
        f12 = -0.5 * (f11 * f22) ** 0.5
        S6 = ply.t12
        f66 = 1 / (S6 ** 2)
        # print(f1, f2, f11, f22, f66, f12)

        local_stress_strain = self.local_stress_strain(angle, ply_num, 'middle', print_enabled=False)
        # print(local_stress_strain)

        s1 = local_stress_strain[0]
        s2 = local_stress_strain[1]
        T12 = local_stress_strain[2]

        a = (f11 * s1 ** 2) + (f22 * s2 ** 2) + (f66 * T12 ** 2) - (f11 * s1 * s2)
        b = (f1 * s1) + (f2 * s2)
        c = -1

        try:
            fos = (1 / (2 * a)) * (np.sqrt(b ** 2 + (4 * a)) - b)
            tsai_wu_criterion = (f1 * s1) + (f2 * s2) + (f11 * s1 ** 2) + (f22 * s2 ** 2) + (f66 * T12 ** 2) + (
                    2 * f12 * s1 * s2)
        except Exception as e:
            pass

        if print_enabled:
            print(f'Factor of safety: {fos}, Ply: {ply_num}, Angle: {angle}')
            print(f'Tsai-Wu criterion: {tsai_wu_criterion}, Ply: {ply_num}, Angle: {angle} \n')

        return tsai_wu_criterion, fos

    # Checks for fiber failure in laminate
    def check_fiber_failure(self):
        fiber_failure = False

        # Failure Mode check
        for ply_num, angle in enumerate(self.layup):
            ply = self.plies[ply_num]
            s1 = self.local_stress_strain(angle, ply_num, 'middle')[0]
            s2 = self.local_stress_strain(angle, ply_num, 'middle')[1]
            t6 = ply.t12
            F6 = t6

            if s1 > 0:
                F1 = ply.s1T
            else:
                F1 = ply.s1C

            if s2 > 0:
                F2 = ply.s2T
            else:
                F2 = ply.s2C

            fiber_failure = ((s2 / F2) ** 2) + ((t6 / F6) ** 2) < ((s1 / F1) ** 2)
            # print(f'{((s2 / F2) ** 2) + ((t6 / F6) ** 2)} < {((s1 / F1) ** 2)}')

            if fiber_failure:
                break

        return fiber_failure

    # Determines plies of failure
    def check_ply_failure(self, print_enabled=False):
        failed_plies = []
        # print(self.load)
        for ply_num, angle in enumerate(self.layup):
            tsai_wu_criterion, factor_of_safety = self.tsai_wu(ply_num, angle, print_enabled)
            if tsai_wu_criterion >= 1:
                failed_plies.append(ply_num)

        if print_enabled:
            print(f'Failed Plies; \n {failed_plies} \n')
        return failed_plies


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
s1C = -1500 * 10 ** 6
s2T = 40 * 10 ** 6
s2C = -246 * 10 ** 6
t12 = 68 * 10 ** 6

# Thermal coefficients
a1 = 0.00000002
a2 = 0.0000225
a12 = 0

# Material Density
density = 2300  # kg/m3

# Material Cost
cost = 10  # $/kg

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
INPUTS
########################################################################################################################
"""

length = 1.87
diameter = 0.84
test_pressure = 9000000
Nx = 0.5 * (test_pressure / (diameter / 2))
Ny = test_pressure * (diameter / 2)


def build_test_laminate():
    test_load = np.array([[0], [0], [0], [0], [0], [0]], dtype=np.float64)
    test_delta_t = 0

    test_layup = [
        90, 45, 0
    ]

    test_thickness = 0.00125

    test_material_parameters = [Vf, E1, E2, v12, G12, s1T, s1C, s2T, s2C, t12, a1, a2, a12]

    # Fills in default list of material parameters per ply
    test_ply_parameters = []
    for ply_num, angle in enumerate(test_layup):
        test_ply_parameters.append(test_material_parameters.copy())

    # Initializes test laminate object
    test_laminate = Laminate(test_layup, test_material_parameters, test_ply_parameters,
                             test_thickness, test_delta_t, test_load)

    return test_laminate


"""
########################################################################################################################
FUNCTIONS
########################################################################################################################
"""


# Increases load on laminate
def increase_load(laminate, load_increase, *loads_to_change):
    for load in loads_to_change[0][0]:
        laminate.load[load] = load_increase + laminate.load[load]

    laminate.load[0] = laminate.load[1] * 0.5

    return laminate


# Checks for matrix failure
def check_matrix_failure(laminate):
    matrix_failure = False

    # Failure Mode check
    for ply_num, angle in enumerate(laminate.layup):
        ply = laminate.laminate[ply_num]
        s1 = laminate.local_stress_strain(angle, ply_num, 'middle')[0]
        s2 = laminate.local_stress_strain(angle, ply_num, 'middle')[1]
        t6 = ply.t12
        F6 = t6

        # Sign check for strength coefficient
        if s1 > 0:
            F1 = ply.s1T
        else:
            F1 = ply.s1C

        if s2 > 0:
            F2 = ply.s2T
        else:
            F2 = ply.s2C

        matrix_failure = ((s2 / F2) ** 2) + ((t6 / F6) ** 2) > ((s1 / F1) ** 2)

        if matrix_failure:
            break

    return matrix_failure


# Should make a new laminate instance with updated ply material parameters
def update_ply_material_params(laminate):
    failed_plies = laminate.check_ply_failure()
    new_ply_material_parameters = laminate.ply_material_parameters.copy()

    # Updates E2 and G12 for failes plies
    for ply in failed_plies:
        ply_params = new_ply_material_parameters[ply]
        ply_params[2] = ply_params[2] * 0.25
        ply_params[4] = ply_params[4] * 0.25
        new_ply_material_parameters[ply] = ply_params

    return new_ply_material_parameters


# Runs progressive failure analysis
def progressive_failure(laminate, load_increase, *loads_to_change, print_enabled):
    fiber_failure = False
    initial_ply_material_parameters = laminate.ply_material_parameters
    while not fiber_failure:
        failed_plies = []
        while len(failed_plies) == 0:
            failed_plies = laminate.check_ply_failure(print_enabled)
            laminate = increase_load(laminate, load_increase, loads_to_change)
            laminate.__init__(laminate.layup, laminate.laminate_material_parameters,
                              initial_ply_material_parameters, laminate.ply_thickness, laminate.delta_t, laminate.load)

        fiber_failure = laminate.check_fiber_failure()
        if fiber_failure:
            break
        print(f'=====================================================================')
        print(f'Pressure: \n {laminate.load[1][0] / (diameter / 2)} Pa \n')
        print(f'Matrix failure load: \n {laminate.load} \n')
        print(f'Failed plies: {failed_plies}')
        print(f'ABD: \n {laminate.abd_matrix} \n')
        print(f'=====================================================================')
        initial_ply_material_parameters = update_ply_material_params(laminate)

    print(f'Fiber failure load: \n {laminate.load} \n')
    print(f'Pressure: \n {laminate.load[1][0] / (diameter / 2)} Pa \n')
    print(f'ABD: \n {laminate.abd_matrix} \n')


# Optimization
def rosen_der(x):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200 * (xm - xm_m1 ** 2) - 400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm)
    der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
    der[-1] = 200 * (x[-1] - x[-2] ** 2)
    return der


def mass(laminate):
    M = density * laminate.ply_thickness * length * diameter * np.pi * len(laminate.layup)
    return M


def check_odd(n):
    if n % 2 == 0:
        return False
    else:
        return True


def check_performance(laminate):
    weakest_ply = 0
    strongest_ply = 1
    for ply_num, angle in enumerate(laminate.layup):
        tsai_wu, fos = laminate.tsai_wu(ply_num, angle)
        if tsai_wu > weakest_ply:
            weakest_ply = ply_num

        if tsai_wu < strongest_ply:
            strongest_ply = ply_num

    return weakest_ply, strongest_ply


def average_tsai_wu(laminate):
    tsai_wus = []

    for ply_num, angle in enumerate(laminate.layup):
        tsai_wu, fos = laminate.tsai_wu(ply_num, angle)
        tsai_wus.append(tsai_wu)

    avg_tsai_wu = np.average(tsai_wus)
    return avg_tsai_wu


def get_pairs(position):
    # Gets ply pairs and their symmetric sister pair
    if check_odd(position):
        pair = position - 1, position
    else:
        pair = position, position + 1

    return pair

# Finds the best laminate and returns it
def choose_best(laminate):
    test_layup = laminate.layup.copy()
    layup_pairs = [[0, 90], [45, -45], [60, -60]]
    weakest_position, strongest_position = check_performance(laminate)
    weakest_mirror_position = -weakest_position
    strongest_mirror_position = -strongest_position

    # Gets all index pairs for strongest and weakest
    weakest_pair = get_pairs(weakest_position)
    weakest_mirror_pair = get_pairs(weakest_mirror_position)

    strongest_pair = get_pairs(strongest_position)
    strongest_mirror_pair = get_pairs(strongest_mirror_position)

    # Iterates through pairs, updating laminate object and checking performance
    average_tsai_wus = []
    for new_pair in layup_pairs:
        avg_tsai_wu = average_tsai_wu(laminate)
        average_tsai_wus.append(avg_tsai_wu)
        test_layup[weakest_pair[0]] = new_pair[0]
        test_layup[weakest_pair[1]] = new_pair[1]
        laminate.layup = test_layup
        laminate.__init__(laminate.layup, laminate.laminate_material_parameters,
                          laminate.ply_material_parameters, laminate.ply_thickness, laminate.delta_t, laminate.load)

    # Finds and sets laminate to optimal layup, updates laminate
    best_performance = min(average_tsai_wus)
    best_performance = average_tsai_wus.index(best_performance)
    test_layup[weakest_pair[0]] = layup_pairs[best_performance][0]
    test_layup[weakest_pair[1]] = layup_pairs[best_performance][1]
    test_layup[weakest_mirror_pair[0]] = layup_pairs[best_performance][0]
    test_layup[weakest_mirror_pair[1]] = layup_pairs[best_performance][1]
    test_layup.pop(strongest_pair[0])
    test_layup.pop(strongest_pair[1])
    test_layup.pop(strongest_mirror_pair[0])
    test_layup.pop(strongest_mirror_pair[1])

    laminate.layup = test_layup
    laminate.__init__(laminate.layup, laminate.laminate_material_parameters,
                      laminate.ply_material_parameters, laminate.ply_thickness, laminate.delta_t, laminate.load)

    return laminate



# Pressure vessel calcualtion
def calculate_pressure_vessel(laminate):
    failure = laminate.check_ply_failure()
    failure_pct = len(failure) / len(laminate.layup)
    volume = np.pi * diameter * length * (laminate.laminate_thickness)
    mass = density * volume
    total_cost = mass * cost
    print(f'Failure: {failure}')
    print(f'Failure %, {failure_pct * 100} %')
    print(f'Volume m3: {volume}')
    print(f'Number of plies: {len(laminate.layup)}')
    print(f'Wall thickness: {laminate.laminate_thickness * 1000} mm')
    print(f'Total mass: {mass} kg')
    print(f'Total cost: $ {total_cost}')


def test_vessesl(laminate):
    failure = False
    while not failure:
        if len(laminate.check_ply_failure()) > 0:
            failure = True
            break
        else:
            pass

        best = choose_best(laminate)
        calculate_pressure_vessel(best)
        print(f'Current Layup: \n {best.layup}')

"""
########################################################################################################################
CALLS
########################################################################################################################
"""
qb11s = []
qb12s = []
qb22s = []
qb16s = []
qb26s = []
qb66s = []

a11s = []
a12s = []
a22s = []
a16s = []
a26s = []
a66s = []

b11s = []
b12s = []
b22s = []
b16s = []
b26s = []
b66s = []

d11s = []
d12s = []
d22s = []
d16s = []
d26s = []
d66s = []

a = []
b = []
d = []

q_bar = []

xs = []
for x in range(90):
    # theta = np.rad2deg((x/100) * np.pi/2)
    theta = 30 + x/100 * 75

    test_load = np.array([[0], [0], [0], [0], [0], [0]], dtype=np.float64)
    test_delta_t = 0

    test_layup = [
        theta
    ]

    test_thickness = 0.00125

    test_material_parameters = [Vf, E1, E2, v12, G12, s1T, s1C, s2T, s2C, t12, a1, a2, a12]

    # Fills in default list of material parameters per ply
    test_ply_parameters = []
    for ply_num, angle in enumerate(test_layup):
        test_ply_parameters.append(test_material_parameters.copy())

    # Initializes test laminate object
    test_laminate = Laminate(test_layup, test_material_parameters, test_ply_parameters,
                            test_thickness, test_delta_t, test_load)
    abd = test_laminate.abd_matrix
    
    test_ply = Ply(theta, 0, test_material_parameters)
    qb = test_ply.q_bar

    qb11 = qb[0][0]
    qb12 = qb[0][1]
    qb22 = qb[1][1]
    qb16 = qb[0][2]
    qb26 = qb[1][2]
    qb66 = qb[2][2]

    qb11s.append(qb11)
    qb12s.append(qb12)
    qb22s.append(qb22)
    qb16s.append(qb16)
    qb26s.append(qb26)
    qb66s.append(qb66)

    a11 = abd[0][0]
    a12 = abd[0][1]
    a22 = abd[1][1]
    a16 = abd[2][0]
    a26 = abd[1][2]
    a66 = abd[2][2]

    a11s.append(a11)
    a12s.append(a12)
    a22s.append(a22)
    a16s.append(a16)
    a26s.append(a26)
    a66s.append(a66)

    b11 = abd[3][0]
    b12 = abd[3][1]
    b22 = abd[4][1]
    b16 = abd[5][0]
    b26 = abd[4][2]
    b66 = abd[5][2]

    b11s.append(b11)
    b12s.append(b12)
    b22s.append(b22)
    b16s.append(b16)
    b26s.append(b26)
    b66s.append(b66)

    d11 = abd[3][3]
    d12 = abd[3][4]
    d22 = abd[4][4]
    d16 = abd[5][3]
    d26 = abd[4][5]
    d66 = abd[5][5]

    d11s.append(d11)
    d12s.append(d12)
    d22s.append(d22)
    d16s.append(d16)
    d26s.append(d26)
    d66s.append(d66)

    xs.append(x)


a.append(a11s)
a.append(a12s)
a.append(a22s)
a.append(a16s)
a.append(a26s)
a.append(a66s)

b.append(b11s)
b.append(b12s)
b.append(b22s)
b.append(b16s)
b.append(b26s)
b.append(b66s)

d.append(d11s)
d.append(d12s)
d.append(d22s)
d.append(d16s)
d.append(d26s)
d.append(d66s)

q_bar.append(qb11s)
q_bar.append(qb12s)
q_bar.append(qb22s)
q_bar.append(qb16s)
q_bar.append(qb26s)
q_bar.append(qb66s)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(8, 4))
fig_2, ((ax_11, ax_12, ax_16), (ax_21, ax_22, ax_26), (ax_61, ax_62, ax_66)) = plt.subplots(3, 3, figsize=(8, 4))

label_list = ['a11', 'a12', 'a22', 'a16', 'a26', 'a66']

for i, e in enumerate(a):
    ax1.plot(xs, e, '-', label=label_list[i])

for i, e in enumerate(b):
    ax2.plot(xs, e, '-', label=label_list[i])
    ax3.plot(xs, e, '-', label=label_list[i])

for i, e in enumerate(d):
    ax4.plot(xs, e, '-', label=label_list[i])

ax_11.plot(xs, q_bar[0], '-')
ax_12.plot(xs, q_bar[1], 'r-')
ax_21.plot(xs, q_bar[1], 'r-')
ax_22.plot(xs, q_bar[2], 'y-')
ax_16.plot(xs, q_bar[3], 'y-')
ax_61.plot(xs, q_bar[3], 'y-')
ax_26.plot(xs, q_bar[4], 'g-')
ax_62.plot(xs, q_bar[4], 'g-')
ax_66.plot(xs, q_bar[5], '-')



ax1.xaxis.set_ticks(np.arange(0, 91, 15))
ax2.xaxis.set_ticks(np.arange(0, 91, 15))
ax3.xaxis.set_ticks(np.arange(0, 91, 15))
ax4.xaxis.set_ticks(np.arange(0, 91, 15))
plt.legend()
plt.show()








# progressive_failure(laminate_to_study, 5000, [0, 1], print_enabled=False)
# test = mass(laminate_to_study)
# calculate_pressure_vessel(laminate_to_study)
# best = choose_best(laminate_to_study)
# calculate_pressure_vessel(best)
# test_vessesl(laminate_to_study)