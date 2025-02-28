import taichi as ti
import taichi.math as tm
import numpy as np
from util import *

from pywavefront import Wavefront
from scene import Scene, Init

ti.init(arch=ti.vulkan)

## Models
# 'c': Co-rotated linear model
# 'v': St. Venant Kirchhoff model ( StVK )
# 'n': Neo-Hookean model
model = 'c'
prev_model = model
ModelSelector = ti.field(ti.i32, ())
ModelSelector[None] = 0

## physical quantities
# You are encouraged to add sliders for these params!
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
YoungsModulus[None] = 1e3
PoissonsRatio[None] = 0
gravity = ti.Vector([0, -9.8])
kspring = 100 # This is the stiffness of the user-input spring force!

##############################################################
# TODO: Put additional parameters here
# e.g. Lame parameters 
Lame_mu = ti.field(ti.f32, ())
Lame_lambda = ti.field(ti.f32, ())
##############################################################

@ti.kernel
def update_lame_parameters():
    Lame_mu[None] = YoungsModulus[None] / (2.0 * (1.0 + PoissonsRatio[None]))
    Lame_lambda[None] = YoungsModulus[None] * PoissonsRatio[None] / ((1.0 + PoissonsRatio[None]) * (1.0 - PoissonsRatio[None]))

## Load geometry of the simulation scenes
obj = Wavefront('models/woody-halfres.obj', collect_faces=True)
va = np.array(obj.vertices, dtype=np.float32)[:,:2] * 0.8

# Move the obj to center of the screen
x_avg = 0.5*(np.amin(va[:,0])+np.amax(va[:,0]))
y_avg = 0.5*(np.amin(va[:,1])+np.amax(va[:,1]))
va += np.array([0.5-x_avg, 0.5-y_avg])

mesh = obj.mesh_list[0]
faces = np.array(mesh.faces, dtype=np.int32)
mesh_triangles = ti.field(int, shape=np.prod(faces.shape))
mesh_triangles.from_numpy(faces.ravel())

# Initialize the bounding box for collision
house = Scene(Init.GB_BOX)

# Number of triangles
N_triangles = faces.shape[0]

triangles = ti.Vector.field(3, ti.i32, N_triangles)
for i in range(N_triangles):
    triangles[i] = ti.Vector(faces[i])

# We also need to draw the edges
edges_set = set()
for i in range(N_triangles):
    edges_set.add((faces[i][0],faces[i][1]) if faces[i][0] < faces[i][1] else (faces[i][1], faces[i][0]))
    edges_set.add((faces[i][1],faces[i][2]) if faces[i][1] < faces[i][2] else (faces[i][2], faces[i][1]))
    edges_set.add((faces[i][2],faces[i][0]) if faces[i][2] < faces[i][0] else (faces[i][0], faces[i][2]))

# Number of edges
N_edges = len(edges_set)
np_edges = np.array([list(e) for e in edges_set])
edges = ti.Vector.field(2, shape=N_edges, dtype=int)
edges.from_numpy(np_edges)

#############################################################
## Deformable object Simulation

# time-step size (for simulation, 16.7ms)
h = 16.7e-3
# substepping
substepping = 100
# time-step size (for time model)
dh = h/substepping

# Number of vertices
N = va.shape[0]
# Mass
m = 1/N*25 

# Simulation components: x and v
x = ti.Vector.field(2, shape=N, dtype=ti.f32)
x.from_numpy(va) # x is initialized directly to the mesh vertices (in 2D)
v = ti.Vector.field(2, ti.f32, N)

force = ti.Vector.field(2, ti.f32, N)
# spring force generated by user mouse drag
force_idx = ti.field(ti.i32, ()) # vertex being dragged on
spring_force = ti.Vector.field(2, ti.f32, 1) # spring force applied to the vertex
# forces created by mouse drags in Simulation Mode
force_start_pos = np.array([0,0])
force_end_pos = np.array([0,0])
draw_force = False

# Indices pinned by users
pins = ti.field(ti.i32, N)
num_pins = ti.field(ti.i32, ())

# TODO: Put additional fields here for storing D (etc.)
D0 = ti.Matrix.field(2, 2, dtype=ti.f32, shape=N_triangles)
area0 = ti.field(ti.f32, shape=N_triangles)

@ti.func
def compute_D(p0, p1, p2):
    return ti.Matrix.cols([p1 - p0, p2 - p0])

@ti.kernel
def initialize_D0():
    for t in range(N_triangles):
        i0 = triangles[t][0]
        i1 = triangles[t][1]
        i2 = triangles[t][2]
        # Compute D₀ using the initial positions stored in x (rest configuration)
        D0[t] = compute_D(x[i0], x[i1], x[i2])

@ti.kernel
def initialize_area0():
    for t in range(N_triangles):
        area0[t] = 0.5 * ti.abs(D0[t].determinant())

@ti.func
def compute_F(t: ti.i32) -> ti.Matrix:
    i0 = triangles[t][0]
    i1 = triangles[t][1]
    i2 = triangles[t][2]
    # Compute current D
    D = compute_D(x[i0], x[i1], x[i2])
    # Compute and return the deformation gradient F
    return D @ D0[t].inverse()
    
@ti.func
def polar_decomposition(F):
    u, s, v = ti.svd(F)
    R = u @ v.transpose()
    S = R.transpose() @ F
    return R, S

@ti.func
def compute_P(F):
    I2 = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
    # Initialize the first Piola-Kirchhoff stress tensor
    P_ret = ti.Matrix.zero(ti.f32, 2, 2)
    if ModelSelector[None] == 0:
        # Corotated linear model
        R, S = polar_decomposition(F)
        E_c = S - I2
        P_ret = R @ (2.0 * Lame_mu[None] * E_c + Lame_lambda[None] * E_c.trace() * I2)
    elif ModelSelector[None] == 1:
        # St. Venant-Kirchhoff model
        E_green = 0.5 * (F.transpose() @ F - I2)
        P_ret = F @ (2.0 * Lame_mu[None] * E_green + Lame_lambda[None] * E_green.trace() * I2)
    else:
        # Neo-Hookean model
        detF = F.determinant()
        invT = F.inverse().transpose()
        P_ret = Lame_mu[None] * (F - invT) + Lame_lambda[None] * ti.log(detF) * invT
    return P_ret

vertex_strain = ti.field(ti.f32, shape=N)

# TODO: Implement the initialization and timestepping kernel for the deformable object
@ti.kernel
def timestep(currmode: int):
    # Zero out forces and per-vertex strain
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])
    for i in range(N):
        vertex_strain[i] = 0.0  # [ADDED]
    
    for t in range(N_triangles):
        i0 = triangles[t][0]
        i1 = triangles[t][1]
        i2 = triangles[t][2]
        # Compute the current deformation gradient
        F = compute_F(t)
        # Compute the first Piola-Kirchhoff stress tensor
        P = compute_P(F)
        # Compute the inverse-transpose
        invT = D0[t].inverse().transpose()
        # Get the rest area A_e
        A_e = area0[t]
        # Compute force contribution matrix
        H = -A_e * (P @ invT)
        col0 = ti.Vector([H[0, 0], H[1, 0]])
        col1 = ti.Vector([H[0, 1], H[1, 1]])
        force[i1] += col0
        force[i2] += col1
        force[i0] += -(col0 + col1)

        # Frobenius norm of (F - I2)
        I2 = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])
        strain_val = (F - I2).norm_sqr()

        vertex_strain[i0] += strain_val
        vertex_strain[i1] += strain_val
        vertex_strain[i2] += strain_val
    
    for i in range(N):
        f_total = force[i] + (m * gravity / 50 if currmode == 2 else ti.Vector([0.0, 0.0]))
        v[i] += dh * (f_total / m)
        x[i] += dh * v[i]
        # Boundary projection
        if x[i].x < 0.0:
            x[i].x = 0.0
            v[i].x = 0.0
        if x[i].x > 1.0:
            x[i].x = 1.0
            v[i].x = 0.0
        if x[i].y < 0.0:
            x[i].y = 0.0
            v[i].y = 0.0
        if x[i].y > 1.0:
            x[i].y = 1.0
            v[i].y = 0.0

    ## Add the user-input spring force
    for i in ti.ndrange(num_pins[None]):
        v[pins[i]] = ti.Vector([0,0])

    # Spring force applied to a single vertex
    cur_spring_len = tm.sqrt(tm.dot(spring_force[0],spring_force[0]))
    if force_idx[None] > 0:
        v[force_idx[None]] += dh*kspring*spring_force[0]/(m)*cur_spring_len

    if currmode == 2:
        # TODO: Resolve Collision
        for i in range(N):
            for j in range(house.nboundary):
                bp = house.boundaries.p[j]    # boundary point
                bn = house.boundaries.n[j]    # boundary normal
                eps = house.boundaries.eps[j]
                # Compute signed distance from vertex to boundary
                d = (x[i] - bp).dot(bn)
                if d < eps:
                    # Vertex penetrates
                    correction = eps - d
                    # Project the vertex out along the normal
                    x[i] += correction * bn
                    # Remove the normal component of the velocity
                    v[i] -= (v[i].dot(bn)) * bn

##############################################################

## GUI

per_vertex_color = ti.Vector.field(3, ti.f32, shape=N)
pin_color = ti.Vector([0,1,0])

# For drawing the line indicating force direction/magnitude
draw_force_vertices = ti.Vector.field(2, shape=2, dtype=ti.f32)
draw_force_indices = ti.Vector.field(2, shape=1, dtype=int)
draw_force_indices[0] = ti.Vector([0,1])

# Parameters for interface
# Mode == 0: Edit
# Mode == 1: Simulate
# Mode == 2: Collision
cur_mode_idx = 0

@ti.kernel
def reset_user_drag():
    force_idx[None] = -1
    spring_force[0] = ti.Vector([0,0])

def reset_state():
    ModelSelector[None] = 'cvn'.find(model)
    x.from_numpy(va)
    for i in range(N):
        v[i] = ti.Vector([0,0])
    reset_user_drag()
    initialize_D0()
    initialize_area0()
    update_lame_parameters()

@ti.kernel
def reset_pins():
    for i in range(num_pins[None]):
        per_vertex_color[pins[i]] = ti.Vector([0, 0, 0])
        pins[i] = 0
    num_pins[None] = 0

####################################################################
# TODO: Run your initialization code 
####################################################################\

reset_state()

paused = True
dm = DistanceMap(N, x)
window = ti.ui.Window("Linear FEM", (600, 600))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))

while window.running:
    if window.is_pressed(ti.ui.LMB):
        if not draw_force and (cur_mode_idx == 1 or cur_mode_idx == 2):
            draw_force = True
            force_idx[None] = dm.get_closest_vertex(ti.Vector(window.get_cursor_pos()))
            if force_idx[None] > 0:
                force_start_pos = np.array([x[force_idx[None]][0],
                    x[force_idx[None]][1]])
                force_end_pos = force_start_pos
        elif (cur_mode_idx == 1 or cur_mode_idx == 2):
            if force_idx[None] > 0:
                force_start_pos = np.array([x[force_idx[None]][0],
                    x[force_idx[None]][1]])
                force_end_pos = np.array(window.get_cursor_pos())
                spring_force[0] = ti.Vector(force_end_pos-force_start_pos)
                draw_force_vertices.from_numpy(
                        np.stack((force_start_pos,force_end_pos))
                        .astype(np.float32))
                canvas.lines(vertices=draw_force_vertices,
                        indices=draw_force_indices,
                        color=(1,0,0),width=0.002)
        

    for e in window.get_events(ti.ui.RELEASE):
        if e.key == ti.ui.LMB and (cur_mode_idx == 1 or cur_mode_idx == 2):
            reset_user_drag()
            draw_force = False

    for e in window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'v' or e.key == 'c' or e.key == 'n':
            prev_model = model
            model = e.key
            if prev_model != model:
                reset_state()
        elif e.key == 'm' or e.key == 'M':
            if cur_mode_idx == 2:
                cur_mode_idx = 0
                draw_force = False
                reset_state()
            else:
                cur_mode_idx = (cur_mode_idx + 1) % 2
        elif e.key == 't' or e.key == 'T':
            if cur_mode_idx == 2:
                cur_mode_idx = 0
                reset_state()
                draw_force = False
            else:
                cur_mode_idx = 2
                reset_pins()
                reset_state()
                draw_force = False
        elif e.key == ti.GUI.LMB and cur_mode_idx == 0:
            vertex_idx = dm.get_closest_vertex(ti.Vector(window.get_cursor_pos()))
            is_pinned = False
            pin_idx = -1
            if vertex_idx >= 0:
                for i in range(num_pins[None]):
                    if pins[i] == vertex_idx:
                        is_pinned = True
                        pin_idx = i
                        break
                if not is_pinned:
                    pins[num_pins[None]] = vertex_idx
                    num_pins[None] += 1
                    per_vertex_color[vertex_idx] = pin_color
                else:
                    for i in range(pin_idx, num_pins[None]-1):
                        pins[i] = pins[i+1]
                    num_pins[None] -= 1
                    per_vertex_color[vertex_idx] = ti.Vector([0,0,0])

        elif e.key =='r' or e.key == 'R':
            if cur_mode_idx == 0:
                reset_pins()
            else:
                reset_state()

        elif e.key == ti.GUI.SPACE:
            paused = not paused

    ##############################################################
    if not paused:
        # TODO: run all of your simulation code here
        for i in range(substepping):
            timestep(cur_mode_idx)
            pass
    ##############################################################

    # Update the distance map
    max_val = 0.0
    for i in range(N):
        if vertex_strain[i] > max_val:
            max_val = vertex_strain[i]
    # Avoid division by zero
    if max_val < 1e-8:
        max_val = 1e-8
    for i in range(N):
        c = vertex_strain[i] / max_val
        per_vertex_color[i] = ti.Vector([c, 0.0, 1.0 - c])  # (blue-to-red)
    
    # Override pinned vertices to show green
    for i in range(num_pins[None]):
        per_vertex_color[pins[i]] = pin_color

    # Draw wireframe of mesh
    canvas.lines(vertices=x, indices=edges, width=0.002, color=(0,0,0))

    canvas.triangles(vertices=x, indices=triangles, per_vertex_color=per_vertex_color)

    canvas.circles(x, per_vertex_color=per_vertex_color, radius=0.005)

    # Draw the gingerbread house if we switch to collision testing mode
    if cur_mode_idx == 2:
        canvas.lines(house.boundaries.p, width=0.01, indices=house.boundary_indices, color=(0.4, 0.2, 0.0))

    # text
    gui = window.get_gui()
    with gui.sub_window("Controls", 0.02, 0.02, 0.4, 0.25):
        if model == 'c':
            gui.text('Co-rotated linear model')
        elif model == 'v':
            gui.text('Venant-Kirchhoff model')
        else:
            gui.text('Neo-Hookean model')

        if cur_mode_idx == 0:
            gui.text('Edit mode: Click to pin/unpin vertices')
        elif cur_mode_idx == 1:
            gui.text('Simulation mode: Drag to create forces')
        else:
            gui.text('Collision mode: Drag to move the deformable')

        gui.text('Press \'c,v,n\' to switch model')
        gui.text('Press \'m\' to switch mode')
        gui.text('Press \'t\' to switch between two simulation scenes')
        gui.text('Press \'r\' to: ')
        gui.text('* Clear all pinned points (edit mode)')
        gui.text('* Reset to initial state (simulation mode)')
        gui.text('Press \'SPACE\' to pause/unpause')

        if paused:
            gui.text('Simulation PAUSED')

    window.show()