import taichi as ti
import taichi.math as tm
import numpy as np
from pywavefront import Wavefront
from util import *

ti.init(arch=ti.vulkan)

# Models
# 1: Co-rotated linear model
# 2: St. Venant Kirchhoff model ( StVK )
# 3: Neo-Hookean model
model = 'c'
prev_model = model
damping_toggle = ti.field(ti.i32, ())
damping_toggle[None] = 1
ModelSelector = ti.field(ti.i32, ())
ModelSelector[None] = 0

# physical quantities
YoungsModulus = ti.field(ti.f32, ())
PoissonsRatio = ti.field(ti.f32, ())
# Default values for reproducing the reference
YoungsModulus[None] = 1e2
PoissonsRatio[None] = 0.5

##############################################################
# TODO: Put additional parameters here
# e.g. Lame parameters 
Lame_mu = ti.field(ti.f32, ())
Lame_lambda = ti.field(ti.f32, ())

@ti.kernel
def update_lame_parameters():
    Lame_mu[None] = YoungsModulus[None] / (2.0 * (1.0 + PoissonsRatio[None]))
    Lame_lambda[None] = YoungsModulus[None] * PoissonsRatio[None] / ((1.0 + PoissonsRatio[None]) * (1.0 - PoissonsRatio[None]))
##############################################################

## Load geometry of the test scenes
rect_obj = Wavefront("models/rect.obj", collect_faces=True)
rect_obj_stretch = Wavefront("models/rect_stretch.obj", collect_faces=True)
rect_obj_compress = Wavefront("models/rect_compress.obj", collect_faces=True)

va = np.array(rect_obj.vertices, dtype=np.float32)[:,:2]
va_stretch = np.array(rect_obj_stretch.vertices, dtype=np.float32)[:,:2]
va_compress = np.array(rect_obj_compress.vertices, dtype=np.float32)[:,:2]

# The objs have the exact same topology, e.g. the meshes are the same
mesh = rect_obj.mesh_list[0]
faces = np.array(mesh.faces, dtype=np.int32)
mesh_triangles = ti.field(int, shape=np.prod(faces.shape))
mesh_triangles.from_numpy(faces.ravel())

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
# damping parameter
k_drag = 10 * m

# Simulation components: x and v
x_rest = ti.Vector.field(2, shape=N, dtype=ti.f32)
# x_rest is initialized directly to the mesh vertices (in 2D)
# and remains unchanged.
x_rest.from_numpy(va)

x_stretch = ti.Vector.field(2, shape=N, dtype=ti.f32)
x_stretch.from_numpy(va_stretch)

x_compress = ti.Vector.field(2, shape=N, dtype=ti.f32)
x_compress.from_numpy(va_compress)

# Deformed shape
x = ti.Vector.field(2, shape=N, dtype=ti.f32)
v = ti.Vector.field(2, ti.f32, N)

# Pinned indices
# We pin the top and botton edges of the rectangle (known a priori)
pins = ti.field(ti.i32, N)
num_pins = ti.field(ti.i32, ())
num_pins[None] = 40
for i in range(int(0.5*num_pins[None])):
    pins[2*i] = 40*i
    pins[2*i+1] = 40*i+39

is_pinned = ti.field(ti.i32, shape=N)

@ti.kernel
def initialize_pinned_mask():
    for i in range(N):
        is_pinned[i] = 0
    for j in range(num_pins[None]):
        is_pinned[pins[j]] = 1


# TODO: Put additional fields here for storing D (etc.)
D0 = ti.Matrix.field(2, 2, ti.f32, shape=N_triangles)
area0 = ti.field(ti.f32, shape=N_triangles)

@ti.func
def compute_D(p0, p1, p2):
    return ti.Matrix.cols([p1 - p0, p2 - p0])

@ti.kernel
def initialize_D0():
    # Compute D0 from x_rest
    for t in range(N_triangles):
        i0 = triangles[t][0]
        i1 = triangles[t][1]
        i2 = triangles[t][2]
        D0[t] = compute_D(x_rest[i0], x_rest[i1], x_rest[i2])

@ti.kernel
def initialize_area0():
    for t in range(N_triangles):
        area0[t] = 0.5 * ti.abs(D0[t].determinant())

@ti.func
def compute_F(t: ti.i32) -> ti.Matrix:
    i0 = triangles[t][0]
    i1 = triangles[t][1]
    i2 = triangles[t][2]
    # Compute current D from deformed shape x
    D = compute_D(x[i0], x[i1], x[i2])
    return D @ D0[t].inverse()

@ti.func
def polar_decomposition(F):
    u, s, v = ti.svd(F)
    R = u @ v.transpose()
    S = R.transpose() @ F
    return R, S

@ti.func
def compute_P(F):
    I2 = ti.Matrix([[1.0,0.0],[0.0,1.0]])
    P_ret = ti.Matrix.zero(ti.f32, 2,2)
    if ModelSelector[None] == 0:
        R, S = polar_decomposition(F)
        E_c = S - I2
        P_ret = R @ (2.0 * Lame_mu[None] * E_c + Lame_lambda[None] * E_c.trace() * I2)
    elif ModelSelector[None] == 1:
        E_green = 0.5*(F.transpose()@F - I2)
        P_ret = F @ (2.0 * Lame_mu[None] * E_green + Lame_lambda[None] * E_green.trace() * I2)
    else:
        detF = F.determinant()
        invT = F.inverse().transpose()
        P_ret = Lame_mu[None]*(F - invT) + Lame_lambda[None]*ti.log(detF)*invT
    return P_ret

forces = ti.Vector.field(2, ti.f32, shape=N)

# TODO: Implement the initialization and timestepping kernel for the deformable object
@ti.kernel
def timestep():
    # Clear forces for all vertices
    for i in range(N):
        forces[i] = ti.Vector([0.0, 0.0])
    
    # Compute forces on all triangles
    for t in range(N_triangles):
        i0 = triangles[t][0]
        i1 = triangles[t][1]
        i2 = triangles[t][2]
        F = compute_F(t)
        P = compute_P(F)
        invT = D0[t].inverse().transpose()
        A_e = area0[t]
        H = -A_e * (P @ invT)
        col0 = ti.Vector([H[0,0], H[1,0]])
        col1 = ti.Vector([H[0,1], H[1,1]])
        forces[i1] += col0
        forces[i2] += col1
        forces[i0] += -(col0 + col1)
    
    # Update velocity and position only for non-pinned vertices
    for i in range(N):
        # Skip update if vertex is pinned
        if is_pinned[i] == 1:
            continue

        if damping_toggle[None]:
            v[i] -= v[i] * k_drag / m * dh
        v[i] += dh * (forces[i] / m)
        x[i] += dh * v[i]

##############################################################

is_stretch = ti.field(ti.i32, ())
is_stretch[None] = 1

def reset_state():
    ModelSelector[None] = 'cvn'.find(model)
    initialize()
    update_lame_parameters()
    initialize_D0()
    initialize_area0()
    initialize_pinned_mask()

@ti.kernel
def initialize():
    if is_stretch[None] == 1:
        for i in range(N):
            x[i] = x_stretch[i]
    else:
        for i in range(N):
            x[i] = x_compress[i]

    for i in range(N):
        v[i] = ti.Vector([0,0])

# initialize system state
initialize()

####################################################################
# TODO: Run your initialization code 
####################################################################

reset_state()

paused = False
window = ti.ui.Window("Linear FEM", (600, 600))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))

while window.running:
    for e in window.get_events(ti.ui.PRESS):
        if e.key in [ti.ui.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == 'v' or e.key == 'c' or e.key == 'n':
            prev_model = model
            model = e.key
            if prev_model != model:
                reset_state()
        elif e.key =='d' or e.key == 'D':
            damping_toggle[None] = not damping_toggle[None]

        elif e.key == ti.GUI.UP:
            is_stretch[None] = 1
            initialize()

        elif e.key == ti.GUI.DOWN:
            is_stretch[None] = 0
            initialize()

        elif e.key == ti.GUI.SPACE:
            paused = not paused

    ##############################################################
    if not paused:
        # TODO: run all of your simulation code here
        for i in range(substepping):
            timestep()
        pass
    ##############################################################

    # Draw wireframe of mesh
    canvas.lines(vertices=x,indices=edges,width=0.002,color=(0,0,0))

    # text
    gui = window.get_gui()
    with gui.sub_window("Controls", 0.02, 0.02, 0.4, 0.2):
        if model == 'c':
            gui.text('Co-rotated linear model')
        elif model == 'v':
            gui.text('Venant-Kirchhoff model')
        else:
            gui.text('Neo-Hookean model')

        gui.text('Press \'c,v,n\' to switch model')
        gui.text('Press up to test stretch mode')
        gui.text('Press down to test compression mode')
        gui.text('Press \'SPACE\' to pause/unpause')
        gui.text('Press \'d\' to toggle damping')

        if damping_toggle[None]:
            gui.text('D: Damping On')
        else:
            gui.text('D: Damping Off')

        if paused:
            gui.text('Simulation PAUSED')

    window.show()