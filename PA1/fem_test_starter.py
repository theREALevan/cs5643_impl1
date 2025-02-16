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
    edges_set.add(frozenset([faces[i][0],faces[i][1]]))
    edges_set.add(frozenset([faces[i][1],faces[i][2]]))
    edges_set.add(frozenset([faces[i][2],faces[i][0]]))

# Number of edges
N_edges = len(edges_set)
edges = ti.Vector.field(2, shape=N_edges, dtype=int)

edge_idx = 0
for edge in edges_set:
    cnt = 0
    for val in edge:
        edges[edge_idx][cnt] = val
        cnt += 1
    edge_idx += 1

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

# TODO: Put additional fields here for storing D (etc.)
# TODO: Implement the initialization and timestepping kernel for the deformable object
@ti.kernel
def timestep():
    # TODO: Sympletic integration of the internal elastic forces 

    for i in ti.ndrange(num_pins[None]):
        v[pins[i]] = ti.Vector([0,0])

    # viscous damping
    for i in v:
        if damping_toggle[None]:
            v[i] -= v[i] * k_drag / m * dh 
    
    # TODO: Sympletic integration of the internal elastic forces

##############################################################

is_stretch = ti.field(ti.i32, ())
is_stretch[None] = 1

def reset_state():
    ModelSelector[None] = 'cvn'.find(model)
    initialize()

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