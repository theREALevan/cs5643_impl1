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
##############################################################

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


Dm = ti.Matrix.field(2, 2, ti.f32, N_triangles)  
F = ti.Matrix.field(2, 2, ti.f32, N_triangles)  

@ti.kernel
def compute_Dm():
    for i in range(N_triangles):
        v0, v1, v2 = triangles[i]
        X0 = x[v0]
        Dm[i] = ti.Matrix.cols([x[v1] - X0, x[v2] - X0])

compute_Dm() 

# stress tensor
P = ti.Matrix.field(2, 2, ti.f32, N_triangles) 

# TODO: Put additional fields here for storing D (etc.)
# TODO: Implement the initialization and timestepping kernel for the deformable object
@ti.kernel
def timestep(currmode: int):
    # Compute Lame parameters (updated 2D formula)
    nu = PoissonsRatio[None]
    mu = YoungsModulus[None] / (2 * (1 + nu))
    lmbda = (YoungsModulus[None] * nu) / ((1 + nu) * (1 - nu))
    
    # Existing F calculation
    for i in range(N_triangles):
        v0, v1, v2 = triangles[i]
        x0 = x[v0]
        Ds = ti.Matrix.cols([x[v1] - x0, x[v2] - x0])
        F[i] = Ds @ Dm[i].inverse()
        
        if ModelSelector[None] == 0:  # Corotated
            R, S = ti.polar_decompose(F[i])
            epsilon_c = S - ti.Matrix.identity(ti.f32, 2)
            trace_epsilon = epsilon_c.trace()
            P[i] = R @ (2*mu*epsilon_c + lmbda*trace_epsilon*ti.Matrix.identity(ti.f32, 2))
        elif ModelSelector[None] == 1:  # StVK
            E = 0.5*(F[i].transpose() @ F[i] - ti.Matrix.identity(ti.f32, 2))
            P[i] = F[i] @ (2*mu*E + lmbda*E.trace()*ti.Matrix.identity(ti.f32, 2))
        elif ModelSelector[None] == 2:  # Neo-Hookean
            J = F[i].determinant()
            FinvT = F[i].inverse().transpose()
            P[i] = mu*(F[i] - FinvT) + lmbda*ti.log(J)*FinvT
            
    compute_forces()

    for i in range(N):
        force[i] += m * gravity
    
    forces_np = force.to_numpy()
    for i in range(N):
        print(forces_np[i])
    
    ti.sync()  # Ensure kernel execution completes before continuing
    
        
            
    ## Add the user-input spring force
    for i in ti.ndrange(num_pins[None]):
        v[pins[i]] = ti.Vector([0,0])

    # Spring force applied to a single vertex
    cur_spring_len = tm.sqrt(tm.dot(spring_force[0],spring_force[0]))
    if force_idx[None] > 0:
        v[force_idx[None]] += dh*kspring*spring_force[0]/(m)*cur_spring_len

    if currmode == 2:
        # TODO: Resolve Collision
        pass
    

    
    for i in range(N):
        v[i] += dh * force[i] / m
        x[i] += dh * v[i]

    
    
@ti.kernel
def compute_forces():
    # Reset forces
    for i in range(N):
        force[i] = ti.Vector([0.0, 0.0])

    # Loop over all triangles to compute internal elastic forces
    for e in range(N_triangles):
        v0, v1, v2 = triangles[e]
        
        # Compute inverse of Dm
        Dm_inv = Dm[e].inverse()
        
        # Compute dF/dx (gradient of F w.r.t. vertex positions)
        dF_dx = ti.Matrix.cols([
            [-Dm_inv[0, 0] - Dm_inv[1, 0], Dm_inv[0, 0], Dm_inv[1, 0]],
            [-Dm_inv[0, 1] - Dm_inv[1, 1], Dm_inv[0, 1], Dm_inv[1, 1]]
        ])
        
        # Compute area of the triangle
        Ae = 0.5 * abs(Dm[e].determinant())
        
        # Compute force contribution from this triangle
        P_local = P[e]  # First Piola-Kirchhoff stress tensor
        H = -Ae * P_local @ dF_dx  # Elemental force matrix
        
        # Map elemental forces to vertices
        force[v0] += ti.Vector([H[0, 0], H[1, 0]])
        force[v1] += ti.Vector([H[0, 1], H[1, 1]])
        force[v2] += ti.Vector([H[0, 2], H[1, 2]])
        
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

@ti.kernel
def reset_pins():
    for i in range(num_pins[None]):
        per_vertex_color[pins[i]] = ti.Vector([0, 0, 0])
        pins[i] = 0
    num_pins[None] = 0

####################################################################
# TODO: Run your initialization code 
####################################################################

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

    # Draw wireframe of mesh
    canvas.lines(vertices=x,indices=edges,width=0.002,color=(0,0,0))

    # Draw a circle at each vertex
    # Some of them are highlighted / pinned
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