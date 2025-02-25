import taichi as ti
import taichi.math as tm
import numpy as np

from scene import Scene, Init

# Have to use Vulkan arch on Mac for compatibility with GGUI
ti.init(arch=ti.vulkan)

# Collision Obstacle
obstacle = Scene(Init.CLOTH_SPHERE)
contact_eps = 1e-4
record = False

# cloth is square with n x n particles
# Use smaller n values for debugging
n = 128 
quad_size = 1.0 / (n-1)

# particles are affected by gravity
gravity = ti.Vector([0, -9.81, 0])
particle_mass = 1.0 / (n*n)

# some particles can be pinned in place
# pins are named by indices in the particle grid
# TODO: Add pinned particles
pins = [(0, 0), (0, n - 1)]

# A spherical collision object, not used until the very last part
ball_center = ti.Vector.field(3, dtype=float, shape=(1, ))
ball_center[0] = [0.5, 0, 0.5]
ball_radius = 0.3

### System state

x = ti.Vector.field(3, dtype=float, shape=(n, n))
# TODO: Create field for particle velocities
v = ti.Vector.field(3, dtype=float, shape=(n, n))

dt = 1/60000

# Set up initial state on the y  = 0.6 plane
@ti.kernel
def init_cloth():
    # TODO
    for i, j in ti.ndrange(n, n):
        x[i, j] = [i * quad_size, 0.6, j * quad_size]
        v[i, j] = [0.0, 0.0, 0.0]

# Execute a single symplectic Euler timestep
@ti.kernel
def timestep():
    # TODO: Forces for each particle
    # Compute x and v with symplectic Euler
    
    # rest lengths for spring types
    structural_rest = quad_size
    shear_rest = ti.sqrt(2.0) * quad_size
    flexion_rest = 2.0 * quad_size

    k_struct = 400
    k_shear = 100
    k_flex = 400

    k_stiff_damp = 0.0000002
    k_mass_damp = 0.0000001

    for i, j in ti.ndrange(n, n):
        is_pinned = False
        for pin in ti.static(pins):
            if i == pin[0] and j == pin[1]:
                is_pinned = True
        if is_pinned:
            continue
        
        # object is not pinned
        force = ti.Vector([0.0, 0.0, 0.0])

        # Structural springs
        for offset in ti.static([ti.Vector([1, 0]), ti.Vector([-1, 0]),
                                  ti.Vector([0, 1]), ti.Vector([0, -1])]):
            # get neighboring indices
            ni = i + offset[0]
            nj = j + offset[1]
            
            if 0 <= ni < n and 0 <= nj < n: # check in bounds
                diff = x[i, j] - x[ni, nj]
                dist = diff.norm()
                unit_dir = diff / dist
                # Spring force
                spring_force = -k_struct * (dist - structural_rest) * unit_dir
                # Stiffness damping force
                spring_damp = -k_stiff_damp * k_struct * unit_dir * (v[i, j].dot(unit_dir))
                force += spring_force + spring_damp

        # Shear springs
        for offset in ti.static([ti.Vector([1, 1]), ti.Vector([1, -1]),
                                  ti.Vector([-1, 1]), ti.Vector([-1, -1])]):
            
            # get shear indices
            ni = i + offset[0]
            nj = j + offset[1]
            
            if 0 <= ni < n and 0 <= nj < n:
                diff = x[i, j] - x[ni, nj]
                dist = diff.norm()
                unit_dir = diff / dist
                # Spring force
                spring_force = -k_shear * (dist - shear_rest) * unit_dir
                # Stiffness damping force
                spring_damp = -k_stiff_damp * k_shear * unit_dir * (v[i, j].dot(unit_dir))
                force += spring_force + spring_damp

        # Flexion springs
        for offset in ti.static([ti.Vector([2, 0]), ti.Vector([-2, 0]),
                                  ti.Vector([0, 2]), ti.Vector([0, -2])]):
            
            # get flexion indices
            ni = i + offset[0]
            nj = j + offset[1]
            
            if 0 <= ni < n and 0 <= nj < n:
                diff = x[i, j] - x[ni, nj]
                dist = diff.norm()
                unit_dir = diff / dist
                # Spring force
                spring_force = -k_flex * (dist - flexion_rest) * unit_dir
                # Stiffness damping force
                spring_damp = -k_stiff_damp * k_flex * unit_dir * (v[i, j].dot(unit_dir))
                force += spring_force + spring_damp

        # add gravitational force
        force += gravity * particle_mass

        # Add mass-proportional damping force
        force += -k_mass_damp * particle_mass * v[i, j]

        # Symplectic Euler step
        a = force / particle_mass
        v[i, j] += dt * a
        x[i, j] += dt * v[i, j]


### GUI

# Data structures for drawing the mesh
num_triangles = (n - 1) * (n - 1) * 2
indices = ti.field(int, shape=num_triangles * 3)
vertices = ti.Vector.field(3, dtype=float, shape=n * n)
colors = ti.Vector.field(3, dtype=float, shape=n * n)

# Set up the mesh
@ti.kernel
def initialize_mesh_indices():
    for i, j in ti.ndrange(n - 1, n - 1):
        quad_id = (i * (n - 1)) + j
        # 1st triangle of the square
        indices[quad_id * 6 + 0] = i * n + j
        indices[quad_id * 6 + 1] = (i + 1) * n + j
        indices[quad_id * 6 + 2] = i * n + (j + 1)
        # 2nd triangle of the square
        indices[quad_id * 6 + 3] = (i + 1) * n + j + 1
        indices[quad_id * 6 + 4] = i * n + (j + 1)
        indices[quad_id * 6 + 5] = (i + 1) * n + j

    for i, j in ti.ndrange(n, n):
        if (i % 20 < 4 and i % 20 >= 0) or (j % 20 < 4 and j % 20 >= 0):
            colors[i * n + j] = (1.0, 0.97, 0.95)
        else:
            colors[i * n + j] = (1.0, 0.2, 0.4)


# Copy vertex state into mesh vertex positions
@ti.kernel
def update_vertices():
    for i, j in ti.ndrange(n, n):
        vertices[i * n + j] = x[i, j]

result_dir = './recordings/'
video_manager = ti.tools.VideoManager(output_dir=result_dir, framerate=60, automatic_build=False)

# Create Taichi UI
scene = ti.ui.Scene()
camera = ti.ui.Camera()
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((0.6, 0.6, 1.0))
cam_pos = np.array([0.0, 0.8, 5.0])
camera.position(cam_pos[0], cam_pos[1], cam_pos[2]);
camera.lookat(0.5, 0.2, 0.5)
camera.fov(30.0)

# Initialize sim
start_t = 0.0
current_t = 0.0

init_cloth()
initialize_mesh_indices()

substeps = int((1/200) / dt)
# Run sim
for ii in range(300000):
    # TODO:
    # Call timestep() function
    # Increase current time t by dt

    for _ in range(substeps):
        timestep()
    current_t += 1/200

    update_vertices()

    scene.set_camera(camera)
    scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.mesh(vertices,
               indices=indices,
               per_vertex_color=colors,
               two_sided=True)
    
    # Uncomment this part for collision
    # scene.mesh(obstacle.verts,
    #            indices=obstacle.tris,
    #            color=(0.8, 0.7, 0.6))
    canvas.scene(scene)
    
    if record:
        img = window.get_image_buffer_as_numpy()
        video_manager.write_frame(img)
    window.show()
    
if record:
    video_manager.make_video(gif=False, mp4=True)