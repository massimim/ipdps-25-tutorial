import numpy as np
import matplotlib.pyplot as plt
import warp as wp
from pyevtk.hl import imageToVTK

# Initialize Warp
wp.init()

# ---------------------
# Grid and simulation parameters
# ---------------------
nx, ny      = 128, 128
num_steps   = 2000

# Desired Reynolds number
Re            = 1e5
lid_velocity  = 0.2     # lid speed in lattice units
L             = nx - 1  # cavity length in lattice units

# Compute lattice‐viscosity and relaxation time
nu  = lid_velocity * L / Re
tau = 3.0 * nu + 0.5

print(f"Re={Re:.0f}, U={lid_velocity:.3f}, L={L} → ν={nu:.3e}, τ={tau:.6f}")

# Total cells and distribution entries
N = nx * ny
F = 9 * N

# ---------------------
# Host‐side lookup tables
# ---------------------
w_host        = np.array([4/9] + [1/9]*4 + [1/36]*4, dtype=np.float32)
cx_host       = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
cy_host       = np.array([0, 0, 1,  0, -1, 1,  1, -1, -1], dtype=np.int32)
opposite_host = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# ---------------------
# Upload lookup tables to GPU
# ---------------------
w        = wp.array(w_host,        dtype=wp.float32, device="cuda")
cx       = wp.array(cx_host,       dtype=wp.int32,   device="cuda")
cy       = wp.array(cy_host,       dtype=wp.int32,   device="cuda")
opposite = wp.array(opposite_host, dtype=wp.int32,   device="cuda")

# ---------------------
# Equilibrium function
# ---------------------
@wp.func
def feq(d: int,
        rho: float,
        ux: float,
        uy: float,
        w:  wp.array(dtype=wp.float32),
        cx: wp.array(dtype=wp.int32),
        cy: wp.array(dtype=wp.int32)) -> float:

    cxi = float(cx[d])
    cyi = float(cy[d])
    cu  = 3.0 * (cxi * ux + cyi * uy)
    u2  = ux*ux + uy*uy
    return w[d] * rho * (1.0 + cu + 0.5*cu*cu - 1.5*u2)

# ---------------------
# Kernels
# ---------------------
@wp.kernel
def collide(f:   wp.array(dtype=wp.float32),  # length F = 9*N
            rho: wp.array(dtype=wp.float32),  # length N
            ux:  wp.array(dtype=wp.float32),  # length N
            uy:  wp.array(dtype=wp.float32),  # length N
            w:   wp.array(dtype=wp.float32),
            cx:  wp.array(dtype=wp.int32),
            cy:  wp.array(dtype=wp.int32),
            nx:  int,
            ny:  int,
            tau: float):

    i, j = wp.tid()
    if i >= nx or j >= ny:
        return

    idx     = j*nx + i
    rho_val = 0.0
    ux_val  = 0.0
    uy_val  = 0.0

    # compute macroscopic density & velocity
    for d in range(9):
        di   = d*nx*ny + idx
        fi   = f[di]
        rho_val += fi
        ux_val  += fi * float(cx[d])
        uy_val  += fi * float(cy[d])

    ux_val /= rho_val
    uy_val /= rho_val

    rho[idx] = rho_val
    ux[idx]  = ux_val
    uy[idx]  = uy_val

    # BGK collision
    for d in range(9):
        di      = d*nx*ny + idx
        feq_v   = feq(d, rho_val, ux_val, uy_val, w, cx, cy)
        f[di]   = f[di] - (f[di] - feq_v)/tau

@wp.kernel
def stream(f_in:  wp.array(dtype=wp.float32),  # length F
           f_out: wp.array(dtype=wp.float32),  # length F
           cx:    wp.array(dtype=wp.int32),
           cy:    wp.array(dtype=wp.int32),
           nx:    int,
           ny:    int):

    i, j = wp.tid()
    if i >= nx or j >= ny:
        return

    for d in range(9):
        ii  = wp.clamp(i - int(cx[d]), 0, nx - 1)
        jj  = wp.clamp(j - int(cy[d]), 0, ny - 1)
        src = d*nx*ny + jj*nx + ii
        dst = d*nx*ny + j*nx  + i
        f_out[dst] = f_in[src]

@wp.kernel
def bounce_back(f:        wp.array(dtype=wp.float32),  # length F
                opposite: wp.array(dtype=wp.int32),
                nx:       int,
                ny:       int):

    i, j = wp.tid()
    if i >= nx or j >= ny:
        return

    idx = j*nx + i
    if i==0 or i==nx-1 or j==0 or j==ny-1:
        for d in range(9):
            d_opp = opposite[d]
            di     = d*nx*ny   + idx
            di_opp = d_opp*nx*ny + idx
            tmp    = f[di]
            f[di]  = f[di_opp]
            f[di_opp] = tmp

@wp.kernel
def apply_lid(f:        wp.array(dtype=wp.float32),  # length F
              rho:      wp.array(dtype=wp.float32),  # length N
              opposite: wp.array(dtype=wp.int32),
              w:        wp.array(dtype=wp.float32),
              cx:       wp.array(dtype=wp.int32),
              cy:       wp.array(dtype=wp.int32),
              nx:       int,
              ny:       int,
              u_lid:    float):

    i = wp.tid()
    if i >= nx:
        return

    j   = ny - 1
    idx = j*nx + i

    for d in range(9):
        d_opp = opposite[d]
        feq_v = feq(d_opp, rho[idx], u_lid, 0.0, w, cx, cy)
        f[d*nx*ny + idx] = feq_v

# ---------------------
# Initialization (flattened buffers)
# ---------------------
f_host = np.zeros((9, ny, nx), dtype=np.float32)
for d in range(9):
    f_host[d, :, :] = w_host[d]
f_flat   = f_host.reshape(-1)

rho_flat = np.ones((ny*nx,), dtype=np.float32)
ux_flat  = np.zeros((ny*nx,), dtype=np.float32)
uy_flat  = np.zeros((ny*nx,), dtype=np.float32)

f      = wp.array(f_flat,     dtype=wp.float32, device="cuda")
f_temp = wp.array(f_flat.copy(), dtype=wp.float32, device="cuda")
rho    = wp.array(rho_flat,   dtype=wp.float32, device="cuda")
ux     = wp.array(ux_flat,    dtype=wp.float32, device="cuda")
uy     = wp.array(uy_flat,    dtype=wp.float32, device="cuda")

# ---------------------
# Main time-stepping
# ---------------------
for step in range(num_steps):
    wp.launch(collide,     dim=(nx, ny),
              inputs=[f, rho, ux, uy, w, cx, cy, nx, ny, tau],
              device="cuda")

    wp.launch(stream,      dim=(nx, ny),
              inputs=[f, f_temp, cx, cy, nx, ny],
              device="cuda")

    wp.launch(bounce_back, dim=(nx, ny),
              inputs=[f_temp, opposite, nx, ny],
              device="cuda")

    wp.launch(apply_lid,   dim=(nx,),
              inputs=[f_temp, rho, opposite, w, cx, cy, nx, ny, lid_velocity],
              device="cuda")

    # swap buffers
    f, f_temp = f_temp, f

# ---------------------
# Compute final velocity magnitude
# ---------------------
u   = ux.numpy().reshape((ny, nx))
v   = uy.numpy().reshape((ny, nx))
vel = np.sqrt(u**2 + v**2)

# ---------------------
# Save JPEG of final field
# ---------------------
plt.figure()
plt.imshow(vel[::-1, :], cmap="plasma", origin="lower")
plt.title(f"Final Velocity Magnitude (Re={Re:.0f})")
plt.colorbar()
plt.savefig("velocity_final.jpg", format="jpeg", dpi=300)
print("→ saved velocity_final.jpg")

# ---------------------
# Export VTI for ParaView
# ---------------------
# reshape to (nx, ny, 1) with x fastest
vel3d = vel.T.reshape((nx, ny, 1))

imageToVTK(
    "lbm_output",
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    pointData={"velocity": vel3d}
)
print("→ exported lbm_output.vti")