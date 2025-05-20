import lbm

params = lbm.Parameters()
print(params)
mem = lbm.Memory(params)
fun = lbm.Functions(params)
kernels = lbm.Kernels(params, mem, fun)
