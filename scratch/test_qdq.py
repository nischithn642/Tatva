import tvm
from tvm import relax
from tvm.script import relax as R

@tvm.script.ir_module
class InputModule:
    @R.function
    def main(x: R.Tensor((1, 32), "float32")) -> R.Tensor((1, 32), "float32"):
        # Just return x
        return x

def main():
    mod = InputModule
    print("Original module:")
    print(mod.script())

    from tvm.relax.op import quantize, dequantize
    
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((1, 32), "float32"))
    with bb.function("main", [x]):
        scale = relax.const(0.1, "float32")
        zp = relax.const(0, "int32")
        q = quantize(x, scale, zp, out_dtype="int8")
        dq = dequantize(q, scale, zp, out_dtype="float32")
        bb.emit_func_output(dq)
    
    new_mod = bb.finalize()
    print("New module:")
    print(new_mod.script())

    # Legalize
    seq = tvm.transform.Sequential([relax.transform.LegalizeOps()])
    legal_mod = seq(new_mod)
    print("Legalized:")
    print(legal_mod.script())

    # Build
    lib = relax.build(legal_mod, target="c")
    print("C Code:")
    print(lib.mod.imports[0].inspect_source()[:500])

if __name__ == "__main__":
    main()
