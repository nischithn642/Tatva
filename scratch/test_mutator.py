import tvm
from tvm import relax
from tvm.script import relax as R

@tvm.script.ir_module
class InputModule:
    @R.function
    def main(x: R.Tensor((1, 32), "float32")) -> R.Tensor((1, 32), "float32"):
        # Let's add a dummy matmul or dense
        w = R.const(0.5, "float32")
        # We simulate a dense op
        y = R.multiply(x, w)
        return y

def main():
    mod = InputModule
    print("Original module:")
    print(mod.script())

    @relax.expr_functor.mutator
    class QuantizationMutator(relax.PyExprMutator):
        def visit_call_(self, call: relax.Call) -> relax.Expr:
            call = super().visit_call_(call)
            if isinstance(call.op, tvm.ir.Op) and call.op.name == "relax.multiply":
                arg0 = call.args[0]
                arg1 = call.args[1]
                
                scale = relax.const(0.1, "float32")
                zp = relax.const(0, "int32")
                
                # Emit via builder_
                q = self.builder_.emit(relax.op.quantize(arg0, scale, zp, out_dtype="int8"))
                dq = self.builder_.emit(relax.op.dequantize(q, scale, zp, out_dtype="float32"))
                
                return relax.Call(call.op, [dq, arg1], call.attrs, call.sinfo_args, call.span)
            return call

    mutator = QuantizationMutator(mod)
    # We mutate the function main
    mutated_func = mutator.visit_expr(mod["main"])

    # Resolve global variable for 'main'
    gv = None
    for g_var in mod.get_global_vars():
        if g_var.name_hint == "main":
            gv = g_var
            break

    bb = relax.BlockBuilder()
    bb.update_func(gv, mutated_func)
    mutated_mod = bb.finalize()

    print("Mutated module:")
    print(mutated_mod.script())

    # Legalize
    seq = tvm.transform.Sequential([relax.transform.LegalizeOps()])
    legal_mod = seq(mutated_mod)
    print("Legalized module:")
    print(legal_mod.script())

if __name__ == "__main__":
    main()
