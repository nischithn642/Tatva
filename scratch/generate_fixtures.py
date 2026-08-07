import os
import numpy as np
import onnx
from onnx import helper, TensorProto
from onnxruntime.quantization import quantize_dynamic, QuantType

def generate():
    os.makedirs("models", exist_ok=True)
    
    # 1. Baseline Model
    input_ids = helper.make_tensor_value_info('input_ids', TensorProto.INT64, [1, 'seq_len'])
    attention_mask = helper.make_tensor_value_info('attention_mask', TensorProto.INT64, [1, 'seq_len'])
    token_type_ids = helper.make_tensor_value_info('token_type_ids', TensorProto.INT64, [1, 'seq_len'])
    last_hidden_state = helper.make_tensor_value_info('last_hidden_state', TensorProto.FLOAT, [1, 'seq_len', 5])

    np.random.seed(42)
    emb_data = np.random.randn(128, 32).astype(np.float32)
    embeddings = helper.make_tensor('embeddings', TensorProto.FLOAT, [128, 32], emb_data.flatten())
    w_data = np.random.randn(32, 5).astype(np.float32)
    w = helper.make_tensor('W', TensorProto.FLOAT, [32, 5], w_data.flatten())
    b_data = np.random.randn(5).astype(np.float32)
    b = helper.make_tensor('B', TensorProto.FLOAT, [5], b_data.flatten())

    gather_node = helper.make_node('Gather', ['embeddings', 'input_ids'], ['gathered'], axis=0)
    cast_mask = helper.make_node('Cast', ['attention_mask'], ['mask_f'], to=TensorProto.FLOAT)
    cast_tok = helper.make_node('Cast', ['token_type_ids'], ['tok_f'], to=TensorProto.FLOAT)
    add_inputs = helper.make_node('Add', ['mask_f', 'tok_f'], ['inputs_sum'])
    unsqueeze_inputs = helper.make_node('Unsqueeze', ['inputs_sum'], ['inputs_sum_3d'], axes=[2])
    add_gather = helper.make_node('Add', ['gathered', 'inputs_sum_3d'], ['gather_with_inputs'])
    matmul_node = helper.make_node('MatMul', ['gather_with_inputs', 'W'], ['matmul_out'])
    add_bias = helper.make_node('Add', ['matmul_out', 'B'], ['add_out'])
    softmax_node = helper.make_node('Softmax', ['add_out'], ['last_hidden_state'], axis=-1)

    graph = helper.make_graph(
        [gather_node, cast_mask, cast_tok, add_inputs, unsqueeze_inputs, add_gather, matmul_node, add_bias, softmax_node],
        'synthetic_baseline',
        [input_ids, attention_mask, token_type_ids],
        [last_hidden_state],
        initializer=[embeddings, w, b]
    )

    opset = helper.make_operatorsetid('', 11)
    model = helper.make_model(graph, producer_name='tatva', opset_imports=[opset])
    onnx.save(model, "models/model.onnx")

    # 2. Quantized Model
    quantize_dynamic(
        model_input="models/model.onnx",
        model_output="models/model_quant.onnx",
        weight_type=QuantType.QInt8
    )

    # 3. Unsupported Op Model
    # We change the Softmax node's op_type to UnsupportedOpXYZ
    softmax_node_bad = helper.make_node('UnsupportedOpXYZ', ['add_out'], ['last_hidden_state'])
    graph_bad = helper.make_graph(
        [gather_node, cast_mask, cast_tok, add_inputs, unsqueeze_inputs, add_gather, matmul_node, add_bias, softmax_node_bad],
        'synthetic_unsupported',
        [input_ids, attention_mask, token_type_ids],
        [last_hidden_state],
        initializer=[embeddings, w, b]
    )
    model_bad = helper.make_model(graph_bad, producer_name='tatva', opset_imports=[opset])
    onnx.save(model_bad, "models/model_unsupported.onnx")
    
    print("Model fixtures generated successfully under models/")

if __name__ == "__main__":
    generate()
