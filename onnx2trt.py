import tensorrt as trt
import argparse


MAX_BATCH = 64

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, help='Path to the ONNX file')
    parser.add_argument('--engine', type=str, help='Path to output the engine file')
    parser.add_argument('--fp16', default=False, action='store_true', help='Use FP16 precision')
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as model:
        parser.parse(model.read())

    config = builder.create_builder_config()
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if args.fp16:
       config.set_flag(trt.BuilderFlag.FP16)
    else:
       config.set_flag(trt.BuilderFlag.TF32)

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        input_name = network.get_input(i).name
        input_shape = network.get_input(i).shape
        min_shape = [1 if dim == -1 else dim for dim in input_shape]
        opt_shape = [1 if dim == -1 else dim for dim in input_shape]
        max_shape = [MAX_BATCH if dim == -1 else dim for dim in input_shape]
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build the engine.")
    else:
        with open(args.engine, "wb") as f:
            f.write(serialized_engine)