import tensorrt as trt
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, help='Path to the ONNX file')
    parser.add_argument('--engine', type=str, help='Path to output the engine file')
    parser.add_argument('--fp16', default=False, action='store_true', help='Use FP16 precision')
    parser.add_argument('--dynamic_batch', default=False, action='store_true', help='Use dynamic batch size')
    parser.add_argument('--min_batch', type=int, default=1, help='Minimum batch size')
    parser.add_argument('--opt_batch', type=int, default=False, help='Optimum batch size')
    parser.add_argument('--max_batch', type=int, default=64, help='Maximum batch size')
    args = parser.parse_args()

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(args.onnx, "rb") as model:
        parser.parse(model.read())

    config = builder.create_builder_config()
    if args.fp16:
       config.set_flag(trt.BuilderFlag.FP16)

    if args.dynamic_batch:
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            input_name = network.get_input(i).name
            input_shape = network.get_input(i).shape
            min_shape = (args.min_batch, *input_shape[1:])
            opt_shape = (args.opt_batch, *input_shape[1:])
            max_shape = (args.max_batch, *input_shape[1:])
            profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build the engine.")
    else:
        with open(args.engine, "wb") as f:
            f.write(serialized_engine)

if __name__ == '__main__':
    main()