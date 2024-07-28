import tensorrt as trt
import argparse


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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if args.fp16:
       config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)

    with open(args.engine, "wb") as f:
        f.write(serialized_engine)