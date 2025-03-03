import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

import tensorrt as trt
import yaml

TRT_LOGGER = trt.Logger()
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

def allocate_buffers_encoder(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    out_shapes = []
    input_shapes = []
    out_names = []
    max_batch_size = engine.get_profile_shape(0, 0)[2][0]
    print('Profile shape: ', engine.get_profile_shape(0, 0))
    for binding in engine:
        binding_shape = engine.get_tensor_shape(binding)
        print('binding:', binding, '- binding_shape:', binding_shape)
        if binding == 'input':
            max_width = engine.get_profile_shape(0, 0)[2][3]
            max_height = engine.get_profile_shape(0, 0)[2][2]
            size = max_batch_size * max_width * max_height * 3
        else:
            size = max_batch_size * binding_shape[0] * binding_shape[2]

        dtype = trt.nptype(engine.get_tensor_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if binding == 'input':
            inputs.append(HostDeviceMem(host_mem, device_mem))
            input_shapes.append(engine.get_tensor_shape(binding))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
            #Collect original output shapes and names from engine
            out_shapes.append(binding_shape)
            out_names.append(binding)
    return inputs, outputs, bindings, stream, input_shapes, out_shapes, out_names, max_batch_size



# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class EncoderTrtModel():
    def __init__(self, model, max_size):
        self.engine_file = model
        self.engine = None
        self.inputs = None
        self.outputs = None
        self.bindings = None
        self.stream = None
        self.context = None
        self.input_shapes = None
        self.out_shapes = None
        self.max_batch_size = 1
        self.max_size = max_size

    def build(self):
        with open(self.engine_file, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        # Allocate
        print('Maximum image size: {}x{}'.format(self.max_size, self.max_size))
        self.inputs, self.outputs, self.bindings, self.stream, self.input_shapes, self.out_shapes, self.out_names, self.max_batch_size = \
                allocate_buffers_encoder(self.engine)
        self.context = self.engine.create_execution_context()
        self.context.active_optimization_profile = 0

    def run(self, input, deflatten: bool = True, as_dict = False):
        # lazy load implementation
        if self.engine is None:
            self.build()

        input = np.asarray(input)
        batch_size, _, im_height, im_width = input.shape
        assert batch_size <= self.max_batch_size
        assert max(im_width, im_height) <= self.max_size, "Invalid shape: {}x{}, max shape: {}".format(im_width, im_height, self.max_size)
        allocate_place = np.prod(input.shape)
        self.inputs[0].host[:allocate_place] = input.flatten(order='C').astype(np.float32)
        self.context.set_binding_shape(0, input.shape)
        trt_outputs = do_inference(
            self.context, bindings=self.bindings,
            inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        # Reshape TRT outputs to original shape instead of flattened array
        if deflatten:
            out_shapes = [(self.out_shapes[0][0], batch_size, self.out_shapes[0][2])]
            trt_outputs = [output[:np.prod(shape)].reshape(shape) for output, shape in zip(trt_outputs, out_shapes)]
        if as_dict:
            return {self.out_names[ix]: trt_output[:batch_size] for ix, trt_output in enumerate(trt_outputs)}
        
        return [trt_output for trt_output in trt_outputs]


if __name__ == "__main__":
    import cv2
    bgr_img = cv2.imread("/home1/data/thaitran/Research/OCR/source/end2end_univietocr/vietocr/convert2deploy/onnx/hi.jpg")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (128, 32))
    inp = rgb_img.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0

    bgr_img = cv2.imread("/home1/data/thaitran/Research/OCR/source/end2end_univietocr/vietocr/convert2deploy/onnx/hi1.jpg")
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb_img = cv2.resize(rgb_img, (128, 32))
    inp1 = rgb_img.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
    
    batch_inp = np.zeros((2, 3, 32, 128), dtype=np.float32)
    batch_inp[0] = inp.transpose(2, 0, 1)
    batch_inp[1] = inp1.transpose(2, 0, 1)

    encoder_trt = "/home1/data/thaitran/Research/OCR/source/end2end_univietocr/vietocr/convert2deploy/trt/models/encoder_merge.trt"
    model = EncoderTrtModel(encoder_trt, 128)
    out_encoder = model.run(batch_inp)[0]
    decoder_trt = "/home1/data/thaitran/Research/OCR/source/end2end_univietocr/vietocr/convert2deploy/trt/models/modelsdecoder_merge.trt"
    # decode = DecoderTrTModel(decoder_trt)

    # for i in range(0,10):
    #     max_length = 1
    #     max_seq_length = 128
    #     eos_token = 2
    #     translated_sentence = [[1]*2]
    #     while max_length <= max_seq_length and not all(np.any(np.asarray(translated_sentence).T==eos_token, axis=1)):
    #         tgt_inp = np.array(translated_sentence)
    #         out_decoder = decode.run(tgt_inp, out_encoder)
    #         out_decoder = out_decoder.tolist()
    #         translated_sentence.append(out_decoder)
    #         max_length += 1
            
    #     translated_sentence = np.asarray(translated_sentence).T
    #     translated_sentence = translated_sentence.tolist()
    #     config = Cfg.load_config_from_file("../config.yml")
    #     vocab = config.vocab
    #     vocab = Vocab(vocab, max_target_length= 128)
    #     output_text = vocab.batch_decode(translated_sentence)
    #     print(output_text)
